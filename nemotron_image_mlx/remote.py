"""Read tensors of a Hugging Face safetensors checkpoint with HTTP range requests.

Lets ``convert`` build an MLX tier directly from the Hub without storing the
128 GB fp32 checkpoint: each tensor is fetched, converted and dropped.
"""

from __future__ import annotations

import json
import struct
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "BF16": np.uint16}


class RemoteUpstream:
    def __init__(self, repo: str, revision: str = "main", local_meta: Path | None = None, workers: int = 48):
        self.repo, self.rev = repo, revision
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self._lock = threading.Lock()
        self._http = None
        idx = json.loads(self._get_small("model.safetensors.index.json"))
        self.map = idx["weight_map"]
        self._headers = {}
        self._lock = threading.Lock()
        self._http = None
        self.pool = ThreadPoolExecutor(workers)
        self.bytes = 0

    def _client(self):
        import httpx
        if getattr(self, "_http", None) is None:
            self._http = httpx.Client(follow_redirects=False, timeout=180,
                                      limits=httpx.Limits(max_connections=128, max_keepalive_connections=128))
            self._loc = {}
        return self._http

    def _resolve(self, path):
        """Resolve the Hub URL once to its (presigned) CDN location."""
        c = self._client()
        with self._lock:
            if path in self._loc:
                return self._loc[path]
        r = c.head(self.base + path)
        loc = r.headers.get("location", self.base + path) if r.status_code in (301, 302, 303, 307, 308) else self.base + path
        if loc.startswith("/"):
            loc = "https://huggingface.co" + loc
        with self._lock:
            self._loc[path] = loc
        return loc

    def _open(self, path, rng=None, tries=8):
        c = self._client()
        for t in range(tries):
            try:
                url = self._resolve(path)
                r = c.get(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}, follow_redirects=True)
                if r.status_code in (401, 403):     # expired presigned URL
                    with self._lock:
                        self._loc.pop(path, None)
                    raise IOError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.content
            except Exception:
                if t == tries - 1:
                    raise
                time.sleep(2 * (t + 1))

    def _get_small(self, path) -> bytes:
        return self._open(path)

    def _header(self, f):
        with self._lock:
            if f in self._headers:
                return self._headers[f]
        n = struct.unpack("<Q", self._open(f, (0, 7)))[0]
        h = json.loads(self._open(f, (8, 8 + n - 1)))
        h.pop("__metadata__", None)
        with self._lock:
            self._headers[f] = (8 + n, h)
        return self._headers[f]

    def get(self, name: str) -> np.ndarray:
        f = self.map[name]
        off, h = self._header(f)
        meta = h[name]
        s, e = meta["data_offsets"]
        n = e - s
        piece = 64 << 20
        if n > 4 * piece:
            # large tensors (embedding, lm_head: 3.5 GB in fp32): parallel ranged pieces
            from concurrent.futures import ThreadPoolExecutor
            spans = [(off + a, off + min(a + piece, e) - 1) for a in range(s, e, piece)]
            with ThreadPoolExecutor(16) as ex:
                parts = list(ex.map(lambda sp: self._open(f, sp), spans))
            buf = b"".join(parts)
        else:
            buf = self._open(f, (off + s, off + e - 1))
        self.bytes += len(buf)
        return np.frombuffer(buf, dtype=_DT[meta["dtype"]]).reshape(meta["shape"])

    def prefetch(self, names):
        """Fetch many tensors in parallel; returns dict name -> array."""
        futs = {n: self.pool.submit(self.get, n) for n in names}
        return {n: f.result() for n, f in futs.items()}

    def has(self, name):
        return name in self.map
