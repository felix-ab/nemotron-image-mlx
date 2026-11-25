"""Loading, teacher-forced scoring and image generation."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import Args, Block, NemotronImageLM, rms_norm
from .noise import exponential_row
from .prompt import PATCH, PromptTokenizer
from .weights import ArrayEmbedding, MemmapEmbedding, Upstream, head_tensors, layer_tensors, quantize_module

BF16, F32 = mx.bfloat16, mx.float32


def _recipe(cfg) -> dict:
    r = cfg.get("mlx_recipe") or {}
    return {k: tuple(v) if v else None for k, v in r.items()}


def load_model(path, layers_resident: bool = True, embed: str = "auto") -> NemotronImageLM:
    """embed: "gpu" keeps the bf16 embedding table on the GPU heap (+1.76 GB, lets decode
    pipeline without a per-token CPU sync); "cpu" memory-maps it; "auto" picks gpu when the
    device's recommended working set leaves >= 6 GB after the weights."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    a = Args.from_config(cfg)
    recipe = _recipe(cfg)
    model = NemotronImageLM(a)
    for blk in model.layers:
        if recipe:
            quantize_module(blk, recipe)
    if recipe.get("lm_head"):
        bits, gs, mode = recipe["lm_head"]
        model.lm_head = model.lm_head.to_quantized(group_size=gs, bits=bits, mode=mode)
    w = []
    for i in range(len(a.pattern)):
        for k, v in mx.load(str(path / f"layers-{i:02d}.safetensors")).items():
            w.append((f"layers.{i}.{k}", v))
    w += list(mx.load(str(path / "head.safetensors")).items())
    model.load_weights(w, strict=True)
    mx.eval(model.parameters())
    mm = MemmapEmbedding(path / "embed.npy", BF16)
    if embed == "auto":
        ws = mx.device_info().get("max_recommended_working_set_size", 0)
        embed = "gpu" if ws - mx.get_active_memory() - mm.raw.nbytes >= 6e9 else "cpu"
    model.embed = ArrayEmbedding(mx.array(np.asarray(mm.raw)).view(BF16)) if embed == "gpu" else mm
    model.embed_device = embed
    return model


# --------------------------------------------------------------------------- streamed (layer-by-layer) forward


class LayerSource:
    """Yields Block modules one at a time, either from the upstream fp32 checkpoint
    (cast to ``dtype``, optionally quantized by ``recipe``) or from a converted MLX dir."""

    def __init__(self, path, dtype=BF16, recipe: Optional[dict] = None):
        self.path = Path(path)
        self.dtype = dtype
        self.recipe = recipe or {}
        self.upstream = (self.path / "model.safetensors.index.json").exists()
        cfg = json.loads((self.path / "config.json").read_text())
        self.a = Args.from_config(cfg)
        if not self.upstream:
            self.recipe = _recipe(cfg)
            self.dtype = BF16
        self.up = Upstream(self.path) if self.upstream else None

    def embed(self, ids: np.ndarray) -> mx.array:
        if self.upstream:
            # read only the needed rows (fp32 source)
            from safetensors import safe_open
            t = self.up._h(self.up.map["backbone.embeddings.weight"]).get_slice("backbone.embeddings.weight")
            uniq, inv = np.unique(ids, return_inverse=True)
            rows = np.stack([t[int(u):int(u) + 1][0] for u in uniq])
            return mx.array(rows[inv.reshape(ids.shape)]).astype(self.dtype)
        return MemmapEmbedding(self.path / "embed.npy", BF16)(mx.array(ids))

    def layer(self, i: int) -> Block:
        a = self.a
        blk = Block(a, a.pattern[i])
        if self.upstream:
            blk.load_weights(list(layer_tensors(self.up, a, i, self.dtype).items()), strict=True)
            if self.recipe:
                quantize_module(blk, self.recipe)
        else:
            if self.recipe:
                quantize_module(blk, self.recipe)
            blk.load_weights(list(mx.load(str(self.path / f"layers-{i:02d}.safetensors")).items()), strict=True)
        mx.eval(blk.parameters())
        return blk

    def head(self):
        if self.upstream:
            h = head_tensors(self.up, self.dtype)
            lm = nn.Linear(self.a.hidden_size, self.a.vocab_size, bias=False)
            lm.weight = h["lm_head.weight"]
            if self.recipe.get("lm_head"):
                bits, gs, mode = self.recipe["lm_head"]
                lm = lm.to_quantized(group_size=gs, bits=bits, mode=mode)
            return h["norm_f_weight"], lm
        raw = mx.load(str(self.path / "head.safetensors"))
        lm = nn.Linear(self.a.hidden_size, self.a.vocab_size, bias=False)
        if self.recipe.get("lm_head"):
            bits, gs, mode = self.recipe["lm_head"]
            lm = lm.to_quantized(group_size=gs, bits=bits, mode=mode)
        lm.load_weights([(k[len("lm_head."):], v) for k, v in raw.items() if k.startswith("lm_head.")], strict=True)
        return raw["norm_f_weight"], lm


def streamed_logits(
    src: LayerSource,
    ids: np.ndarray,
    positions: Optional[np.ndarray] = None,
    hooks: Optional[Callable[[int, mx.array], None]] = None,
    chunk: int = 4096,
) -> mx.array:
    """Teacher-forced logits for ``ids`` [B, L], one layer resident at a time.

    Returns logits at ``positions`` (indices into L; default all) as float32.
    """
    h = src.embed(ids)
    mx.eval(h)
    for i in range(len(src.a.pattern)):
        blk = src.layer(i)
        h = blk(h, None)
        mx.eval(h)
        if hooks:
            hooks(i, h)
        del blk
        gc.collect()
        mx.clear_cache()
    nw, lm = src.head()
    if positions is not None:
        h = h[:, mx.array(positions)]
    out = []
    for s in range(0, h.shape[1], chunk):
        hs = rms_norm(h[:, s:s + chunk], nw, src.a.eps)
        out.append(lm(hs).astype(F32))
        mx.eval(out[-1])
    return mx.concatenate(out, axis=1)


# --------------------------------------------------------------------------- generation


@dataclass
class GenConfig:
    width: int = 1024
    height: int = 1024
    temperature: float = 0.9
    max_tokens: int = 4352
    seed: int = 42
    noise: str = "vllm"          # "vllm": CUDA-Philox emulation; "mlx": MLX RNG
    gpu: str = "h100-sxm"
    constrained: bool = False    # mask tokens that cannot be valid at each grid position


@dataclass
class GenResult:
    tokens: List[int]
    codes: Optional[List[int]] = None
    grid: Optional[tuple] = None
    header: str = ""
    error: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)


class Sampler:
    """vLLM v1 random_sample: argmax(softmax(logits / T) / q), q ~ Exp(1) per request."""

    def __init__(self, cfg: GenConfig, vocab: int, seeds: Sequence[int]):
        self.cfg, self.vocab, self.seeds = cfg, vocab, list(seeds)
        self.step = 0
        if cfg.noise == "mlx":
            self.keys = [mx.random.key(s) for s in self.seeds]

    def noise(self) -> mx.array:
        if self.cfg.noise == "vllm":
            q = mx.stack([exponential_row(s, self.step, self.vocab, self.cfg.gpu) for s in self.seeds])
        else:
            us = []
            for j, k in enumerate(self.keys):
                k, sub = mx.random.split(k)
                self.keys[j] = k
                us.append(mx.random.uniform(shape=(self.vocab,), key=sub))
            q = -mx.log1p(-mx.stack(us))
        self.step += 1
        return q

    def __call__(self, logits: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        l = logits.astype(F32)
        if self.cfg.temperature <= 0:
            self.step += 1
            return mx.argmax(l, axis=-1)
        l = l / self.cfg.temperature
        if mask is not None:
            l = mx.where(mask, l, -mx.inf)
        probs = mx.softmax(l, axis=-1)
        return mx.argmax(probs / self.noise(), axis=-1)


def generate(
    model: NemotronImageLM,
    tok: PromptTokenizer,
    prompt: str,
    cfg: GenConfig,
    seeds: Optional[Sequence[int]] = None,
    on_token: Optional[Callable[[int, mx.array], None]] = None,
) -> List[GenResult]:
    seeds = list(seeds) if seeds is not None else [cfg.seed]
    B = len(seeds)
    t0 = time.perf_counter()
    pids = tok.encode(prompt, cfg.width, cfg.height)
    cache = model.make_cache()
    sampler = Sampler(cfg, model.a.vocab_size, seeds)
    ids = mx.array([pids] * B)
    logits = model(ids, cache)[:, -1, :]
    tok_ = sampler(logits)
    mx.eval(tok_)
    t_prefill = time.perf_counter() - t0
    out = [[int(t)] for t in np.array(tok_)]
    done = np.array([o[-1] == tok.ids.img_end for o in out])
    n = 1
    t1 = time.perf_counter()
    while n < cfg.max_tokens and not done.all():
        logits = model(tok_[:, None], cache)[:, -1, :]
        tok_ = sampler(logits)
        mx.async_eval(tok_)
        arr = np.array(tok_)
        for j in range(B):
            if not done[j]:
                out[j].append(int(arr[j]))
                done[j] = arr[j] == tok.ids.img_end
        if on_token:
            on_token(n, tok_)
        n += 1
    t_decode = time.perf_counter() - t1
    res = []
    for j in range(B):
        r = GenResult(tokens=out[j], timings={"prefill_s": t_prefill, "decode_s": t_decode, "tokens": n,
                                              "tok_per_s": (n - 1) / max(t_decode, 1e-9)})
        try:
            r.codes, r.grid, r.header = tok.parse_image(out[j], cfg.width, cfg.height)
        except RuntimeError as e:
            r.error = str(e)
        res.append(r)
    return res


def generate_fast(
    model: NemotronImageLM,
    tok: PromptTokenizer,
    prompt: str,
    cfg: GenConfig,
    seeds: Optional[Sequence[int]] = None,
    compile: bool = True,
    screen: bool = False,
    prefill: str = "parallel",
) -> List[GenResult]:
    """Same sampling as ``generate`` (vLLM noise), with the compiled fixed-shape decode step
    and a one-step-late token readback so the CPU never waits on the GPU."""
    from .fast import FastDecoder

    assert cfg.noise == "vllm", "fast path implements the vLLM noise stream"
    seeds = list(seeds) if seeds is not None else [cfg.seed]
    B = len(seeds)
    t0 = time.perf_counter()
    pids = tok.encode(prompt, cfg.width, cfg.height)
    P = len(pids)
    max_len = P + cfg.max_tokens
    dec = FastDecoder(model, B, max_len, cfg.temperature, seeds, cfg.gpu, compile=compile, screen=screen)
    if prefill == "parallel":
        cache = model.make_cache()
        ids = mx.array([pids] * B)
        logits = model(ids, cache)[:, -1, :]
        sampler = Sampler(cfg, model.a.vocab_size, seeds)
        tok_ = sampler(logits)                   # step 0 (prompt's last position)
        state = dec.state_from_cache(cache, P)
        del cache
    else:
        # recurrent prefill through the compiled decode step (no prefill-only kernels to JIT)
        state = dec.zero_state()
        for t, pid in enumerate(pids):
            x = model.embed(mx.array([[pid]] * B))
            tok_, need, h, state = dec.step(x, t, 0, state)
            if t % 16 == 15:
                mx.eval(state)
        if screen:
            nd = np.array(need)
            if nd.any():
                tok_ = mx.where(mx.array(nd), dec.exact(h, 0), tok_)
    mx.eval(tok_, state)
    t_prefill = time.perf_counter() - t0
    out = [[] for _ in range(B)]
    done = np.zeros(B, bool)
    t1 = time.perf_counter()
    cur = tok_                                   # token sampled at step n-1 (already final)
    n = 1
    if isinstance(model.embed, ArrayEmbedding) and not screen:
        # GPU-resident embedding: queue step n+1 before reading step n's token (no CPU stall).
        lag = []
        while True:
            if n < cfg.max_tokens and not done.all():
                x = model.embed(cur[:, None])
                nxt, need, h, state = dec.step(x, P + n - 1, n, state)
                mx.async_eval(nxt)
            else:
                nxt = None
            arr = np.array(cur)
            for j in range(B):
                if not done[j]:
                    out[j].append(int(arr[j]))
                    done[j] = arr[j] == tok.ids.img_end
            if nxt is None or done.all():
                break
            cur = nxt
            n += 1
    while not (isinstance(model.embed, ArrayEmbedding) and not screen):
        arr = np.array(cur)
        for j in range(B):
            if not done[j]:
                out[j].append(int(arr[j]))
                done[j] = arr[j] == tok.ids.img_end
        if done.all() or n >= cfg.max_tokens:
            break
        x = model.embed(cur[:, None])
        nxt, need, h, state = dec.step(x, P + n - 1, n, state)
        mx.async_eval(nxt, need, h, state)
        if screen:
            nd = np.array(need)
            if nd.any():
                nxt = mx.where(mx.array(nd), dec.exact(h, n), nxt)
        cur = nxt
        n += 1
    mx.eval(state)
    t_decode = time.perf_counter() - t1
    res = []
    for j in range(B):
        r = GenResult(tokens=out[j], timings={"prefill_s": t_prefill, "decode_s": t_decode, "tokens": n,
                                              "tok_per_s": n / max(t_decode, 1e-9), "prompt_tokens": P,
                                              "exact_fallbacks": dec.n_exact})
        try:
            r.codes, r.grid, r.header = tok.parse_image(out[j], cfg.width, cfg.height)
        except RuntimeError as e:
            r.error = str(e)
        res.append(r)
    return res


def load_model_hub(repo: str, meta_dir, recipe_spec: str, embed: str = "auto") -> NemotronImageLM:
    """Build a tier in memory straight from the Hub's fp32 tensors (no disk copy).

    Same numerics as ``convert`` + ``load_model``: every tensor is cast to bf16 and the
    recipe's modules are quantized from those bf16 weights.
    """
    from .convert import _Prefetched, fetch_small
    from .remote import RemoteUpstream
    from .weights import parse_recipe

    meta_dir = Path(meta_dir)
    fetch_small(repo, meta_dir)
    recipe = parse_recipe(recipe_spec)
    a = Args.load(meta_dir)
    up = _Prefetched(RemoteUpstream(repo))
    model = NemotronImageLM(a)
    for i in range(len(a.pattern)):
        up.load(f"backbone.layers.{i}.")
        blk = Block(a, a.pattern[i])
        blk.load_weights(list(layer_tensors(up, a, i, BF16).items()), strict=True)
        if recipe:
            quantize_module(blk, recipe)
        mx.eval(blk.parameters())
        model.layers[i] = blk
        up.cache = {}
        gc.collect()
        mx.clear_cache()
    h = head_tensors(up, BF16)
    model.norm_f_weight = h["norm_f_weight"]
    model.lm_head.weight = h["lm_head.weight"]
    if recipe.get("lm_head"):
        bits, gs, mode = recipe["lm_head"]
        model.lm_head = model.lm_head.to_quantized(group_size=gs, bits=bits, mode=mode)
    mx.eval(model.parameters())
    table = mx.array(up.get("backbone.embeddings.weight")).astype(BF16)
    if embed == "auto":
        ws = mx.device_info().get("max_recommended_working_set_size", 0)
        embed = "gpu" if ws - mx.get_active_memory() - table.nbytes >= 6e9 else "cpu"
    if embed == "gpu":
        model.embed = ArrayEmbedding(table)
    else:
        model.embed = _NumpyEmbedding(np.array(table.view(mx.uint16)))
    model.embed_device = embed
    return model


class _NumpyEmbedding:
    def __init__(self, raw):
        self.raw = raw

    def __call__(self, ids):
        idx = np.array(ids)
        return mx.array(np.ascontiguousarray(self.raw[idx.reshape(-1)])).view(BF16).reshape(*idx.shape, -1)


def image_schedule(tok: PromptTokenizer, width: int, height: int) -> List[Optional[int]]:
    """Tokens after <|img_start|> in a valid image: forced ids, or None where a visual code is sampled."""
    hdr = tok.tok.encode(f"height={height} width={width}", add_special_tokens=False).ids
    rows, cols = height // PATCH, width // PATCH
    sched: List[Optional[int]] = list(hdr)
    for r in range(rows):
        sched.append(tok.ids.vis_row)
        sched += [None] * cols
    sched.append(tok.ids.img_end)
    return sched


def generate_constrained(
    model: NemotronImageLM,
    tok: PromptTokenizer,
    prompt: str,
    cfg: GenConfig,
    seeds: Optional[Sequence[int]] = None,
    compile: bool = True,
    prefill: str = "parallel",
) -> List[GenResult]:
    """vLLM-seeded sampling restricted to the image grammar.

    Tokens before <|img_start|> are sampled freely. After it, the size header and the row/end
    markers are fixed and in-row positions sample among visual codes only. Masking rescales
    the allowed probabilities by a common factor, so argmax(p/q) over the allowed set is the
    unconstrained sample whenever that sample is allowed: every run the reference would accept
    is reproduced token for token; runs it would reject (wrong header, ragged rows) become valid.
    Forced positions skip the output head; visual positions use only its visual rows.
    """
    from .fast import FastDecoder

    seeds = list(seeds) if seeds is not None else [cfg.seed]
    B = len(seeds)
    V = model.a.vocab_size
    ids_ = tok.ids
    t0 = time.perf_counter()
    pids = tok.encode(prompt, cfg.width, cfg.height)
    P = len(pids)
    sched = image_schedule(tok, cfg.width, cfg.height)
    max_len = P + cfg.max_tokens
    dec = FastDecoder(model, B, max_len, cfg.temperature, seeds, cfg.gpu, compile=compile)
    dec.enable_constrained(ids_.vis_start, ids_.vis_end)
    if prefill == "parallel":
        cache = model.make_cache()
        logits = model(mx.array([pids] * B), cache)[:, -1, :]
        tok0 = Sampler(cfg, V, seeds)(logits)
        state = dec.state_from_cache(cache, P)
        del cache
    else:
        state = dec.zero_state()
        for t, pid in enumerate(pids):
            tok0, _, _, state = dec.step(model.embed(mx.array([[pid]] * B)), t, 0, state)
            if t % 16 == 15:
                mx.eval(state)
    mx.eval(tok0, state)
    t_prefill = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = [[int(v)] for v in np.array(tok0)]
    ptr = [0 if o[0] == ids_.img_start else None for o in out]   # index into sched of the NEXT token
    cur = tok0
    n = 1
    counts = {"forced": 0, "visual": 0, "masked": 0, "free": 0}
    while n < cfg.max_tokens:
        if all(p is not None and p >= len(sched) for p in ptr):
            break
        modes = []
        for j in range(B):
            p = ptr[j]
            if p is None:
                modes.append(("free", -1))
            elif p >= len(sched):
                modes.append(("forced", ids_.img_end))      # finished row: keep feeding a harmless token
            elif sched[p] is None:
                modes.append(("visual", -1))
            else:
                modes.append(("forced", sched[p]))
        kinds = {m[0] for m in modes}
        x = model.embed(cur[:, None])
        if kinds == {"forced"}:
            h, state = dec.step_none(x, P + n - 1, state)
            nxt = mx.array([m[1] for m in modes], mx.int32)
            counts["forced"] += 1
        elif kinds == {"visual"}:
            nxt, h, state = dec.step_vis(x, P + n - 1, n, state)
            counts["visual"] += 1
        elif kinds == {"free"}:
            nxt, _, h, state = dec.step(x, P + n - 1, n, state)
            counts["free"] += 1
        else:
            forced = [m[1] if m[0] == "forced" else -1 for m in modes]
            lo = [ids_.vis_start if m[0] == "visual" else 0 for m in modes]
            hi = [ids_.vis_end if m[0] == "visual" else V for m in modes]
            nxt, h, state = dec.step_masked(x, P + n - 1, n, forced, lo, hi, state)
            counts["masked"] += 1
        mx.async_eval(nxt, state)
        arr = np.array(nxt)
        for j in range(B):
            p = ptr[j]
            if p is not None and p >= len(sched):
                continue
            out[j].append(int(arr[j]))
            if p is None:
                if arr[j] == ids_.img_start:
                    ptr[j] = 0
            else:
                ptr[j] = p + 1
        cur = nxt
        n += 1
    mx.eval(state)
    t_decode = time.perf_counter() - t1
    res = []
    for j in range(B):
        r = GenResult(tokens=out[j], timings={"prefill_s": t_prefill, "decode_s": t_decode, "tokens": n,
                                              "tok_per_s": n / max(t_decode, 1e-9), "prompt_tokens": P,
                                              "steps": counts})
        try:
            r.codes, r.grid, r.header = tok.parse_image(out[j], cfg.width, cfg.height)
        except RuntimeError as e:
            r.error = str(e)
        res.append(r)
    return res
