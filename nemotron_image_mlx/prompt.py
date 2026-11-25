"""Prompt construction and visual-token framing without transformers.

Reproduces ``tokenizer.apply_chat_template(..., add_generation_prompt=True,
enable_thinking=False)`` of the upstream ``UnitronLazyVisionTokenizer`` (the
chat template is rendered with jinja2 the way transformers does, the text is
encoded with the base ``tokenizers`` model), plus the formula-based visual IDs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

SYSTEM_PROMPT = (
    "Generate an image matching the requested resolution by detailing the color, "
    "quantity, text, shape, size, texture, spatial relationships of the objects "
    "and background."
)
PATCH = 16


@dataclass
class VisualIds:
    img_start: int
    img_end: int
    vis_row: int
    vis_start: int
    vis_end: int


class PromptTokenizer:
    def __init__(self, path):
        path = Path(path)
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        man = json.loads((path / "unitron_extension_manifest.json").read_text())
        r = man["unitron_token_ranges"]
        self.ids = VisualIds(
            img_start=r["framing"]["<|img_start|>"],
            img_end=r["framing"]["<|img_end|>"],
            vis_row=r["framing"]["<|vis_row|>"],
            vis_start=r["vision"]["start_id"],
            vis_end=r["vision"]["end_id"],
        )
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.exceptions.TemplateError(m))
        self.template = env.from_string((path / "chat_template.jinja").read_text())

    def render(self, prompt: str, width: int, height: int, system: str = SYSTEM_PROMPT) -> str:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Generate a {width}x{height} image.\n\n{prompt}"},
        ]
        return self.template.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)

    def encode(self, prompt: str, width: int, height: int, system: str = SYSTEM_PROMPT) -> List[int]:
        if width % PATCH or height % PATCH or width <= 0 or height <= 0:
            raise ValueError("width and height must be positive multiples of 16")
        return self.tok.encode(self.render(prompt, width, height, system), add_special_tokens=False).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode([i for i in ids if i < self.ids.img_start], skip_special_tokens=False)

    def parse_image(self, ids: List[int], width: int, height: int) -> Tuple[List[int], Tuple[int, int], str]:
        """Validate one generated image block and return codebook indices (as upstream infer.py)."""
        v = self.ids
        try:
            s = ids.index(v.img_start)
            e = ids.index(v.img_end, s + 1)
        except ValueError as exc:
            raise RuntimeError("generation did not contain a complete image block") from exc
        prefix, rows, cur = [], [], None
        for t in ids[s + 1:e]:
            if t == v.vis_row:
                if cur is not None:
                    rows.append(cur)
                cur = []
            elif v.vis_start <= t < v.vis_end:
                if cur is None:
                    raise RuntimeError("visual token before the first row marker")
                cur.append(t - v.vis_start)
            elif cur is None:
                prefix.append(t)
            else:
                raise RuntimeError("non-visual token inside the image grid")
        if cur is not None:
            rows.append(cur)
        header = self.decode(prefix)
        m = re.search(r"height=(\d+)\s+width=(\d+)", header)
        if m is None:
            raise RuntimeError(f"no valid size header in {header!r}")
        gh, gw = map(int, m.groups())
        if (gh, gw) != (height, width):
            raise RuntimeError(f"generated size {(gw, gh)} != requested {(width, height)}")
        grid = (height // PATCH, width // PATCH)
        if len(rows) != grid[0] or any(len(r) != grid[1] for r in rows):
            raise RuntimeError(f"visual-token grid has the wrong shape: {len(rows)} rows, lengths {sorted(set(map(len, rows)))}")
        return [c for r in rows for c in r], grid, header
