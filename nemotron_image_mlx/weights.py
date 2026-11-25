"""Weight loading: upstream fp32 checkpoint -> MLX modules, conversion, and quantization recipes."""

from __future__ import annotations

import gc
import json
import math
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from .model import Args, Block, NemotronImageLM, MoE

F32, BF16 = mx.float32, mx.bfloat16

# Parameters vLLM keeps in fp32 when the model runs in bf16.
FP32_KEEP = ("A_neg", "e_score_correction_bias")


class Upstream:
    """Lazy reader for the original fp32 safetensors shards."""

    def __init__(self, path):
        self.path = Path(path)
        idx = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.map = idx["weight_map"]
        self._handles = {}

    def _h(self, f):
        if f not in self._handles:
            self._handles[f] = safe_open(str(self.path / f), framework="numpy")
        return self._handles[f]

    def get(self, name: str) -> np.ndarray:
        return self._h(self.map[name]).get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.map

    def close(self):
        self._handles.clear()


def _to(x: np.ndarray, dtype) -> mx.array:
    return mx.array(x).astype(dtype)


def block_tensors(get, kind: str, a: Args, dtype=BF16, n_experts: int = None) -> Dict[str, mx.array]:
    """Module-relative parameters for one block; ``get(name)`` returns block-relative upstream tensors."""
    out = {"norm_weight": _to(get("norm.weight"), dtype)}
    m = "mixer."
    if kind == "M":
        out["mixer.in_proj.weight"] = _to(get(m + "in_proj.weight"), dtype)
        out["mixer.out_proj.weight"] = _to(get(m + "out_proj.weight"), dtype)
        cw = get(m + "conv1d.weight")  # [C, 1, K]
        out["mixer.conv_weight"] = _to(cw.reshape(cw.shape[0], cw.shape[-1]), dtype)
        out["mixer.conv_bias"] = _to(get(m + "conv1d.bias"), dtype)
        out["mixer.dt_bias"] = _to(get(m + "dt_bias"), dtype)
        out["mixer.D"] = _to(get(m + "D"), dtype)
        out["mixer.norm_weight"] = _to(get(m + "norm.weight"), dtype)
        out["mixer.A_neg"] = -mx.exp(mx.array(get(m + "A_log")).astype(F32))
    elif kind == "*":
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[f"mixer.{n}.weight"] = _to(get(m + f"{n}.weight"), dtype)
    elif kind == "E":
        out["mixer.gate_weight"] = _to(get(m + "gate.weight"), dtype)
        out["mixer.e_score_correction_bias"] = mx.array(get(m + "gate.e_score_correction_bias")).astype(F32)
        for n in ("up_proj", "down_proj"):
            out[f"mixer.shared_experts.{n}.weight"] = _to(get(m + f"shared_experts.{n}.weight"), dtype)
        E = n_experts or a.n_routed_experts
        for n, dst in (("up_proj", "fc1"), ("down_proj", "fc2")):
            # stack expert-by-expert in the target dtype to bound peak memory
            out[f"mixer.{dst}.weight"] = mx.stack([_to(get(m + f"experts.{e}.{n}.weight"), dtype) for e in range(E)])
    return out


def layer_tensors(up: Upstream, a: Args, i: int, dtype=BF16) -> Dict[str, mx.array]:
    """Module-relative parameter dict for layer i in the given compute dtype."""
    p = f"backbone.layers.{i}."
    return block_tensors(lambda n: up.get(p + n), a.pattern[i], a, dtype)


def head_tensors(up: Upstream, dtype=BF16) -> Dict[str, mx.array]:
    return {
        "norm_f_weight": _to(up.get("backbone.norm_f.weight"), dtype),
        "lm_head.weight": _to(up.get("lm_head.weight"), dtype),
    }


# --------------------------------------------------------------------------- embedding


class MemmapEmbedding:
    """Embedding rows read from a memory-mapped file on demand (one row per token).

    The 327,680 x 2,688 table is 1.76 GB in bf16, but decoding touches one row
    per step; keeping it off the GPU heap is exact and frees that memory.
    """

    def __init__(self, path: Path, dtype=BF16):
        self.dtype = dtype
        self.raw = np.load(path, mmap_mode="r")  # uint16 (bf16 bits) or float32

    def __call__(self, ids: mx.array) -> mx.array:
        idx = np.array(ids)
        rows = np.ascontiguousarray(self.raw[idx.reshape(-1)])
        if rows.dtype == np.uint16:
            x = mx.array(rows).view(BF16)
        else:
            x = mx.array(rows)
        return x.reshape(*idx.shape, -1).astype(self.dtype)


class ArrayEmbedding:
    def __init__(self, table: mx.array):
        self.table = table

    def __call__(self, ids):
        return self.table[ids]


# --------------------------------------------------------------------------- quantization recipes

# A recipe maps a parameter path to (bits, group_size, mode) or None (keep bf16).
RECIPES = {
    "bf16": {},
    "q8": {"experts": (8, 64, "affine")},
    "q6": {"experts": (6, 64, "affine")},
    "q5": {"experts": (5, 64, "affine")},
    "q4": {"experts": (4, 64, "affine")},
    "mxfp4": {"experts": (4, 32, "mxfp4")},
    "mxfp8": {"experts": (8, 32, "mxfp8")},
}


def parse_recipe(spec: str) -> dict:
    """'q6' or 'experts=6,lm_head=8,dense=8,shared=8'."""
    if spec in RECIPES:
        return dict(RECIPES[spec])
    r = {}
    for part in spec.split(","):
        k, v = part.split("=")
        if v in ("bf16", "none"):
            r[k] = None
        elif v.startswith("mx"):
            r[k] = (int(v[-1]) if v[-1] in "48" else 4, 32, v)
        else:
            bits, _, gs = v.partition("g")
            r[k] = (int(bits), int(gs) if gs else 64, "affine")
    return r


def _category(path: str) -> str:
    if ".fc1" in path or ".fc2" in path:
        return "experts"
    if "shared_experts" in path:
        return "shared"
    if path.startswith("lm_head"):
        return "lm_head"
    if "gate_weight" in path:
        return "router"
    return "dense"  # mamba in/out proj, attention projections


def quantize_module(module: nn.Module, recipe: dict, prefix: str = ""):
    """Quantize linear / switch-linear submodules in place according to recipe."""

    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        cat = _category(prefix + path)
        q = recipe.get(cat)
        if not q:
            return False
        bits, gs, mode = q
        return {"bits": bits, "group_size": gs, "mode": mode}

    nn.quantize(module, class_predicate=pred)


def recipe_bpw(recipe: dict) -> str:
    return json.dumps({k: v for k, v in recipe.items()})
