"""Convert the upstream fp32 checkpoint into an MLX directory (bf16 + optional quantized experts).

Layout of an output directory:
  config.json                 upstream model config (+ "mlx_recipe")
  embed.npy                   embedding table, bf16 bits as uint16 (memory-mapped at run time)
  head.safetensors            final norm + lm_head
  layers-XX.safetensors       one file per decoder layer (module-relative keys)
  tokenizer.json, chat_template.jinja, unitron_extension_manifest.json
  vq/                         pixel decoder (L24SQ) config + weights
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import Args, Block, NemotronImageLM
from .weights import Upstream, head_tensors, layer_tensors, parse_recipe, quantize_module

BF16 = mx.bfloat16


def build_layer(a: Args, i: int, tensors: dict, recipe: dict) -> Block:
    blk = Block(a, a.pattern[i])
    blk.load_weights(list(tensors.items()), strict=True)
    if recipe:
        quantize_module(blk, recipe)
    return blk


SMALL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "unitron_extension_manifest.json", "special_tokens_map.json", "LICENSE", "NOTICE",
               "Nemotron-Labs-Image-Tokenizer/config.json", "Nemotron-Labs-Image-Tokenizer/model.safetensors")


def fetch_small(repo: str, local: Path):
    from huggingface_hub import hf_hub_download
    for f in SMALL_FILES:
        if not (local / f).exists():
            hf_hub_download(repo, f, local_dir=str(local))


class _Prefetched:
    """Wraps a RemoteUpstream: fetches all of a layer's tensors in parallel, then serves them."""

    def __init__(self, remote):
        self.remote, self.cache = remote, {}

    def load(self, prefix):
        self.cache = self.remote.prefetch([n for n in self.remote.map if n.startswith(prefix)])

    def get(self, name):
        return self.cache[name] if name in self.cache else self.remote.get(name)


def convert(src: Path, dst: Path, recipe_spec: str, layers=None, hub: str = None):
    recipe = parse_recipe(recipe_spec)
    if hub:
        from .remote import RemoteUpstream
        fetch_small(hub, src)
        up = _Prefetched(RemoteUpstream(hub))
    else:
        up = Upstream(src)
    a = Args.load(src)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    cfg["mlx_recipe"] = {k: list(v) if v else None for k, v in recipe.items()}
    cfg["mlx_recipe_name"] = recipe_spec
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "unitron_extension_manifest.json",
              "special_tokens_map.json", "LICENSE", "NOTICE"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    vqs = src / "Nemotron-Labs-Image-Tokenizer"
    if vqs.exists() and not (dst / "vq" / "model.safetensors").exists():
        (dst / "vq").mkdir(exist_ok=True)
        for f in ("config.json", "model.safetensors"):
            try:
                import os
                os.link(vqs / f, dst / "vq" / f)  # same volume: no extra disk
            except OSError:
                shutil.copy2(vqs / f, dst / "vq" / f)
    if not (dst / "embed.npy").exists():
        e = mx.array(up.get("backbone.embeddings.weight")).astype(BF16)
        np.save(dst / "embed.npy", np.array(e.view(mx.uint16)))
        del e
    if not (dst / "head.safetensors").exists():
        h = head_tensors(up, BF16)
        hm = nn.Module()
        lm = nn.Linear(a.hidden_size, a.vocab_size, bias=False)
        lm.weight = h["lm_head.weight"]
        if recipe.get("lm_head"):
            bits, gs, mode = recipe["lm_head"]
            lm = lm.to_quantized(group_size=gs, bits=bits, mode=mode)
        flat = {"norm_f_weight": h["norm_f_weight"]}
        flat.update({f"lm_head.{k}": v for k, v in dict(nn.utils.tree_flatten(lm.parameters())).items()})
        mx.save_safetensors(str(dst / "head.safetensors"), flat)
        del h, lm, flat
    for i in (layers if layers is not None else range(len(a.pattern))):
        out = dst / f"layers-{i:02d}.safetensors"
        if out.exists():
            continue
        t0 = time.time()
        if hub:
            up.load(f"backbone.layers.{i}.")
        blk = build_layer(a, i, layer_tensors(up, a, i, BF16), recipe)
        flat = dict(nn.utils.tree_flatten(blk.parameters()))
        mx.eval(flat)
        mx.save_safetensors(str(out), flat)
        print(f"layer {i:2d} {a.pattern[i]} -> {out.stat().st_size / 1e9:.2f} GB ({time.time() - t0:.1f}s)", flush=True)
        del blk, flat
        gc.collect()
        mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--recipe", default="bf16")
    ap.add_argument("--hub", default=None, help="stream tensors from this Hub repo instead of a local checkpoint")
    args = ap.parse_args()
    convert(Path(args.src), Path(args.dst), args.recipe, hub=args.hub)


if __name__ == "__main__":
    main()
