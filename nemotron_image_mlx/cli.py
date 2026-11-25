"""nemotron-image: text-to-image with Nemotron-Labs-Image-30B-A3B on MLX."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .pipeline import GenConfig, generate, generate_constrained, generate_fast, load_model
from .prompt import PromptTokenizer
from .vq import L24SQ, load_vq


def main():
    ap = argparse.ArgumentParser(prog="nemotron-image")
    ap.add_argument("prompt")
    ap.add_argument("--model", default="auto",
                    help="auto (best tier for this Mac), 8bit, 5bit, 6bit, a Hugging Face repo id, or a local directory")
    ap.add_argument("--size", default="1024x1024", help="WIDTHxHEIGHT, multiples of 16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, generated as one batch")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=4352)
    ap.add_argument("--noise", choices=["vllm", "mlx"], default="vllm",
                    help="vllm: reproduce vLLM's CUDA-seeded sampling noise; mlx: MLX RNG")
    ap.add_argument("--prefill", choices=["step", "parallel"], default="parallel",
                    help="step: prompt through the compiled decode step (less cold-start JIT); parallel: chunked prefill")
    ap.add_argument("--embed", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--vq-dtype", choices=["float32", "bfloat16"], default="float32",
                    help="pixel decoder precision (float32 is closer to the true decoder than the bf16 reference)")
    ap.add_argument("--screen", action="store_true", help="exact screened lm_head (skips most text-token rows)")
    ap.add_argument("--eager", action="store_true", help="uncompiled reference decoding loop")
    ap.add_argument("--unconstrained", action="store_true",
                    help="sample exactly like the reference even when it produces an invalid image "
                         "(default: image grammar enforced; identical tokens on every valid reference run)")
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--save-tokens", action="store_true")
    args = ap.parse_args()

    w, h = map(int, args.size.lower().split("x"))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    from .hub import resolve_model
    args.model = str(resolve_model(args.model))
    t0 = time.perf_counter()
    model = load_model(args.model, embed=args.embed)
    tok = PromptTokenizer(args.model)
    t_load = time.perf_counter() - t0
    cfg = GenConfig(width=w, height=h, temperature=args.temperature, max_tokens=args.max_tokens,
                    seed=seeds[0], noise=args.noise)
    if args.eager or args.noise != "vllm":
        results = generate(model, tok, args.prompt, cfg, seeds)
    elif args.unconstrained:
        results = generate_fast(model, tok, args.prompt, cfg, seeds, screen=args.screen, prefill=args.prefill)
    else:
        results = generate_constrained(model, tok, args.prompt, cfg, seeds, prefill=args.prefill)
    del model
    mx.clear_cache()
    t1 = time.perf_counter()
    vq = load_vq(Path(args.model) / "vq", getattr(mx, args.vq_dtype))
    out = Path(args.out)
    for s, r in zip(seeds, results):
        path = out if len(seeds) == 1 else out.with_name(f"{out.stem}_s{s}{out.suffix}")
        meta = {"prompt": args.prompt, "width": w, "height": h, "seed": s, "temperature": args.temperature,
                "noise": args.noise, "model": str(args.model), "timings": dict(r.timings, load_s=t_load),
                "n_tokens": len(r.tokens), "header": r.header, "error": r.error}
        if r.codes is not None:
            img = vq.decode(mx.array(r.codes), r.grid)
            Image.fromarray(L24SQ.to_uint8(img)).save(path)
        if args.save_tokens:
            np.save(path.with_suffix(".tokens.npy"), np.array(r.tokens, np.int32))
        meta["timings"]["decode_vq_s"] = time.perf_counter() - t1
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
