"""Generation benchmark + image dump.

Measures: cold load, prompt prefill, decode tok/s, VQ decode, peak MLX memory and
peak process footprint; saves PNG, tokens and a JSON record per run.

  python bench_gen.py --model DIR --cases cases.json --out DIR [--mode fast|eager] [--warm N]
cases.json: [{"name", "prompt", "w", "h", "seeds": [..]}]
"""
import argparse, json, os, resource, subprocess, time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from nemotron_image_mlx.pipeline import GenConfig, generate, generate_constrained, generate_fast, load_model
from nemotron_image_mlx.prompt import PromptTokenizer
from nemotron_image_mlx.vq import L24SQ, load_vq


def footprint_gb():
    # phys_footprint of this process (what Activity Monitor reports as Memory)
    try:
        out = subprocess.run(["footprint", "-p", str(os.getpid()), "-f", "bytes"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "phys_footprint:" in line:
                return int(line.split()[-2 if line.split()[-1] == "B" else -1]) / 1e9
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fast", choices=["fast", "eager", "constrained"])
    ap.add_argument("--vq-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-tokens", type=int, default=4352)
    ap.add_argument("--prefill", default="parallel", choices=["parallel", "step"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text())
    rec = {"device": mx.device_info(), "model": args.model, "mode": args.mode, "mlx": mx.__version__,
           "tf32": os.environ.get("MLX_ENABLE_TF32"), "runs": []}
    mx.reset_peak_memory()
    t = time.perf_counter()
    model = load_model(args.model)
    tok = PromptTokenizer(args.model)
    rec["load_s"] = time.perf_counter() - t
    rec["load_peak_mlx_gb"] = mx.get_peak_memory() / 1e9
    rec["active_after_load_gb"] = mx.get_active_memory() / 1e9
    vq = load_vq(Path(args.model) / "vq", getattr(mx, args.vq_dtype))
    ap2 = {"fast": generate_fast, "eager": generate, "constrained": generate_constrained}
    gen = ap2[args.mode]
    for c in cases:
        cfg = GenConfig(width=c["w"], height=c["h"], max_tokens=args.max_tokens)
        mx.reset_peak_memory()
        t = time.perf_counter()
        kw = {"prefill": args.prefill} if args.mode != "eager" else {}
        res = gen(model, tok, c["prompt"], cfg, c["seeds"], **kw)
        t_gen = time.perf_counter() - t
        for s, r in zip(c["seeds"], res):
            name = f"{c['name']}_s{s}"
            row = {"name": name, "case": c["name"], "seed": s, "w": c["w"], "h": c["h"], "gen_s": t_gen,
                   "batch": len(c["seeds"]), **r.timings, "error": r.error, "header": r.header, "n_tokens": len(r.tokens)}
            np.save(out / f"{name}.tokens.npy", np.array(r.tokens, np.int32))
            if r.codes is not None:
                tv = time.perf_counter()
                img = vq.decode(mx.array(r.codes), r.grid)
                u8 = L24SQ.to_uint8(img)
                row["vq_decode_s"] = time.perf_counter() - tv
                Image.fromarray(u8).save(out / f"{name}.png")
            row["peak_mlx_gb"] = mx.get_peak_memory() / 1e9
            row["footprint_gb"] = footprint_gb()
            rec["runs"].append(row)
            print(json.dumps(row), flush=True)
        (out / "bench.json").write_text(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
