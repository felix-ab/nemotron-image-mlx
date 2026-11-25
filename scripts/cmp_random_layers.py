"""MLX blocks vs the upstream torch module on random weights (see ref_random_layers.py)."""
import json, sys, numpy as np, mlx.core as mx
from pathlib import Path
from dataclasses import replace
from nemotron_image_mlx.model import Args, Block, MambaCache, KVCache
from nemotron_image_mlx.weights import block_tensors

d = Path(sys.argv[2]); a = replace(Args.from_config(json.load(open(sys.argv[1]))), n_routed_experts=16)
ref = np.load(d / "outputs.npz")
def rel(x, y): return float(np.linalg.norm(x - y) / np.linalg.norm(y))
def cos(x, y): x, y = x.ravel().astype(np.float64), y.ravel().astype(np.float64); return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
rows = []
for kind in ("M", "*", "E"):
    sd = dict(np.load(d / f"w_{kind}.npz"))
    x = ref[f"{kind}_x"]
    out = {}
    for dname, dt in (("fp32", mx.float32), ("bf16", mx.bfloat16)):
        b = Block(a, kind)
        b.load_weights(list(block_tensors(lambda n: sd[n], kind, a, dt, 16).items()), strict=True)
        xm = mx.array(x).astype(dt)
        y = b(xm, MambaCache() if kind == "M" else KVCache() if kind == "*" else None)
        out[dname] = np.array(y.astype(mx.float32))
        # decode consistency: prefill L-1, then one step
        if kind in "M*":
            c = MambaCache() if kind == "M" else KVCache()
            b(xm[:, :-1], c)
            ys = np.array(b(xm[:, -1:], c).astype(mx.float32))
            rows.append((kind, dname, "step(L-1) vs own full[L-1]", rel(ys[0, -1], out[dname][0, -1])))
            rows.append((kind, dname, "step(L-1) vs torch full[L-1]", rel(ys[0, -1], ref[f"{kind}_full_{dname}"][0, -1])))
        if kind == "E":
            from nemotron_image_mlx.model import rms_norm
            inds, w = b.mixer.route(rms_norm(xm, b.norm_weight, a.eps))
            ri = np.sort(ref[f"E_inds_{dname}"], -1); mi = np.sort(np.array(inds)[0], -1)
            rows.append((kind, dname, "routing: tokens with identical top-6 set", float((ri == mi).all(-1).mean())))
    t32, t16 = ref[f"{kind}_full_fp32"], ref[f"{kind}_full_bf16"]
    rows.append((kind, "fp32", "MLX vs torch (fp32)", rel(out["fp32"], t32)))
    rows.append((kind, "bf16", "MLX bf16 vs torch fp32", rel(out["bf16"], t32)))
    rows.append((kind, "bf16", "torch bf16 vs torch fp32 (floor)", rel(t16, t32)))
    rows.append((kind, "bf16", "MLX bf16 vs torch bf16", rel(out["bf16"], t16)))
for r in rows: print(f"{r[0]:2s} {r[1]:5s} {r[2]:42s} {r[3]:.3e}")
