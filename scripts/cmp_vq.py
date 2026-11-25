"""MLX L24SQ decoder/encoder vs the upstream torch module (outputs of ref_vq.py).

Also: round-trip of an official gallery PNG (encode -> decode -> compare), which tells
whether gallery images are exact decoder outputs (then their encoded codes are the
model's own token sequence).
"""
import json, os, sys
import numpy as np, mlx.core as mx
from pathlib import Path
from PIL import Image
from nemotron_image_mlx.vq import load_vq, L24SQ

tokdir, ref_path, gallery = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ref = np.load(ref_path)
idx = mx.array(ref["idx"]); grid = (24, 32)
def psnr(a, b, rng=2.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)); return 10 * np.log10(rng ** 2 / max(mse, 1e-20))
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
out = {"tf32_env": os.environ.get("MLX_ENABLE_TF32")}
for name, dt in (("fp32", mx.float32), ("bf16", mx.bfloat16)):
    m = load_vq(tokdir, dt, with_encoder=True)
    img = np.array(m.decode(idx, grid).astype(mx.float32))
    out[f"decode_{name}_vs_torch_{name}_rel"] = rel(img, ref[f"dec_{name}"])
    out[f"decode_{name}_vs_torch_fp32_psnr_dB"] = psnr(img, ref["dec_fp32"])
    u8m = L24SQ.to_uint8(mx.array(img)); u8t = np.round((np.clip(ref[f"dec_{name}"], -1, 1) + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
    out[f"decode_{name}_uint8_identical_frac"] = float((u8m == u8t).mean())
    if f"enc_{name}" in ref:
        im = np.asarray(Image.open(gallery).convert("RGB")).astype(np.float32) / 255
        x = mx.array(im.transpose(2, 0, 1)) * 2 - 1
        codes, g = m.encode(x)
        codes = np.array(codes)
        out[f"encode_{name}_agree_with_torch_{name}"] = float((codes == ref[f"enc_{name}"]).mean())
        rec = np.array(m.decode(mx.array(codes), g).astype(mx.float32))
        out[f"gallery_roundtrip_{name}_psnr_dB"] = psnr(rec, im.transpose(2, 0, 1) * 2 - 1)
        out[f"gallery_roundtrip_{name}_uint8_identical_frac"] = float((L24SQ.to_uint8(mx.array(rec)) == np.asarray(Image.open(gallery).convert("RGB"))).mean())
    del m
ref_tt = ref["dec_bf16"]; out["torch_bf16_vs_torch_fp32_psnr_dB (floor)"] = psnr(ref_tt, ref["dec_fp32"])
if "enc_fp32" in ref:
    out["torch_encode_bf16_vs_fp32_agree"] = float((ref["enc_bf16"] == ref["enc_fp32"]).mean())
print(json.dumps(out, indent=1))
