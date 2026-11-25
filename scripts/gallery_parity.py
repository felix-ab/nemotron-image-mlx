"""Row-band comparison of our seed-N renders with the official gallery images.

For each token row (16 px band) we report PSNR against the official PNG. With the same
noise stream and close logits, an autoregressive render matches the reference from the
top until the first sampled token differs, then diverges. Unrelated images of the same
prompt sit near 10-13 dB per band.

  python gallery_parity.py --cases cases.json --ours DIR --gallery DIR --out parity.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--cases", required=True); ap.add_argument("--ours", required=True)
ap.add_argument("--gallery", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
res = []
for c in json.loads(Path(args.cases).read_text()):
    p = Path(args.ours) / f"{c['name']}_s{args.seed}.png"
    g = Path(args.gallery) / c["gallery"]
    if not p.exists() or not g.exists():
        continue
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
    b = np.asarray(Image.open(g).convert("RGB")).astype(np.float64)
    if a.shape != b.shape:
        res.append({"name": c["name"], "error": f"shape {a.shape} vs {b.shape}"}); continue
    bands = []
    for r in range(0, a.shape[0], 16):
        mse = np.mean((a[r:r + 16] - b[r:r + 16]) ** 2)
        bands.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    full = 10 * np.log10(255 ** 2 / np.mean((a - b) ** 2))
    match = next((i for i, v in enumerate(bands) if v < 20), len(bands))
    res.append({"name": c["name"], "psnr_full": full, "rows_matching_from_top(>=20dB)": match, "n_rows": len(bands),
                "band_psnr_first8": [round(v, 1) for v in bands[:8]], "band_psnr_median": float(np.median(bands))})
    print(json.dumps(res[-1]))
Path(args.out).write_text(json.dumps(res, indent=1))
