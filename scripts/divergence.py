"""Where do two token streams of the same prompt/seed first differ? Prints token index and grid position.

  python divergence.py A.tokens.npy B.tokens.npy --w 1024 [--img A.png --img2 B.png --out marked.png]
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--w", type=int, required=True)
ap.add_argument("--img"); ap.add_argument("--img2"); ap.add_argument("--out")
args = ap.parse_args()
a, b = np.load(args.a), np.load(args.b)
n = min(len(a), len(b))
d = next((i for i in range(n) if a[i] != b[i]), None)
VIS_ROW, IMG_START = 131074, 131072
def pos(t, i):
    rows = int(np.sum(t[:i] == VIS_ROW))
    last = max([k for k in range(i) if t[k] == VIS_ROW], default=None)
    col = i - last - 1 if last is not None else None
    return rows, col
if d is None:
    print(f"identical for all {n} tokens (lengths {len(a)}, {len(b)})")
else:
    r, c = pos(a, d)
    print(f"first difference at token {d} of {n}: grid row {r - 1 if r else 'header'}, column {c}; "
          f"{100 * d / n:.1f}% of the stream identical; ids {a[d]} vs {b[d]}")
    if args.img and args.img2 and args.out and r:
        from PIL import Image, ImageDraw
        A = Image.open(args.img).convert("RGB"); B = Image.open(args.img2).convert("RGB")
        y = (r - 1) * 16
        out = Image.new("RGB", (A.width * 2 + 8, A.height), "white")
        out.paste(A, (0, 0)); out.paste(B, (A.width + 8, 0))
        dr = ImageDraw.Draw(out)
        for x0 in (0, A.width + 8):
            dr.line([(x0, y), (x0 + A.width, y)], fill=(255, 0, 0), width=2)
        out.save(args.out)
