"""Contact sheets: rows = cases, columns = variants (optionally blinded), plus 100% crops.

  python contact_sheet.py --dirs A=dirA B=dirB ... --out sheet.jpg [--blind key.json] [--crop 256]
Images are matched by file name across dirs.
"""
import argparse, json, random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--blind", default=None, help="write the column key here and label columns A, B, C...")
ap.add_argument("--thumb", type=int, default=512)
ap.add_argument("--crop", type=int, default=0, help="add a 100%% center crop of this size under each thumb")
ap.add_argument("--names", nargs="*", default=None)
args = ap.parse_args()
dirs = [d.split("=", 1) for d in args.dirs]
names = args.names or sorted({p.name for _, d in dirs for p in Path(d).glob("*.png")})
names = [n for n in names if all((Path(d) / n).exists() for _, d in dirs)]
cols = list(range(len(dirs)))
key = {}
rng = random.Random(0)
font = ImageFont.load_default()
T, C = args.thumb, args.crop
cell_h = T + (C if C else 0) + 24
sheet = Image.new("RGB", (len(dirs) * (T + 8) + 8, len(names) * (cell_h + 8) + 8), "white")
dr = ImageDraw.Draw(sheet)
for r, n in enumerate(names):
    order = cols[:]
    if args.blind:
        rng.shuffle(order)
        key[n] = {chr(65 + i): dirs[o][0] for i, o in enumerate(order)}
    for ci, o in enumerate(order):
        label, d = dirs[o]
        im = Image.open(Path(d) / n).convert("RGB")
        x0, y0 = 8 + ci * (T + 8), 8 + r * (cell_h + 8)
        th = im.copy(); th.thumbnail((T, T))
        sheet.paste(th, (x0, y0))
        if C:
            w, h = im.size
            cr = im.crop((w // 2 - C // 2, h // 2 - C // 2, w // 2 + C // 2, h // 2 + C // 2))
            sheet.paste(cr, (x0, y0 + T))
        dr.text((x0, y0 + T + (C if C else 0) + 4), f"{chr(65 + ci) if args.blind else label}  {n}", fill="black", font=font)
sheet.save(args.out, quality=92)
if args.blind:
    Path(args.blind).write_text(json.dumps(key, indent=1))
print(args.out, len(names), "rows")
