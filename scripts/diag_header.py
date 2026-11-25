"""Teacher-force the correct size header after a prompt; print p(correct) and top-3 at each header position."""
import json, sys
import mlx.core as mx, numpy as np
from nemotron_image_mlx.pipeline import load_model
from nemotron_image_mlx.prompt import PromptTokenizer
M = sys.argv[1]
model = load_model(M, embed="cpu"); tok = PromptTokenizer(M)
g = {i["id"]: i for i in json.load(open("upstream/gallery/gallery.json"))["items"]}
cases = [("g023", g["023"]["prompt"], 880, 1184), ("fox", "A cinematic photograph of a red fox in a snowy pine forest at dawn.", 880, 1184),
         ("fox_sq", "A cinematic photograph of a red fox in a snowy pine forest at dawn.", 1024, 1024),
         ("fox_land", "A cinematic photograph of a red fox in a snowy pine forest at dawn.", 1248, 832)]
for name, p, w, h in cases:
    pids = tok.encode(p, w, h)
    hdr = tok.tok.encode(f"height={h} width={w}", add_special_tokens=False).ids
    tgt = [1010, tok.ids.img_start] + hdr + [tok.ids.vis_row]
    ids = mx.array([pids + tgt])
    logits = model(ids)[0].astype(mx.float32) / 0.9
    pr = mx.softmax(logits, axis=-1)
    P = len(pids)
    out = []
    for k, t in enumerate(tgt):
        row = pr[P - 1 + k]
        top = np.array(mx.argsort(-row)[:3])
        out.append(f"{tok.tok.decode([t]) if t < 131072 else t!r}:{float(row[t]):.3f} top3=" +
                   ",".join(f"{tok.tok.decode([int(i)]) if i < 131072 else int(i)!r}:{float(row[int(i)]):.2f}" for i in top))
    print(name, w, h, "P=", P); print("   " + "\n   ".join(out), flush=True)
