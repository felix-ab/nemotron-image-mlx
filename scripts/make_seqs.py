"""Build teacher-forcing sequences (prompt ids + generated tokens) for eval_paths.py.

  python make_seqs.py --model DIR --cases cases.json --runs RUN_DIR [RUN_DIR ...] --out seqs.json
Each run dir holds <case>_s<seed>.tokens.npy from bench_gen.py.
"""
import argparse, json
from pathlib import Path

import numpy as np

from nemotron_image_mlx.prompt import PromptTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--cases", required=True, nargs="+")
ap.add_argument("--runs", required=True, nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--max", type=int, default=999)
args = ap.parse_args()
tok = PromptTokenizer(args.model)
cases = {}
for f in args.cases:
    for c in json.loads(Path(f).read_text()):
        cases[c["name"]] = c
seqs = []
for d in args.runs:
    for f in sorted(Path(d).glob("*.tokens.npy")):
        name = f.name[: -len(".tokens.npy")]
        cname, seed = name.rsplit("_s", 1)
        c = cases[cname]
        pids = tok.encode(c["prompt"], c["w"], c["h"])
        gen = np.load(f).tolist()
        if gen[-1] != tok.ids.img_end:
            print("skip (no img_end):", name)
            continue
        seqs.append({"name": f"{Path(d).name}/{name}", "ids": pids + gen, "prompt_len": len(pids),
                     "seed": int(seed), "w": c["w"], "h": c["h"]})
seqs = seqs[: args.max]
Path(args.out).write_text(json.dumps(seqs))
print(len(seqs), "sequences;", sum(len(s["ids"]) for s in seqs), "tokens")
