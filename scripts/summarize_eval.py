"""eval_paths.py output -> markdown tables."""
import json, sys
import numpy as np

d = json.load(open(sys.argv[1]))
S = d["summary"]
names = {"bf16": "bf16 (all weights)", "q8": "experts 8-bit", "q6": "experts 6-bit", "q5": "experts 5-bit", "q4": "experts 4-bit"}
def nice(p):
    if p in names:
        return names[p]
    return p.replace("experts=", "experts ").replace("+dense=8", ", mamba/attn 8").replace("+shared=8", ", shared 8").replace("+lm_head=8", ", head 8")
print(f"Sequences: {d['n_seqs']}, generated tokens scored: {sum(r['n_tokens'] for r in d['per_seq'])}, device {d['device']}, truth = {d['truth']}\n")
print("| Path | KL vs fp32 (nats/token) | KL vs bf16 | top-1 = fp32 | sample flips vs fp32 (per 1k tokens) | flips vs bf16 (per 1k) | mean tokens to first flip vs fp32 | NLL - fp32 NLL (nats/token) |")
print("|---|---|---|---|---|---|---|---|")
for p, s in S.items():
    kb = s.get("kl_vs_bf16"); fb = s.get("flip_rate_vs_bf16")
    print(f"| {nice(p)} | {s['kl']:.2e} | {'' if kb is None else f'{kb:.2e}'} | {100 * s['top1']:.2f}% | {1000 * s['flip_rate']:.2f} | "
          f"{'' if fb is None else f'{1000 * fb:.2f}'} | {s['mean_tokens_to_first_flip']:.0f} | {s['nll'] - s['nll_truth']:+.4f} |")
print()
# per-sequence-group breakdown
groups = {"gallery prompts": [r for r in d["per_seq"] if "gallery" in r["name"]], "visual suite": [r for r in d["per_seq"] if "suite" in r["name"]]}
print("| Path | " + " | ".join(f"flips/1k vs fp32: {g}" for g in groups) + " |")
print("|---|" + "---|" * len(groups))
for p in S:
    cells = []
    for g, rows in groups.items():
        n = sum(r["n_tokens"] for r in rows)
        cells.append(f"{1000 * sum(r[p]['flip_rate'] * r['n_tokens'] for r in rows) / max(n, 1):.2f}")
    print(f"| {nice(p)} | " + " | ".join(cells) + " |")
print()
# hidden-state drift at selected layers
L = d["layers"]
sel = [0, 12, 25, 38, 51]
print("| Path | " + " | ".join(f"rel. hidden diff vs fp32, layer {i}" for i in sel) + " |")
print("|---|" + "---|" * len(sel))
for p in S:
    print(f"| {nice(p)} | " + " | ".join(f"{L[i]['rel_hidden'].get(p, float('nan')):.2e}" for i in sel) + " |")
if d.get("torch_check"):
    tc = [t["rel_mlx_fp32_vs_torch_fp32"] for t in d["torch_check"]]
    print(f"\nMLX fp32 vs upstream torch fp32 (first 400 tokens of sequence 0), per layer: median {np.median(tc):.1e}, max {max(tc):.1e} (layer {int(np.argmax(tc))})")
