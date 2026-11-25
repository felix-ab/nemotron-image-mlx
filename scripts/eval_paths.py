"""Layer-major, multi-path teacher-forced fidelity evaluation.

Each layer's fp32 weights are read once (local checkpoint or streamed from the Hub)
and every precision path (fp32 truth, bf16, quantized tiers) is built from them and
run over all sequences. At the head, per-token statistics are computed against the
truth path:

* kl      KL(truth || path) of the sampling distribution softmax(logits / T)
* top1    argmax agreement
* flip    noise-coupled sample disagreement: both paths sample with the same vLLM
          seed-42 exponential noise for that decode step (expected divergence rate
          of a seeded generation)
* nll     NLL of the sequence's own next token (mean over the image region)

Usage:
  MLX_ENABLE_TF32=0 python eval_paths.py --src <ckpt dir> | --hub <repo> --meta <dir>
       --seqs seqs.json --paths fp32,bf16,q8,q6,q5,q4 --out results.json
"""
import argparse, gc, json, sys, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nemotron_image_mlx.model import Args, Block, rms_norm
from nemotron_image_mlx.weights import Upstream, block_tensors, parse_recipe, quantize_module
from nemotron_image_mlx.noise import exponential_row

F32, BF16 = mx.float32, mx.bfloat16


def path_spec(name):
    if name == "fp32":
        return F32, {}
    if name == "bf16":
        return BF16, {}
    return BF16, parse_recipe(name.replace("+", ","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src"); ap.add_argument("--hub"); ap.add_argument("--meta")
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--paths", default="fp32,bf16,q8")
    ap.add_argument("--truth", default="fp32")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-hidden", default=None, help="dir to save final hidden states per path (npz)")
    ap.add_argument("--torch-check", type=int, default=0, help="also run the upstream torch module (CPU, fp32) on the first N tokens of seq 0")
    args = ap.parse_args()

    if args.hub:
        from nemotron_image_mlx.remote import RemoteUpstream
        up = RemoteUpstream(args.hub)
        meta = Path(args.meta)
    else:
        up = Upstream(args.src)
        meta = Path(args.src)
    a = Args.load(meta)
    seqs = json.loads(Path(args.seqs).read_text())   # [{"name", "ids", "prompt_len", ...}]
    paths = args.paths.split(",")
    assert args.truth in paths
    specs = {p: path_spec(p) for p in paths}
    t0 = time.time()

    # embeddings (fp32 rows -> per-path dtype)
    E = up.get("backbone.embeddings.weight")
    H = {p: [mx.array(E[np.array(s["ids"])][None]).astype(specs[p][0]) for s in seqs] for p in paths}
    del E
    mx.eval(H)
    log = {"paths": paths, "truth": args.truth, "n_seqs": len(seqs), "layers": []}
    th = None
    if args.torch_check:
        import torch
        sys.path.insert(0, str(Path(__file__).parent / "shims"))
        from nemotron_ref.configuration_nemotron_h import NemotronHConfig
        from nemotron_ref import modeling_nemotron_h as HFM
        cfgd = json.loads((meta / "config.json").read_text())
        cfgd["time_step_limit"] = [float(cfgd["time_step_limit"][0]), float("inf")]
        cfgd.pop("auto_map", None); cfgd.pop("architectures", None)
        hfconf = NemotronHConfig(**cfgd); hfconf._attn_implementation = "eager"
        Lt = args.torch_check
        th = torch.from_numpy(np.array(H["fp32"][0][:, :Lt].astype(F32)).copy())
        log["torch_check"] = []

    for i, kind in enumerate(a.pattern):
        tl = time.time()
        prefix = f"backbone.layers.{i}."
        if hasattr(up, "prefetch"):
            raw = up.prefetch([n for n in up.map if n.startswith(prefix)])
            get = lambda n: raw[prefix + n]
        else:
            get = lambda n: up.get(prefix + n)
        layer_rel = {}
        for p in paths:
            dt, rec = specs[p]
            blk = Block(a, kind)
            blk.load_weights(list(block_tensors(get, kind, a, dt).items()), strict=True)
            if rec:
                quantize_module(blk, rec)
            mx.eval(blk.parameters())
            for j in range(len(seqs)):
                H[p][j] = blk(H[p][j], None)
                mx.eval(H[p][j])
            del blk
            gc.collect(); mx.clear_cache()
        # hidden-state drift vs truth at this layer
        for p in paths:
            if p == args.truth:
                continue
            num = sum(float(mx.sum((H[p][j].astype(F32) - H[args.truth][j].astype(F32)) ** 2)) for j in range(len(seqs)))
            den = sum(float(mx.sum(H[args.truth][j].astype(F32) ** 2)) for j in range(len(seqs)))
            layer_rel[p] = (num / den) ** 0.5
        if th is not None:
            import torch
            blk = HFM.NemotronHBlock(hfconf, layer_idx=i).float().eval()
            sd = {n[len(prefix):]: torch.from_numpy(np.array(get(n[len(prefix):]), dtype=np.float32))
                  for n in (raw if hasattr(up, "prefetch") else [k for k in up.map if k.startswith(prefix)])}
            missing, unexpected = blk.load_state_dict(sd, strict=False)
            assert not unexpected and all("e_score" not in m for m in missing), (missing, unexpected)
            with torch.inference_mode():
                th = blk(th)
            ml = np.array(H["fp32"][0][:, :th.shape[1]].astype(F32))
            rel_t = float(np.linalg.norm(th.numpy() - ml) / np.linalg.norm(th.numpy()))
            log["torch_check"].append({"i": i, "kind": kind, "rel_mlx_fp32_vs_torch_fp32": rel_t})
            layer_rel["torch_fp32_vs_mlx_fp32"] = rel_t
            del blk, sd
        if hasattr(up, "prefetch"):
            del raw
        log["layers"].append({"i": i, "kind": kind, "rel_hidden": layer_rel, "s": round(time.time() - tl, 1)})
        print(f"layer {i:2d} {kind} {time.time() - tl:6.1f}s " + " ".join(f"{p}:{v:.2e}" for p, v in layer_rel.items()), flush=True)

    # head
    nw = up.get("backbone.norm_f.weight"); lm = up.get("lm_head.weight")
    heads = {}
    for p in paths:
        dt, rec = specs[p]
        l = nn.Linear(a.hidden_size, a.vocab_size, bias=False)
        l.weight = mx.array(lm).astype(dt)
        if rec.get("lm_head"):
            bits, gs, mode = rec["lm_head"]
            l = l.to_quantized(group_size=gs, bits=bits, mode=mode)
        heads[p] = (mx.array(nw).astype(dt), l)
        mx.eval(l.parameters())
    del lm
    T = args.temperature
    stats = {p: {"kl": [], "top1": [], "flip": [], "nll": [], "nll_truth": []} for p in paths if p != args.truth}
    per_seq = []
    for j, s in enumerate(seqs):
        P = s["prompt_len"]
        ids = np.array(s["ids"])
        L = len(ids)
        pos = np.arange(P - 1, L - 1)            # positions whose logits sample a generated token
        acc = {p: {"kl": 0.0, "top1": 0, "flip": 0, "nll": 0.0, "nll_t": 0.0} for p in stats}
        # header detail: top-5 of the sampling distribution at the first 16 generated positions
        hp = pos[:16]
        hdet = {}
        for p in [x for x in (args.truth, "bf16", "q8") if x in paths]:
            nwp, l = heads[p]
            lg = l(rms_norm(H[p][j][0, mx.array(hp)], nwp, a.eps)[None])[0].astype(F32) / T
            pr = mx.softmax(lg, axis=-1)
            top = mx.argsort(-pr, axis=-1)[:, :5]
            hdet[p] = [{"pos": int(t), "target": int(ids[t + 1]), "p_target": float(pr[k, int(ids[t + 1])]),
                        "top5": [[int(i), round(float(pr[k, int(i)]), 5)] for i in np.array(top[k])]} for k, t in enumerate(hp)]
        for c0 in range(0, len(pos), args.chunk):
            cp = pos[c0:c0 + args.chunk]
            tgt = mx.array(ids[cp + 1])
            q = mx.stack([exponential_row(int(s.get("seed", 42)), int(t - (P - 1)), a.vocab_size) for t in cp])
            def logits(p):
                nwp, l = heads[p]
                h = H[p][j][0, mx.array(cp)]
                return l(rms_norm(h, nwp, a.eps)[None])[0].astype(F32) / T
            lt = logits(args.truth)
            lpt = lt - mx.logsumexp(lt, axis=-1, keepdims=True)
            st = mx.argmax(mx.exp(lpt) / q, axis=-1)
            if "bf16" in paths and args.truth != "bf16":
                lb = logits("bf16"); lpb = lb - mx.logsumexp(lb, axis=-1, keepdims=True)
                sb = mx.argmax(mx.exp(lpb) / q, axis=-1)
            else:
                lpb = sb = None
            at = mx.argmax(lt, axis=-1)
            nll_t = -mx.take_along_axis(lpt, tgt[:, None], axis=-1)[:, 0]
            for p in stats:
                lp_ = logits(p)
                lpp = lp_ - mx.logsumexp(lp_, axis=-1, keepdims=True)
                kl = mx.sum(mx.exp(lpt) * (lpt - lpp), axis=-1)
                sp = mx.argmax(mx.exp(lpp) / q, axis=-1)
                nll = -mx.take_along_axis(lpp, tgt[:, None], axis=-1)[:, 0]
                r = acc[p]
                r["kl"] += float(kl.sum()); r["top1"] += int((mx.argmax(lp_, -1) == at).sum())
                r["flip"] += int((sp != st).sum()); r["nll"] += float(nll.sum()); r["nll_t"] += float(nll_t.sum())
                if sb is not None and p != "bf16":
                    r["flip_b"] = r.get("flip_b", 0) + int((sp != sb).sum())
                    r["kl_b"] = r.get("kl_b", 0.0) + float(mx.sum(mx.exp(lpb) * (lpb - lpp), axis=-1).sum())
                if c0 == 0:
                    r["first_flip"] = None
                if r.get("first_flip") is None:
                    nz = np.nonzero(np.array(sp != st))[0]
                    if len(nz):
                        r["first_flip"] = int(c0 + nz[0])
            mx.clear_cache()
        n = len(pos)
        row = {"name": s["name"], "n_tokens": n, "header_detail": hdet}
        for p, r in acc.items():
            row[p] = {"kl": r["kl"] / n, "top1": r["top1"] / n, "flip_rate": r["flip"] / n,
                      "nll": r["nll"] / n, "nll_truth": r["nll_t"] / n, "first_flip": r.get("first_flip"),
                      "flip_rate_vs_bf16": r["flip_b"] / n if "flip_b" in r else None,
                      "kl_vs_bf16": r["kl_b"] / n if "kl_b" in r else None}
        per_seq.append(row)
        print(json.dumps(row), flush=True)
    summ = {}
    for p in stats:
        tot = sum(r["n_tokens"] for r in per_seq)
        summ[p] = {k: sum(r[p][k] * r["n_tokens"] for r in per_seq) / tot for k in ("kl", "top1", "flip_rate", "nll", "nll_truth")}
        for k in ("flip_rate_vs_bf16", "kl_vs_bf16"):
            if per_seq and per_seq[0][p].get(k) is not None:
                summ[p][k] = sum(r[p][k] * r["n_tokens"] for r in per_seq) / tot
        summ[p]["mean_tokens_to_first_flip"] = float(np.mean([r[p]["first_flip"] if r[p]["first_flip"] is not None else r["n_tokens"] for r in per_seq]))
    log.update({"summary": summ, "per_seq": per_seq, "wall_s": time.time() - t0,
                "device": mx.device_info()["device_name"], "tf32": __import__("os").environ.get("MLX_ENABLE_TF32")})
    Path(args.out).write_text(json.dumps(log, indent=1))
    print(json.dumps(summ, indent=1))
    if args.save_hidden:
        d = Path(args.save_hidden); d.mkdir(parents=True, exist_ok=True)
        for p in paths:
            np.savez(d / f"{p}.npz", *[np.array(h.astype(F32)) for h in H[p]])


if __name__ == "__main__":
    main()
