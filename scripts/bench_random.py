"""Decode-speed benchmark with full-size random weights (real shapes and kernels, no checkpoint needed).

  python bench_random.py --recipe q4 --tokens 512 --modes eager,compiled,screened [--batch 1]
"""
import argparse, gc, json, time
import mlx.core as mx, mlx.nn as nn, numpy as np
from nemotron_image_mlx.model import Args, NemotronImageLM
from nemotron_image_mlx.weights import parse_recipe, quantize_module
from nemotron_image_mlx.pipeline import GenConfig, generate, generate_constrained, generate_fast
from nemotron_image_mlx.prompt import PromptTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="upstream/code")
ap.add_argument("--recipe", default="q4")
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--modes", default="eager,compiled,screened")
ap.add_argument("--batch", default="1")
ap.add_argument("--out", default=None)
args = ap.parse_args()
a = Args.load(args.config)
rec = parse_recipe(args.recipe)
m = NemotronImageLM(a)
t = time.perf_counter()
for blk in m.layers:
    blk.set_dtype(mx.bfloat16)
    if blk.kind == "M":
        blk.mixer.A_neg = -mx.exp(mx.random.uniform(0, 2.5, (a.mamba_num_heads,)))
        blk.mixer.dt_bias = (mx.random.normal((a.mamba_num_heads,)) * 0.5 - 2).astype(mx.bfloat16)
    if blk.kind == "E":
        blk.mixer.e_score_correction_bias = mx.zeros((a.n_routed_experts,))
    quantize_module(blk, rec)
    mx.eval(blk.parameters()); gc.collect(); mx.clear_cache()
m.norm_f_weight = mx.ones((a.hidden_size,), mx.bfloat16)
m.lm_head.set_dtype(mx.bfloat16)
if rec.get("lm_head"):
    b, g, md = rec["lm_head"]; m.lm_head = m.lm_head.to_quantized(group_size=g, bits=b, mode=md)
mx.eval(m.parameters())
table = (np.random.default_rng(0).standard_normal((a.vocab_size, a.hidden_size), dtype=np.float32) * 0.02)
table = np.array(mx.array(table).astype(mx.bfloat16).view(mx.uint16))
class SyncEmbed:  # same CPU-side row gather as the memmap embedding
    def __call__(self, ids):
        idx = np.array(ids); return mx.array(np.ascontiguousarray(table[idx.reshape(-1)])).view(mx.bfloat16).reshape(*idx.shape, -1)
m.embed = SyncEmbed()
from nemotron_image_mlx.weights import ArrayEmbedding
gpu_embed = None
if "gpuembed" in args.modes:
    gpu_embed = ArrayEmbedding(mx.array(table).view(mx.bfloat16)); mx.eval(gpu_embed.table)
res = {"device": mx.device_info()["device_name"], "recipe": args.recipe, "active_gb": mx.get_active_memory() / 1e9,
       "build_s": time.perf_counter() - t, "runs": []}
tok = PromptTokenizer(args.config)
prompt = "A cinematic photograph of a red fox in a snowy pine forest at dawn."
tw = time.perf_counter()
m.embed = gpu_embed or SyncEmbed()
generate_fast(m, tok, prompt, GenConfig(max_tokens=16), [42], compile=True)   # JIT warm-up (cold-start cost)
res["warmup_s"] = time.perf_counter() - tw
for B in map(int, args.batch.split(",")):
    for mode in args.modes.split(","):
        cfg = GenConfig(max_tokens=args.tokens)
        seeds = list(range(42, 42 + B))
        mx.reset_peak_memory()
        m.embed = gpu_embed if mode.endswith("gpuembed") else SyncEmbed()
        if mode.startswith("constrained"):
            if not getattr(m, "_biased", False):   # make <|img_start|> win the first free step so the schedule engages
                if hasattr(m.lm_head, "scales"):
                    m.lm_head.biases[131072] = m.lm_head.biases[131072] + 50.0
                else:
                    m.lm_head.weight[131072] = m.lm_head.weight[131072] * 0 + 1.0
                m._biased = True
            r = generate_constrained(m, tok, prompt, cfg, seeds)
        elif mode == "eager":
            r = generate(m, tok, prompt, cfg, seeds)
        else:
            r = generate_fast(m, tok, prompt, cfg, seeds, compile=True, screen=(mode.startswith("screened")))
        tm = r[0].timings
        if mode.startswith("constrained"):
            print("   steps:", tm.get("steps"), "valid:", r[0].error is None or r[0].error[:60])
        row = {"mode": mode, "batch": B, "tok_s_per_seq": tm["tok_per_s"], "agg_tok_s": tm["tok_per_s"] * B,
               "ms_per_step": 1000 / tm["tok_per_s"], "prefill_s": tm["prefill_s"], "peak_gb": mx.get_peak_memory() / 1e9,
               "exact_fallbacks": tm.get("exact_fallbacks")}
        res["runs"].append(row); print(json.dumps(row), flush=True)
print(json.dumps({k: v for k, v in res.items() if k != "runs"}))
if args.out:
    open(args.out, "w").write(json.dumps(res, indent=1))
