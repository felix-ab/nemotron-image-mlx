"""Eager vs compiled vs screened decoding must produce identical token streams (random small model)."""
import time
import mlx.core as mx, mlx.nn as nn, numpy as np
from mlx.utils import tree_flatten, tree_map
from nemotron_image_mlx.model import Args, NemotronImageLM
from nemotron_image_mlx.weights import ArrayEmbedding
from nemotron_image_mlx.pipeline import GenConfig, generate, generate_fast
from nemotron_image_mlx.prompt import PromptTokenizer

a = Args(hidden_size=256, vocab_size=140000, pattern="MEM*EMEM*E", num_attention_heads=4, num_key_value_heads=2,
         head_dim=64, mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=64, n_groups=2, conv_kernel=4,
         moe_intermediate_size=128, moe_shared_expert_intermediate_size=256, n_routed_experts=8, num_experts_per_tok=2)
mx.random.seed(0)
m = NemotronImageLM(a)
def init(p):
    return (mx.random.normal(p.shape) * (0.5 / max(p.shape[-1], 1) ** 0.5 if p.ndim > 1 else 0.1) + (1.0 if p.ndim == 1 else 0)).astype(mx.bfloat16)
m.update(tree_map(init, m.parameters()))
for l in m.layers:
    if l.kind == "M":
        l.mixer.A_neg = -mx.exp(mx.random.uniform(0, 2.5, (a.mamba_num_heads,)))
        l.mixer.dt_bias = (mx.random.normal((a.mamba_num_heads,)) * 0.5 - 2).astype(mx.bfloat16)
    if l.kind == "E":
        l.mixer.e_score_correction_bias = mx.random.normal((a.n_routed_experts,)) * 0.02
m.lm_head.weight = (mx.random.normal((a.vocab_size, a.hidden_size)) * 0.2).astype(mx.bfloat16)
m.embed = ArrayEmbedding((mx.random.normal((a.vocab_size, a.hidden_size))).astype(mx.bfloat16))
mx.eval(m.parameters())
tok = PromptTokenizer("upstream/code")
cfg = GenConfig(width=256, height=256, max_tokens=96)
seeds = [42, 7, 123]
r_e = generate(m, tok, "a test prompt", cfg, seeds)
r_f = generate_fast(m, tok, "a test prompt", cfg, seeds, compile=False)
r_c = generate_fast(m, tok, "a test prompt", cfg, seeds, compile=True)
r_s = generate_fast(m, tok, "a test prompt", cfg, seeds, compile=True, screen=True)
r_p = generate_fast(m, tok, "a test prompt", cfg, seeds, compile=True, prefill="step")
print("step-prefill == eager:", [r_p[j].tokens == r_e[j].tokens for j in range(3)])
for j, s in enumerate(seeds):
    e = r_e[j].tokens
    print(s, "len", len(e), "fast-eager==", r_f[j].tokens == e, "compiled==", r_c[j].tokens == e, "screened==", r_s[j].tokens == e,
          "| first diffs:", [next((i for i, (x, y) in enumerate(zip(e, r.tokens)) if x != y), None) for r in (r_f[j], r_c[j], r_s[j])])
print("screened exact fallbacks:", r_s[0].timings["exact_fallbacks"], "of", 3 * 95)
# single-sequence batch must equal batched rows
r1 = generate_fast(m, tok, "a test prompt", cfg, [7], compile=True)
print("batch row == single:", r1[0].tokens == r_c[1].tokens)
