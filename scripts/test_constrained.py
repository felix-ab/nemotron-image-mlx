"""Constrained decoding == teacher-forced reference rule (random small model, real vocabulary ids)."""
import mlx.core as mx, numpy as np
from mlx.utils import tree_map
from nemotron_image_mlx.model import Args, NemotronImageLM
from nemotron_image_mlx.weights import ArrayEmbedding
from nemotron_image_mlx.pipeline import GenConfig, generate_constrained, image_schedule
from nemotron_image_mlx.prompt import PromptTokenizer
from nemotron_image_mlx.noise import exponential_row

a = Args(hidden_size=256, vocab_size=327680, pattern="MEM*EMEM*E", num_attention_heads=4, num_key_value_heads=2,
         head_dim=64, mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=64, n_groups=2, conv_kernel=4,
         moe_intermediate_size=128, moe_shared_expert_intermediate_size=256, n_routed_experts=8, num_experts_per_tok=2)
mx.random.seed(1)
m = NemotronImageLM(a)
m.update(tree_map(lambda p: (mx.random.normal(p.shape) * (0.5 / max(p.shape[-1], 1) ** 0.5 if p.ndim > 1 else 0.1) + (1.0 if p.ndim == 1 else 0)).astype(mx.bfloat16), m.parameters()))
for l in m.layers:
    if l.kind == "M":
        l.mixer.A_neg = -mx.exp(mx.random.uniform(0, 2.5, (8,))); l.mixer.dt_bias = (mx.random.normal((8,)) * 0.5 - 2).astype(mx.bfloat16)
    if l.kind == "E":
        l.mixer.e_score_correction_bias = mx.random.normal((8,)) * 0.02
W = mx.random.normal((a.vocab_size, a.hidden_size)) * 0.2
W[131072] = W[131072] * 0 + 1.0          # make <|img_start|> win the first free step
m.lm_head.weight = W.astype(mx.bfloat16)
m.embed = ArrayEmbedding(mx.random.normal((a.vocab_size, a.hidden_size)).astype(mx.bfloat16))
m.update(tree_map(lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p, m.parameters())); m.embed = ArrayEmbedding(m.embed.table.astype(mx.float32)); mx.eval(m.parameters())
tok = PromptTokenizer("upstream/code")
cfg = GenConfig(width=64, height=48, max_tokens=200)
seeds = [42, 9]
res = generate_constrained(m, tok, "test", cfg, seeds)
for j, s in enumerate(seeds):
    r = res[j]
    print("seed", s, "valid:", r.error is None, "header", repr(r.header), "steps", r.timings["steps"], "len", len(r.tokens))
    pids = tok.encode("test", 64, 48); P = len(pids)
    seq = pids + r.tokens
    logits = m(mx.array([seq]))[0].astype(mx.float32) / cfg.temperature
    sched = image_schedule(tok, 64, 48)
    start = r.tokens.index(tok.ids.img_start)
    bad = 0
    for k, t in enumerate(r.tokens):
        l = logits[P - 1 + k]
        q = exponential_row(s, k, a.vocab_size)
        if k <= start:                       # free
            ref = int(mx.argmax(mx.softmax(l) / q))
        else:
            sk = sched[k - start - 1]
            if sk is not None:
                ref = sk
            else:
                lo, hi = tok.ids.vis_start, tok.ids.vis_end
                ref = lo + int(mx.argmax(mx.softmax(l[lo:hi]) / q[lo:hi]))
        bad += ref != t
    print("   tokens disagreeing with teacher-forced reference rule:", bad, "of", len(r.tokens))
