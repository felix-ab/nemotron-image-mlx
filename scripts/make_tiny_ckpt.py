"""Synthetic small checkpoint in the upstream layout (for dry-running eval/convert tooling)."""
import json, sys, numpy as np
from pathlib import Path
from safetensors.numpy import save_file
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open("upstream/code/config.json"))
cfg.update({"hidden_size": 256, "hybrid_override_pattern": "MEM*E", "num_hidden_layers": 5, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 64, "mamba_num_heads": 8, "mamba_head_dim": 32, "ssm_state_size": 64,
            "n_groups": 2, "moe_intermediate_size": 128, "moe_shared_expert_intermediate_size": 256, "n_routed_experts": 8,
            "num_experts_per_tok": 2, "vocab_size": 2048, "intermediate_size": 128})
json.dump(cfg, open(out / "config.json", "w"), indent=1)
r = np.random.default_rng(0); T = {}
D = 256
def W(o, i): return (r.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)
T["backbone.embeddings.weight"] = r.standard_normal((2048, D)).astype(np.float32)
T["lm_head.weight"] = (W(2048, D) * 3).astype(np.float32)
T["backbone.norm_f.weight"] = (1 + 0.1 * r.standard_normal(D)).astype(np.float32)
for i, k in enumerate(cfg["hybrid_override_pattern"]):
    p = f"backbone.layers.{i}."; m = p + "mixer."
    T[p + "norm.weight"] = (1 + 0.1 * r.standard_normal(D)).astype(np.float32)
    if k == "M":
        inner, conv = 256, 256 + 2 * 2 * 64
        T[m + "in_proj.weight"] = W(inner + conv + 8, D); T[m + "out_proj.weight"] = W(D, inner)
        T[m + "conv1d.weight"] = (r.standard_normal((conv, 1, 4)) * 0.5).astype(np.float32)
        T[m + "conv1d.bias"] = (r.standard_normal(conv) * 0.1).astype(np.float32)
        T[m + "dt_bias"] = (r.standard_normal(8) * 0.5 - 2).astype(np.float32)
        T[m + "A_log"] = np.log(r.uniform(1, 16, 8)).astype(np.float32)
        T[m + "D"] = np.ones(8, np.float32); T[m + "norm.weight"] = (1 + 0.1 * r.standard_normal(inner)).astype(np.float32)
    elif k == "*":
        T[m + "q_proj.weight"] = W(256, D); T[m + "k_proj.weight"] = W(128, D); T[m + "v_proj.weight"] = W(128, D); T[m + "o_proj.weight"] = W(D, 256)
    else:
        T[m + "gate.weight"] = W(8, D); T[m + "gate.e_score_correction_bias"] = (r.standard_normal(8) * 0.02).astype(np.float32)
        for e in range(8):
            T[m + f"experts.{e}.up_proj.weight"] = W(128, D); T[m + f"experts.{e}.down_proj.weight"] = W(D, 128)
        T[m + "shared_experts.up_proj.weight"] = W(256, D); T[m + "shared_experts.down_proj.weight"] = W(D, 256)
save_file(T, str(out / "model-00001-of-00001.safetensors"))
json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in T}}, open(out / "model.safetensors.index.json", "w"))
seqs = [{"name": f"t{j}", "ids": r.integers(0, 2048, 300).tolist(), "prompt_len": 40, "seed": 42} for j in range(2)]
json.dump(seqs, open(out / "seqs.json", "w"))
print("ok", len(T))
