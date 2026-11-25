"""Random-weight reference for each NemotronH block type, using the upstream torch module on CPU.

Real dimensions, 16 experts instead of 128 (routing logic is identical). Saves weights (fp32) and
outputs (fp32 and bf16 torch) for: full-sequence forward and a cached decode step.
"""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "shims"))
from nemotron_ref.configuration_nemotron_h import NemotronHConfig
from nemotron_ref import modeling_nemotron_h as M

cfg = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cfg["n_routed_experts"] = 16
cfg["time_step_limit"] = [float(cfg["time_step_limit"][0]), float("inf")]
cfg.pop("auto_map", None)
conf = NemotronHConfig(**{k: v for k, v in cfg.items() if k not in ("architectures",)})
conf._attn_implementation = "eager"
torch.manual_seed(0)
L = 300
D = conf.hidden_size

def init_(blk, kind):
    g = torch.Generator().manual_seed({"M": 1, "*": 2, "E": 3}[kind])
    for n, p in blk.named_parameters():
        if p.ndim >= 2:
            p.data = torch.randn(p.shape, generator=g) * (1.0 / p.shape[-1] ** 0.5)
        elif n.endswith("A_log"):
            p.data = torch.log(torch.rand(p.shape, generator=g) * 15 + 1)
        elif n.endswith("dt_bias"):
            p.data = torch.randn(p.shape, generator=g) * 0.5 - 2
        elif "norm" in n:
            p.data = 1 + 0.1 * torch.randn(p.shape, generator=g)
        else:
            p.data = torch.randn(p.shape, generator=g) * 0.1 + (1.0 if n.endswith(".D") else 0.0)
    for n, b in blk.named_buffers():
        if "e_score_correction_bias" in n:
            b.data = torch.randn(b.shape, generator=g) * 0.02
    if kind == "M":  # conv weight is [C,1,K]
        blk.mixer.conv1d.weight.data = torch.randn(blk.mixer.conv1d.weight.shape, generator=g) * 0.5

res = {}
for kind, layer_idx in (("M", 0), ("*", 5), ("E", 1)):
    assert conf.hybrid_override_pattern[layer_idx] == kind
    blk = M.NemotronHBlock(conf, layer_idx=layer_idx).float().eval()
    init_(blk, kind)
    sd = {k: v.detach().float().numpy() for k, v in blk.state_dict().items()}
    x = torch.randn(1, L, D, generator=torch.Generator().manual_seed(7))
    np.savez(out / f"w_{kind}.npz", **sd)
    for dname, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        b = blk.to(dt)
        if kind == "E":
            b.mixer.gate.weight.data = b.mixer.gate.weight.data.float()
            b.mixer.gate.e_score_correction_bias.data = b.mixer.gate.e_score_correction_bias.data.float()
        xin = x.to(dt)
        with torch.inference_mode():
            y = b(xin)
            res[f"{kind}_full_{dname}"] = y.float().numpy()
            if kind == "E":
                ti, tw = b.mixer.gate(b.norm(xin.to(b.norm.weight.dtype)))
                res[f"E_inds_{dname}"] = ti.numpy(); res[f"E_w_{dname}"] = tw.float().numpy()
    res[f"{kind}_x"] = x.numpy()
    print(kind, "done", flush=True)
np.savez(out / "outputs.npz", **res)
