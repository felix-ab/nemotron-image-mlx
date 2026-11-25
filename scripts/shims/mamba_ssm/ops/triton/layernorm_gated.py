"""CPU stand-in for mamba_ssm's Triton gated RMSNorm (copy of mamba_ssm.ops.triton.layernorm_gated.rms_norm_ref)."""
import torch
import torch.nn.functional as F


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        xg = x.reshape(*x.shape[:-1], -1, group_size)
        rstd = 1 / torch.sqrt((xg.square()).mean(dim=-1, keepdim=True) + eps)
        out = (xg * rstd).reshape(*x.shape) * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)
