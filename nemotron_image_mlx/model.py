"""Nemotron-Labs-Image-30B-A3B (NemotronH hybrid Mamba2 / attention / MoE) in MLX.

Numerics follow the reference generation path, vLLM 0.21 in bfloat16 on H100
(`vllm/model_executor/models/nemotron_h.py`, `layers/mamba/mamba_mixer2.py`,
`csrc/layernorm_kernels.cu`, `fused_moe/router/gate_linear.py`):

* residual stream in bf16, each add rounded once;
* RMSNorm reduced in fp32, ``x * rsqrt(var + eps) * w`` rounded once, with the
  bf16 copy of the weight;
* router: bf16 activations x bf16 weight, fp32 accumulation, fp32 logits
  (cuBLAS ``out_dtype=float32`` on SM90), sigmoid + correction bias in fp32,
  top-6 of 128, renormalised, routed output scaled by 2.5;
* experts: ReLU^2 MLPs; shared expert added after the routed sum;
* Mamba2: causal conv + SiLU in fp32, softplus(dt + dt_bias) and the SSM
  recurrence in fp32 with an fp32 state, gated RMSNorm over 8 groups in fp32
  then rounded and multiplied by the bf16 weight;
* attention: GQA 32/2, no positional encoding (positions come from Mamba);
* lm_head logits in bf16, upcast to fp32 for sampling.

``dtype=mx.float32`` runs every op and weight in fp32 (the truth reference for
validation).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear  # noqa: F401


@dataclass
class Args:
    hidden_size: int = 2688
    vocab_size: int = 327680
    pattern: str = ""
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    mamba_num_heads: int = 64
    mamba_head_dim: int = 64
    ssm_state_size: int = 128
    n_groups: int = 8
    conv_kernel: int = 4
    moe_intermediate_size: int = 1856
    moe_shared_expert_intermediate_size: int = 3712
    n_routed_experts: int = 128
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    eps: float = 1e-5
    time_step_limit: tuple = (0.0, float("inf"))

    @classmethod
    def from_config(cls, cfg: dict) -> "Args":
        assert cfg.get("n_group", 1) == 1 and cfg.get("topk_group", 1) == 1
        assert not cfg.get("attention_bias") and not cfg.get("mlp_bias") and not cfg.get("use_bias")
        assert cfg.get("mlp_hidden_act") == "relu2" and cfg.get("mamba_hidden_act") == "silu"
        tsl = cfg.get("time_step_limit", [0.0, float("inf")])
        hi = tsl[1]
        if isinstance(hi, dict):
            hi = float(hi.get("__float__", "inf"))
        return cls(
            hidden_size=cfg["hidden_size"],
            vocab_size=cfg["vocab_size"],
            pattern=cfg["hybrid_override_pattern"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            mamba_num_heads=cfg["mamba_num_heads"],
            mamba_head_dim=cfg["mamba_head_dim"],
            ssm_state_size=cfg["ssm_state_size"],
            n_groups=cfg["n_groups"],
            conv_kernel=cfg["conv_kernel"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            moe_shared_expert_intermediate_size=cfg["moe_shared_expert_intermediate_size"],
            n_routed_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            routed_scaling_factor=cfg["routed_scaling_factor"],
            norm_topk_prob=cfg["norm_topk_prob"],
            eps=cfg["layer_norm_epsilon"],
            time_step_limit=(float(tsl[0]), float(hi)),
        )

    @classmethod
    def load(cls, path) -> "Args":
        return cls.from_config(json.loads((Path(path) / "config.json").read_text()))


F32 = mx.float32

_FP32_GEMM_EXACT = None


def fp32_gemm_exact() -> bool:
    """True when float32 GEMMs (M > 1) run in full fp32 here (not TF32). Probed once."""
    global _FP32_GEMM_EXACT
    if _FP32_GEMM_EXACT is None:
        import numpy as _np
        r = _np.random.default_rng(0)
        a = r.standard_normal((8, 512)).astype(_np.float32)
        b = r.standard_normal((512, 64)).astype(_np.float32)
        ref = a.astype(_np.float64) @ b.astype(_np.float64)
        c = _np.array(mx.array(a) @ mx.array(b))
        _FP32_GEMM_EXACT = bool(_np.linalg.norm(c - ref) / _np.linalg.norm(ref) < 1e-5)
    return _FP32_GEMM_EXACT


def matmul_fp32(x: mx.array, wT: mx.array) -> mx.array:
    """x @ wT in true fp32 even where fp32 GEMMs default to TF32 (falls back to an elementwise reduction)."""
    if x.ndim == 1 or x.size // x.shape[-1] == 1 or fp32_gemm_exact():
        return x.astype(F32) @ wT.astype(F32)
    xs = x.astype(F32).reshape(-1, x.shape[-1])
    w = wT.astype(F32).T
    out = [mx.sum(xs[i:i + 64, None, :] * w[None], axis=-1) for i in range(0, xs.shape[0], 64)]
    return mx.concatenate(out, axis=0).reshape(*x.shape[:-1], w.shape[0])


def rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """vLLM rms_norm: fp32 reduce, x * rsqrt(var + eps) * w rounded once."""
    xf = x.astype(F32)
    var = mx.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * mx.rsqrt(var + eps) * w.astype(F32)).astype(x.dtype)


# --------------------------------------------------------------------------- Mamba2


def _make_ssm_step_kernel():
    # One decode step of the Mamba2 recurrence (vLLM selective_state_update):
    # dt = softplus(dt_raw + dt_bias) [threshold 20], dA = exp(A * dt),
    # s = s * dA + dt * B * x, y = <s, C> + D * x; all fp32, y rounded to T.
    source = """
        uint n = thread_position_in_grid.z;          // batch * heads
        uint h = n % H;
        uint b = n / H;
        uint g = h / (H / G);
        constexpr int n_per_t = Ds / 32;
        uint ds_idx = thread_position_in_threadgroup.x;
        uint d_idx = thread_position_in_grid.y;

        float dtr = static_cast<float>(dt[n]) + static_cast<float>(dt_bias[h]);
        float dt_ = dtr <= 20.0f ? metal::precise::log(1.0f + metal::precise::exp(dtr)) : dtr;
        dt_ = metal::clamp(dt_, tmin[0], tmin[1]);
        float A = A_neg[h];
        float dA = metal::precise::exp(A * dt_);
        float x_ = static_cast<float>(X[n * Dh + d_idx]);
        auto B_ = B + (b * G + g) * Ds;
        auto C_ = C + (b * G + g) * Ds;
        auto i_state = state_in + (n * Dh + d_idx) * Ds;
        auto o_state = state_out + (n * Dh + d_idx) * Ds;
        float acc = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            int s = n_per_t * ds_idx + i;
            float st = i_state[s] * dA + static_cast<float>(B_[s]) * x_ * dt_;
            o_state[s] = st;
            acc += st * static_cast<float>(C_[s]);
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[n * Dh + d_idx] = static_cast<T>(acc + x_ * static_cast<float>(D[h]));
        }
    """
    return mx.fast.metal_kernel(
        name="nemotron_ssm_step",
        input_names=["X", "A_neg", "B", "C", "D", "dt", "dt_bias", "tmin", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


_ssm_step_kernel = None


def ssm_step(x, A_neg, B, C, D, dt, dt_bias, tlim, state):
    """x: [b, h, dh], B/C: [b, g, ds], dt: [b, h], state: [b, h, dh, ds] fp32."""
    global _ssm_step_kernel
    if _ssm_step_kernel is None:
        _ssm_step_kernel = _make_ssm_step_kernel()
    b, h, dh = x.shape
    g, ds = B.shape[-2:]
    return _ssm_step_kernel(
        inputs=[x, A_neg, B, C, D, dt, dt_bias, tlim, state],
        template=[("T", x.dtype), ("Dh", dh), ("Ds", ds), ("H", h), ("G", g)],
        grid=(32, dh, h * b),
        threadgroup=(32, 8, 1),
        output_shapes=[(b, h, dh), state.shape],
        output_dtypes=[x.dtype, F32],
    )


class Mamba2Mixer(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.nh, self.hd, self.ds, self.ng = a.mamba_num_heads, a.mamba_head_dim, a.ssm_state_size, a.n_groups
        self.inner = self.nh * self.hd
        self.conv_dim = self.inner + 2 * self.ng * self.ds
        self.in_proj = nn.Linear(a.hidden_size, self.inner + self.conv_dim + self.nh, bias=False)
        self.conv_weight = mx.zeros((self.conv_dim, a.conv_kernel))  # [C, K]
        self.conv_bias = mx.zeros((self.conv_dim,))
        self.dt_bias = mx.zeros((self.nh,))
        self.A_neg = mx.zeros((self.nh,))  # -exp(A_log), fp32 (vLLM keeps A in fp32)
        self.D = mx.zeros((self.nh,))
        self.norm_weight = mx.zeros((self.inner,))
        self.out_proj = nn.Linear(self.inner, a.hidden_size, bias=False)

    def _conv(self, xbc: mx.array, cache) -> mx.array:
        # Depthwise causal conv, fp32 accumulate + bias + SiLU, rounded once.
        K = self.a.conv_kernel
        B_, L, C = xbc.shape
        prev = cache.conv if (cache is not None and cache.conv is not None) else mx.zeros((B_, K - 1, C), xbc.dtype)
        full = mx.concatenate([prev, xbc], axis=1)
        if cache is not None:
            cache.conv = full[:, -(K - 1):, :]
        w = self.conv_weight.astype(F32)
        acc = self.conv_bias.astype(F32)
        for k in range(K):
            acc = acc + full[:, k:k + L, :].astype(F32) * w[:, k]
        return (acc * mx.sigmoid(acc)).astype(xbc.dtype)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B_, L, _ = x.shape
        dtype = x.dtype
        zxbcdt = self.in_proj(x)
        gate, xbc, dt = mx.split(zxbcdt, [self.inner, self.inner + self.conv_dim], axis=-1)
        xbc = self._conv(xbc, cache)
        xs, Bm, Cm = mx.split(xbc, [self.inner, self.inner + self.ng * self.ds], axis=-1)
        tlim = mx.array(self.a.time_step_limit, dtype=F32)
        state = cache.ssm if cache is not None else None
        if L == 1 and state is not None and mx.default_device() == mx.gpu:
            y, state = ssm_step(
                xs.reshape(B_, self.nh, self.hd),
                self.A_neg.astype(F32),
                Bm.reshape(B_, self.ng, self.ds),
                Cm.reshape(B_, self.ng, self.ds),
                self.D.astype(F32),
                dt.reshape(B_, self.nh),
                self.dt_bias.astype(F32),
                tlim,
                state,
            )
            y = y.reshape(B_, 1, self.inner)
        else:
            # Chunked SSD in fp32 (inputs upcast; vLLM's Triton kernels also run fp32).
            dtf = dt.astype(F32) + self.dt_bias.astype(F32)
            dtf = mx.where(dtf <= 20.0, mx.log1p(mx.exp(dtf)), dtf)
            dtf = mx.clip(dtf, *self.a.time_step_limit)
            # ssm_attn applies softplus(dt + dt_bias) itself; pass an inverse so it is exact:
            # instead we call it with dt_bias=0 and a pre-softplussed dt through a shim.
            y, state = _ssd_fp32(
                xs.astype(F32).reshape(B_, L, self.nh, self.hd),
                self.A_neg.astype(F32),
                Bm.astype(F32).reshape(B_, L, self.ng, self.ds),
                Cm.astype(F32).reshape(B_, L, self.ng, self.ds),
                self.D.astype(F32),
                dtf,
                state,
            )
            y = y.astype(dtype).reshape(B_, L, self.inner)
        if cache is not None:
            cache.ssm = state
        # Gated RMSNorm (vLLM Mixer2RMSNormGated.forward_native, 8 groups).
        yf = y.astype(F32) * nn.silu(gate.astype(F32))
        yg = yf.reshape(B_, L, self.ng, -1)
        yg = yg * mx.rsqrt(mx.mean(yg * yg, axis=-1, keepdims=True) + self.a.eps)
        yn = yg.reshape(B_, L, self.inner).astype(dtype)
        yn = (self.norm_weight * yn).astype(dtype) if dtype != F32 else self.norm_weight * yn
        return self.out_proj(yn)


def _ssd_fp32(x, A_neg, B, C, D, dt, state, step: int = 128):
    """Mamba2 SSD over a sequence, fp32. x: [b,l,h,dh], B/C: [b,l,g,n], dt: [b,l,h] (softplussed)."""
    b, l, h, dh = x.shape
    g, n = B.shape[-2:]
    rep = h // g
    dtA = dt * A_neg.reshape(1, 1, h)                    # [b,l,h]
    dtx = dt[..., None] * x                             # [b,l,h,dh]
    ys = []
    for i in range(0, l, step):
        xs, As, Bs, Cs = dtx[:, i:i + step], dtA[:, i:i + step], B[:, i:i + step], C[:, i:i + step]
        s = xs.shape[1]
        Acs = mx.cumsum(As, axis=1)                     # [b,s,h]
        # decay[t, u] = exp(Acs[t] - Acs[u]) for u <= t
        seg = Acs[:, :, None, :] - Acs[:, None, :, :]   # [b,t,u,h]
        tri = mx.tril(mx.ones((s, s), dtype=mx.bool_))
        decay = mx.where(tri[None, :, :, None], mx.exp(mx.where(tri[None, :, :, None], seg, 0.0)), 0.0)
        CB = mx.einsum("btgn,bugn->bgtu", Cs, Bs)       # [b,g,t,u]
        CB = mx.repeat(CB, rep, axis=1)                 # [b,h,t,u]
        M = CB * decay.transpose(0, 3, 1, 2)            # [b,h,t,u]
        y = mx.einsum("bhtu,buhd->bthd", M, xs)
        if state is not None:
            Cr = mx.repeat(Cs, rep, axis=2)             # [b,s,h,n]
            y_prev = mx.einsum("bshn,bhdn->bshd", Cr, state)
            y = y + mx.exp(Acs)[..., None] * y_prev
        # next state
        last = Acs[:, -1:, :]                           # [b,1,h]
        w = mx.exp(last - Acs)                          # [b,s,h]
        Br = mx.repeat(Bs, rep, axis=2)                 # [b,s,h,n]
        new = mx.einsum("bsh,bshd,bshn->bhdn", w, xs, Br)
        if state is not None:
            new = new + mx.exp(last[:, 0, :])[..., None, None] * state
        state = new
        ys.append(y)
    y = mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, h, 1)
    return y, state


# --------------------------------------------------------------------------- attention


class Attention(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.q_proj = nn.Linear(a.hidden_size, a.num_attention_heads * a.head_dim, bias=False)
        self.k_proj = nn.Linear(a.hidden_size, a.num_key_value_heads * a.head_dim, bias=False)
        self.v_proj = nn.Linear(a.hidden_size, a.num_key_value_heads * a.head_dim, bias=False)
        self.o_proj = nn.Linear(a.num_attention_heads * a.head_dim, a.hidden_size, bias=False)
        self.scale = a.head_dim ** -0.5

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B_, L, _ = x.shape
        a = self.a
        q = self.q_proj(x).reshape(B_, L, a.num_attention_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B_, L, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B_, L, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update(k, v)
        mask = "causal" if L > 1 else None
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B_, L, -1))


# --------------------------------------------------------------------------- MoE


class MLP(nn.Module):
    def __init__(self, d: int, f: int):
        super().__init__()
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def __call__(self, x):
        h = self.up_proj(x)
        hf = mx.maximum(h.astype(F32), 0.0)
        return self.down_proj((hf * hf).astype(x.dtype))


class MoE(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.gate_weight = mx.zeros((a.n_routed_experts, a.hidden_size))
        self.e_score_correction_bias = mx.zeros((a.n_routed_experts,))
        self.fc1 = SwitchLinear(a.hidden_size, a.moe_intermediate_size, a.n_routed_experts, bias=False)
        self.fc2 = SwitchLinear(a.moe_intermediate_size, a.hidden_size, a.n_routed_experts, bias=False)
        self.shared_experts = MLP(a.hidden_size, a.moe_shared_expert_intermediate_size)

    def route(self, x: mx.array):
        a = self.a
        logits = matmul_fp32(x, self.gate_weight.T)                # bf16 x bf16 -> fp32 (cuBLAS out_dtype=fp32)
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(F32)
        k = a.num_experts_per_tok
        inds = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
        # canonical order: descending choice score (vLLM's order is not observable)
        csel = mx.take_along_axis(choice, inds, axis=-1)
        order = mx.argsort(-csel, axis=-1)
        inds = mx.take_along_axis(inds, order, axis=-1)
        w = mx.take_along_axis(scores, inds, axis=-1)
        if a.norm_topk_prob:
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, w

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        inds, w = self.route(x)
        xe = mx.expand_dims(x, (-2, -3))
        sort = x.shape[0] * x.shape[1] >= 64
        if sort:
            from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
            xs, idx, inv = _gather_sort(xe, inds)
            h = self.fc1(xs, idx, sorted_indices=True)
        else:
            h = self.fc1(xe, inds)
        hf = mx.maximum(h.astype(F32), 0.0)
        h = (hf * hf).astype(dtype)
        y = self.fc2(h, idx if sort else inds, sorted_indices=sort)
        if sort:
            y = _scatter_unsort(y, inv, inds.shape)
        y = y.squeeze(-2)                                            # [B,L,k,D]
        routed = (y.astype(F32) * w[..., None]).sum(axis=-2) * self.a.routed_scaling_factor
        routed = routed.astype(dtype)
        return (routed + self.shared_experts(x)).astype(dtype)


# --------------------------------------------------------------------------- model


class Block(nn.Module):
    def __init__(self, a: Args, kind: str):
        super().__init__()
        self.kind = kind
        self.norm_weight = mx.zeros((a.hidden_size,))
        self.eps = a.eps
        if kind == "M":
            self.mixer = Mamba2Mixer(a)
        elif kind == "*":
            self.mixer = Attention(a)
        elif kind == "E":
            self.mixer = MoE(a)
        else:
            raise ValueError(kind)

    def __call__(self, x, cache=None):
        h = rms_norm(x, self.norm_weight, self.eps)
        if self.kind == "E":
            h = self.mixer(h)
        else:
            h = self.mixer(h, cache)
        return (x + h).astype(x.dtype)


class MambaCache:
    def __init__(self):
        self.conv = None
        self.ssm = None

    @property
    def state(self):
        return [self.conv, self.ssm]


class KVCache:
    """Preallocated KV buffer (fixed max length) so decode shapes stay static."""

    def __init__(self, max_len: Optional[int] = None):
        self.k = None
        self.v = None
        self.offset = 0
        self.max_len = max_len

    def update(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)
        self.offset += k.shape[2]
        return self.k, self.v

    @property
    def state(self):
        return [self.k, self.v]


class NemotronImageLM(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.layers = [Block(a, c) for c in a.pattern]
        self.norm_f_weight = mx.zeros((a.hidden_size,))
        self.lm_head = nn.Linear(a.hidden_size, a.vocab_size, bias=False)
        self.embed = None  # set by loader: callable ids -> [.., D]

    def make_cache(self):
        return [MambaCache() if l.kind == "M" else KVCache() if l.kind == "*" else None for l in self.layers]

    def hidden(self, h: mx.array, cache=None) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, c)
        return rms_norm(h, self.norm_f_weight, self.a.eps)

    def __call__(self, ids: mx.array, cache=None) -> mx.array:
        return self.lm_head(self.hidden(self.embed(ids), cache))
