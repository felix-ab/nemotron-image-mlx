"""Compiled fixed-shape decode step.

Eager decoding builds ~1,500 small ops per token (52 layers). This module keeps
all decode state in fixed-shape arrays (preallocated KV buffers with a position
index, Mamba conv/SSM states) so one ``mx.compile``-d function does a whole
step: embedding row in, sampled token and new state out. Fusing the
elementwise chains (norms, gating, routing, residual adds) cuts kernel launches;
pipelining with a one-step-late readback keeps the GPU busy.

Numerics are identical to ``model.py`` (same functions are called).
"""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .model import Block, NemotronImageLM, rms_norm, ssm_step
from .noise import _kernel as _noise_kernel, n_threads

F32, BF16 = mx.float32, mx.bfloat16


def _mamba_step(mx_, x, conv_state, ssm_state):
    """x: [B,1,D] (normed). Returns y [B,1,D], new conv, new ssm."""
    a = mx_.a
    B_ = x.shape[0]
    zxbcdt = mx_.in_proj(x)
    gate, xbc, dt = mx.split(zxbcdt, [mx_.inner, mx_.inner + mx_.conv_dim], axis=-1)
    full = mx.concatenate([conv_state, xbc], axis=1)                # [B,K,C]
    new_conv = full[:, 1:, :]
    acc = mx_.conv_bias.astype(F32) + mx.sum(full.astype(F32) * mx_.conv_weight.astype(F32).T[None], axis=1, keepdims=True)
    xbc = (acc * mx.sigmoid(acc)).astype(x.dtype)
    xs, Bm, Cm = mx.split(xbc, [mx_.inner, mx_.inner + mx_.ng * mx_.ds], axis=-1)
    tlim = mx.array(a.time_step_limit, dtype=F32)
    y, new_ssm = ssm_step(
        xs.reshape(B_, mx_.nh, mx_.hd), mx_.A_neg.astype(F32),
        Bm.reshape(B_, mx_.ng, mx_.ds), Cm.reshape(B_, mx_.ng, mx_.ds),
        mx_.D.astype(F32), dt.reshape(B_, mx_.nh), mx_.dt_bias.astype(F32), tlim, ssm_state,
    )
    y = y.reshape(B_, 1, mx_.inner)
    yf = y.astype(F32) * nn.silu(gate.astype(F32))
    yg = yf.reshape(B_, 1, mx_.ng, -1)
    yg = yg * mx.rsqrt(mx.mean(yg * yg, axis=-1, keepdims=True) + a.eps)
    yn = (mx_.norm_weight * yg.reshape(B_, 1, mx_.inner).astype(x.dtype)).astype(x.dtype)
    return mx_.out_proj(yn), new_conv, new_ssm


def _attn_step(at, x, K, V, pos, valid):
    a = at.a
    B_ = x.shape[0]
    q = at.q_proj(x).reshape(B_, 1, a.num_attention_heads, -1).transpose(0, 2, 1, 3)
    k = at.k_proj(x).reshape(B_, 1, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    v = at.v_proj(x).reshape(B_, 1, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    K = mx.slice_update(K, k, pos, axes=(2,))
    V = mx.slice_update(V, v, pos, axes=(2,))
    o = mx.fast.scaled_dot_product_attention(q, K, V, scale=at.scale, mask=valid)
    return at.o_proj(o.transpose(0, 2, 1, 3).reshape(B_, 1, -1)), K, V


class FastDecoder:
    def __init__(self, model: NemotronImageLM, batch: int, max_len: int, temperature: float,
                 seeds: List[int], noise_gpu: str = "h100-sxm", compile: bool = True, screen: bool = False):
        self.m = model
        self.B, self.max_len, self.T = batch, max_len, temperature
        self.seeds = seeds
        self.V = model.a.vocab_size
        self.nthreads = n_threads(self.V, noise_gpu)
        self.kinds = [l.kind for l in model.layers]
        self.screen = ScreenedHead(model.lm_head, split=131072) if screen else None
        self._step = mx.compile(self._step_impl) if compile else self._step_impl
        self.n_exact = 0
        self.vis_range = None
        self._step_vis = self._step_none = None
        self._compile = compile

    def enable_constrained(self, vis_start: int, vis_end: int):
        """Prepare the visual-rows head and the head-less step used by constrained decoding."""
        self.vis_range = (vis_start, vis_end)
        lm = self.m.lm_head
        if hasattr(lm, "scales"):
            head = nn.QuantizedLinear(self.m.a.hidden_size, 64, bias=False, group_size=lm.group_size, bits=lm.bits, mode=lm.mode)
            head.weight = lm.weight[vis_start:vis_end]
            head.scales = lm.scales[vis_start:vis_end]
            if getattr(lm, "biases", None) is not None:
                head.biases = lm.biases[vis_start:vis_end]
        else:
            head = nn.Linear(1, 1, bias=False)
            head.weight = lm.weight[vis_start:vis_end]
        mx.eval(head.parameters())
        self.head_vis = head
        c = mx.compile if self._compile else (lambda f: f)
        self._step_vis = c(self._step_vis_impl)
        self._step_none = c(self._step_none_impl)

    # ------------------------------------------------------------- state from an eager prefill cache
    def state_from_cache(self, cache, pos: int):
        st = []
        for l, c in zip(self.m.layers, cache):
            if l.kind == "M":
                st += [c.conv, c.ssm]
            elif l.kind == "*":
                k, v = c.k, c.v
                pad = self.max_len - k.shape[2]
                z = mx.zeros((k.shape[0], k.shape[1], pad, k.shape[3]), k.dtype)
                st += [mx.concatenate([k, z], axis=2), mx.concatenate([v, z], axis=2)]
        return st

    def zero_state(self):
        a = self.m.a
        st = []
        conv_dim = a.mamba_num_heads * a.mamba_head_dim + 2 * a.n_groups * a.ssm_state_size
        for l in self.m.layers:
            if l.kind == "M":
                st += [mx.zeros((self.B, a.conv_kernel - 1, conv_dim), BF16),
                       mx.zeros((self.B, a.mamba_num_heads, a.mamba_head_dim, a.ssm_state_size), F32)]
            elif l.kind == "*":
                z = mx.zeros((self.B, a.num_key_value_heads, self.max_len, a.head_dim), BF16)
                st += [z, z]
        return st

    def _noise(self, step):
        params = mx.stack([mx.concatenate([step.astype(mx.uint32), mx.array([0], mx.uint32),
                                           mx.array([s & 0xFFFFFFFF], mx.uint32), mx.array([(s >> 32) & 0xFFFFFFFF], mx.uint32)])
                           for s in self.seeds])
        qs = []
        for b in range(self.B):
            (q,) = _noise_kernel()(inputs=[params[b]], template=[("NUMEL", self.V), ("NTHREADS", self.nthreads)],
                                   grid=(self.V, 1, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(self.V,)], output_dtypes=[F32])
            qs.append(q)
        return mx.stack(qs)

    def _body(self, x, pos, state):
        valid = (mx.arange(self.max_len) <= pos)[None, None, None, :]
        new_state = []
        si = 0
        h = x
        for l in self.m.layers:
            hn = rms_norm(h, l.norm_weight, l.eps)
            if l.kind == "M":
                y, c, s = _mamba_step(l.mixer, hn, state[si], state[si + 1])
                new_state += [c, s]; si += 2
            elif l.kind == "*":
                y, K, V = _attn_step(l.mixer, hn, state[si], state[si + 1], pos, valid)
                new_state += [K, V]; si += 2
            else:
                y = l.mixer(hn)
            h = (h + y).astype(h.dtype)
        return rms_norm(h, self.m.norm_f_weight, self.m.a.eps)[:, 0, :], new_state

    def _step_impl(self, x, pos, step, *state):
        """x: [B,1,D] embedding rows; pos: [1] int32 KV write index; step: [1] uint32 sample step."""
        h, new_state = self._body(x, pos, state)
        q = self._noise(step)
        if self.screen is not None:
            tok, need = self.screen.scores(h, self.T, q)
        else:
            tok, need = self._full_sample(h, q), mx.zeros((self.B,), mx.bool_)
        return [tok, need, h] + new_state

    def _step_vis_impl(self, x, pos, step, *state):
        """Sample restricted to visual tokens (exactly the full sample whenever that is a visual token)."""
        h, new_state = self._body(x, pos, state)
        lo, hi = self.vis_range
        q = self._noise(step)[:, lo:hi]
        lv = self.head_vis(h).astype(F32) / self.T
        p = mx.exp(lv - mx.max(lv, axis=-1, keepdims=True))
        return [mx.argmax(p / q, axis=-1) + lo, h] + new_state

    def _step_none_impl(self, x, pos, *state):
        """Advance the state only (the next token is fixed by the image grammar)."""
        h, new_state = self._body(x, pos, state)
        return [h] + new_state

    def _step_masked_impl(self, x, pos, step, forced, lo, hi, *state):
        """General per-row step: allowed range [lo, hi) per row, or a forced token (forced >= 0)."""
        h, new_state = self._body(x, pos, state)
        logits = self.m.lm_head(h).astype(F32) / self.T
        ids = mx.arange(self.V)[None]
        logits = mx.where((ids >= lo[:, None]) & (ids < hi[:, None]), logits, -mx.inf)
        p = mx.exp(logits - mx.max(logits, axis=-1, keepdims=True))
        tok = mx.argmax(p / self._noise(step), axis=-1)
        return [mx.where(forced >= 0, forced, tok), h] + new_state

    def step_masked(self, x, pos: int, step: int, forced, lo, hi, state):
        if getattr(self, "_step_masked", None) is None:
            self._step_masked = mx.compile(self._step_masked_impl) if self._compile else self._step_masked_impl
        out = self._step_masked(x, mx.array([pos], mx.int32), mx.array([step], mx.uint32),
                                mx.array(forced, mx.int32), mx.array(lo, mx.int32), mx.array(hi, mx.int32), *state)
        return out[0], out[1], out[2:]

    def step_vis(self, x, pos: int, step: int, state):
        out = self._step_vis(x, mx.array([pos], mx.int32), mx.array([step], mx.uint32), *state)
        return out[0], out[1], out[2:]

    def step_none(self, x, pos: int, state):
        out = self._step_none(x, mx.array([pos], mx.int32), *state)
        return out[0], out[1:]

    def _full_sample(self, h, q):
        logits = self.m.lm_head(h).astype(F32) / self.T
        probs = mx.softmax(logits, axis=-1)
        return mx.argmax(probs / q, axis=-1)

    def exact(self, h, step: int):
        """Full-vocabulary vLLM sample for this step (used when the screen cannot decide)."""
        self.n_exact += 1
        return self._full_sample(h, self._noise(mx.array([step], mx.uint32)))

    def step(self, x, pos: int, step: int, state):
        out = self._step(x, mx.array([pos], mx.int32), mx.array([step], mx.uint32), *state)
        return out[0], out[1], out[2], out[3:]


class ScreenedHead:
    """Exact sampling with most of the text-token head skipped.

    Rows >= ``split`` (visual + framing tokens) are evaluated exactly (bf16 logits, as
    the reference). Text rows are scored with a 4-bit copy and a rigorous upper bound
    ``|l_exact - l_approx| <= ||w - w_q||_2 * ||h||_2`` (Cauchy-Schwarz) plus bf16
    rounding slack. If no text token's upper-bound score can beat the best exact
    score, the sample is provably the full-vocabulary sample; otherwise ``exact()``
    recomputes the full head for that step.
    """

    def __init__(self, lm_head, split: int, bits: int = 4, group: int = 64):
        W = lm_head.weight
        assert not hasattr(lm_head, "scales"), "screening needs a bf16 lm_head"
        self.split = split
        self.W_vis = W[split:]
        self.W_txt = W[:split]
        q, s, b = mx.quantize(self.W_txt, group_size=group, bits=bits)
        self.tq = (q, s, b, group, bits)
        deq = mx.dequantize(q, s, b, group_size=group, bits=bits)
        self.err = mx.sqrt(mx.sum((self.W_txt.astype(F32) - deq.astype(F32)) ** 2, axis=-1))
        mx.eval(self.W_vis, self.tq, self.err)

    def scores(self, h, T, q_noise):
        """h: [B, D] (bf16, normed). Returns (token, needs_exact[B])."""
        lv = (h @ self.W_vis.T).astype(F32) / T                    # exact (bf16-rounded like reference)
        qv = q_noise[:, self.split:]
        sv = lv - mx.log(qv)
        best_v = mx.max(sv, axis=-1)
        tok_v = mx.argmax(sv, axis=-1) + self.split
        q_, s_, b_, g, bits = self.tq
        lt = mx.quantized_matmul(h, q_, s_, b_, transpose=True, group_size=g, bits=bits).astype(F32)
        hn = mx.sqrt(mx.sum(h.astype(F32) ** 2, axis=-1, keepdims=True))
        ub = lt + self.err[None] * hn
        ub = ub + mx.abs(ub) * (2.0 ** -8) + 1e-3                   # bf16 rounding + accumulation slack
        st = ub / T - mx.log(q_noise[:, :self.split])
        need = mx.max(st, axis=-1) >= best_v
        return tok_v, need
