"""L24SQ ViT image tokenizer (Nemotron-Labs-Image-Tokenizer) in MLX.

Mirrors upstream ``modeling_l24sq_vit.py``. The reference decodes in bfloat16
(``AutoModel.from_pretrained(..., dtype=torch.bfloat16)``), so the default here
is bf16 with the same rounding points: RMSNorm reduced in fp32 and rounded
before the weight multiply, RoPE tables built in fp32 and cast to bf16,
L2-normalised codebook rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

F32 = mx.float32


def _rms(x, w, eps=1e-6):
    xf = x.astype(F32)
    n = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)).astype(x.dtype)
    return n * w.astype(x.dtype)


def rope_2d(gh: int, gw: int, head_dim: int, theta: float, dtype) -> Tuple[mx.array, mx.array]:
    dq = head_dim // 4
    dh = head_dim // 2
    inv = (1.0 / (np.float32(theta) ** (np.arange(0, dq, dtype=np.float32) * np.float32(2) / np.float32(dh)))).astype(np.float32)
    hp = np.arange(gh, dtype=np.float32)
    wp = np.arange(gw, dtype=np.float32)
    ah = np.repeat(np.outer(hp, inv)[:, None, :], gw, axis=1).reshape(-1, dq)
    aw = np.repeat(np.outer(wp, inv)[None, :, :], gh, axis=0).reshape(-1, dq)
    eh = np.concatenate([ah, ah], -1)
    ew = np.concatenate([aw, aw], -1)
    cos = np.concatenate([np.cos(eh.astype(np.float64)), np.cos(ew.astype(np.float64))], -1).astype(np.float32)
    sin = np.concatenate([np.sin(eh.astype(np.float64)), np.sin(ew.astype(np.float64))], -1).astype(np.float32)
    return mx.array(cos).astype(dtype), mx.array(sin).astype(dtype)


def _rot_half(x):
    a, b = mx.split(x, 2, axis=-1)
    return mx.concatenate([-b, a], axis=-1)


def _rot_half_2d(x):
    a, b = mx.split(x, 2, axis=-1)
    return mx.concatenate([_rot_half(a), _rot_half(b)], axis=-1)


class SwiGLU(nn.Module):
    def __init__(self, d, ratio=4.0):
        super().__init__()
        hid = (int(d * ratio * 2 / 3) + 63) // 64 * 64
        self.gate_proj = nn.Linear(d, hid, bias=False)
        self.up_proj = nn.Linear(d, hid, bias=False)
        self.down_proj = nn.Linear(hid, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, cos, sin):
        B, N, D = x.shape
        q = self.q_proj(x).reshape(B, N, self.h, self.hd)
        k = self.k_proj(x).reshape(B, N, self.h, self.hd)
        v = self.v_proj(x).reshape(B, N, self.h, self.hd)
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        q = q * c + _rot_half_2d(q) * s
        k = k * c + _rot_half_2d(k) * s
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, N, D))


class ViTBlock(nn.Module):
    def __init__(self, d, heads, ratio):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn = Attn(d, heads)
        self.norm2 = nn.RMSNorm(d)
        self.mlp = SwiGLU(d, ratio)

    def __call__(self, x, cos, sin):
        x = x + self.attn(_rms(x, self.norm1.weight), cos, sin)
        return x + self.mlp(_rms(x, self.norm2.weight))


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.patch = c["patch_size"]
        self.patch_embed_w = mx.zeros((c["width"], 3 * self.patch * self.patch))
        self.patch_embed_b = mx.zeros((c["width"],))
        self.blocks = [ViTBlock(c["width"], c["heads"], c["mlp_ratio"]) for _ in range(c["layers"])]
        self.norm = nn.RMSNorm(c["width"])


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blocks = [ViTBlock(c["width"], c["heads"], c["mlp_ratio"]) for _ in range(c["layers"])]
        self.norm = nn.RMSNorm(c["width"])
        self.ffn = nn.Linear(c["width"], c["dim_ffn_output"])
        self.conv_out = nn.Linear(c["dim_ffn_output"], 3 * c["patch_size"] ** 2)


class L24SQ(nn.Module):
    def __init__(self, c: dict, with_encoder: bool = False):
        super().__init__()
        self.c = c
        self.p = c["patch_size"]
        self.hd = c["width"] // c["heads"]
        self.decoder = Decoder(c)
        self.codebook = mx.zeros((c["n_embed"], c["embed_dim"]))
        self.post_quant_embed = nn.Linear(c["embed_dim"], c["width"])
        if with_encoder:
            self.encoder = Encoder(c)
            self.quant_embed = nn.Linear(c["width"], c["embed_dim"])

    # ----------------------------------------------------------------- decode

    def lookup(self, idx: mx.array) -> mx.array:
        z = self.codebook[idx]
        n = mx.sqrt(mx.sum(z.astype(F32) ** 2, axis=-1, keepdims=True)).astype(z.dtype)
        return z / mx.maximum(n, mx.array(1e-12, z.dtype))

    def decode(self, indices: mx.array, grid: Tuple[int, int]) -> mx.array:
        """indices: [N] int -> image [3, H, W] in [-1, 1] (model dtype)."""
        gh, gw = grid
        dt = self.codebook.dtype
        h = self.post_quant_embed(self.lookup(indices))[None]
        cos, sin = rope_2d(gh, gw, self.hd, self.c["vit_rope_theta"], dt)
        for blk in self.decoder.blocks:
            h = blk(h, cos, sin)
        h = _rms(h, self.decoder.norm.weight)
        h = mx.tanh(self.decoder.ffn(h))
        h = self.decoder.conv_out(h)[0]                                     # [N, 3*p*p]
        p = self.p
        img = h.reshape(gh, gw, 3, p, p).transpose(2, 0, 3, 1, 4).reshape(3, gh * p, gw * p)
        return mx.clip(img, -1.0, 1.0)

    @staticmethod
    def to_uint8(img: mx.array) -> np.ndarray:
        x = (mx.clip(img.astype(F32), -1, 1) + 1) / 2 * 255
        return np.array(mx.round(x).astype(mx.uint8).transpose(1, 2, 0))

    # ----------------------------------------------------------------- encode

    def encode(self, img: mx.array, chunk: int = 2048) -> Tuple[mx.array, Tuple[int, int]]:
        """img: [3, H, W] in [-1, 1], H, W multiples of 16 -> indices [N]."""
        _, H, W = img.shape
        p = self.p
        gh, gw = H // p, W // p
        dt = self.codebook.dtype
        x = img.astype(dt).reshape(3, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, 3 * p * p)
        h = (x @ self.encoder.patch_embed_w.T + self.encoder.patch_embed_b)[None]
        cos, sin = rope_2d(gh, gw, self.hd, self.c["vit_rope_theta"], dt)
        for blk in self.encoder.blocks:
            h = blk(h, cos, sin)
        h = _rms(h, self.encoder.norm.weight)
        z = self.quant_embed(h)[0].astype(F32)
        z = z / mx.maximum(mx.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
        cb = self.codebook.astype(F32)
        cb = cb / mx.maximum(mx.linalg.norm(cb, axis=-1, keepdims=True), 1e-12)
        out = []
        for i in range(0, z.shape[0], chunk):
            out.append(mx.argmax(z[i:i + chunk] @ cb.T, axis=-1))
            mx.eval(out[-1])
        return mx.concatenate(out), (gh, gw)


def load_vq(path, dtype=mx.bfloat16, with_encoder: bool = False) -> L24SQ:
    path = Path(path)
    c = json.loads((path / "config.json").read_text())
    m = L24SQ(c, with_encoder)
    raw = mx.load(str(path / "model.safetensors"))
    w = {}
    for k, v in raw.items():
        if k.startswith("decoder.distill_mlp"):
            continue
        if k.startswith("encoder.") and not with_encoder:
            continue
        if k.startswith("quant_embed") and not with_encoder:
            continue
        nk = k
        nk = nk.replace("decoder.ffn.0.", "decoder.ffn.")
        nk = nk.replace("quantize.embedding.weight", "codebook")
        if k == "encoder.patch_embed.weight":
            nk, v = "encoder.patch_embed_w", v.reshape(v.shape[0], -1)
        elif k == "encoder.patch_embed.bias":
            nk = "encoder.patch_embed_b"
        w[nk] = v.astype(dtype)
    m.load_weights(list(w.items()), strict=True)
    mx.eval(m.parameters())
    return m
