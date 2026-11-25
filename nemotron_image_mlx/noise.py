"""Sampling noise that reproduces vLLM's seeded sampler.

vLLM 0.21 (`vllm/v1/sample/ops/topk_topp_sampler.py`, `random_sample`) samples a
token as ``argmax(softmax(logits / T) / q)`` where, for a request with a seed,
``q[i].exponential_(generator=torch.Generator("cuda").manual_seed(seed))``.

On CUDA, torch 2.11 fills that row with ``distribution_nullary_kernel``
(`aten/src/ATen/native/cuda/DistributionTemplates.h`):

* launch: 256 threads per block, ``min(SMs * maxThreadsPerSM / 256, ceil(n / 256))``
  blocks (H100 SXM: 132 SMs x 2048 threads -> 1056 blocks, 270,336 threads);
* each thread calls ``curand_init(seed, subsequence=thread_idx, offset, &state)``
  and draws one ``curand_uniform4``; element ``thread_idx + k * n_threads`` takes
  lane ``k``;
* the generator offset advances by 4 per call (one call per decode step);
* ``exponential``: ``u >= 1 - eps/2 ? eps/2 : -log(u)`` (lambda = 1).

With offset ``4 * step`` and subsequence ``j`` the curand state reduces to one
Philox4x32-10 block with counter ``(step, 0, j, 0)`` and key ``(seed_lo, seed_hi)``.

The Philox core is checked against the Random123 known-answer vectors. The
torch/curand launch layout above is transcribed from source; it has not been
run against a real CUDA device here.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
TWO_POW32_INV = np.float32(2.3283064e-10)
FLT_EPS = np.float32(np.finfo(np.float32).eps)

# Known CUDA launch geometry: (SM count, max threads per SM).
GPU_GEOMETRY = {
    "h100-sxm": (132, 2048),
    "h100-pcie": (114, 2048),
    "a100": (108, 2048),
    "b200": (148, 2048),
}


def n_threads(numel: int, gpu: str = "h100-sxm") -> int:
    sms, max_threads = GPU_GEOMETRY[gpu]
    blocks = min(sms * (max_threads // 256), (numel + 255) // 256)
    return blocks * 256


# --------------------------------------------------------------------------- numpy reference


def philox4x32_10(ctr: np.ndarray, key: np.ndarray) -> np.ndarray:
    """ctr: [..., 4] uint32, key: [..., 2] uint32 -> [..., 4] uint32."""
    c = ctr.astype(np.uint64)
    k0 = key[..., 0].astype(np.uint64)
    k1 = key[..., 1].astype(np.uint64)
    mask = np.uint64(0xFFFFFFFF)
    for r in range(10):
        p0 = np.uint64(PHILOX_M0) * c[..., 0]
        p1 = np.uint64(PHILOX_M1) * c[..., 2]
        hi0, lo0 = p0 >> np.uint64(32), p0 & mask
        hi1, lo1 = p1 >> np.uint64(32), p1 & mask
        c = np.stack([hi1 ^ c[..., 1] ^ k0, lo1, hi0 ^ c[..., 3] ^ k1, lo0], axis=-1)
        if r < 9:
            k0 = (k0 + np.uint64(PHILOX_W0)) & mask
            k1 = (k1 + np.uint64(PHILOX_W1)) & mask
    return c.astype(np.uint32)


def _uniform(x: np.ndarray) -> np.ndarray:
    # curand: x * 2^-32 + 2^-33 in float32 (x converted to float first).
    return x.astype(np.float32) * TWO_POW32_INV + np.float32(TWO_POW32_INV / np.float32(2.0))


def exponential_row_numpy(seed: int, step: int, numel: int, gpu: str = "h100-sxm") -> np.ndarray:
    T = n_threads(numel, gpu)
    j = np.arange(numel, dtype=np.int64)
    thread = j % T
    lane = j // T
    assert lane.max() < 4
    ctr = np.zeros((numel, 4), np.uint32)
    ctr[:, 0] = np.uint32(step & 0xFFFFFFFF)
    ctr[:, 1] = np.uint32(step >> 32)
    ctr[:, 2] = thread.astype(np.uint32)
    key = np.broadcast_to(np.array([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF], np.uint32), (numel, 2))
    out = philox4x32_10(ctr, key)
    u = _uniform(out[j, lane]) if False else _uniform(np.take_along_axis(out, lane[:, None], 1)[:, 0])
    # log in float64 then rounded: correctly rounded logf (CUDA logf is within 1 ulp).
    log = np.log(u.astype(np.float64)).astype(np.float32)
    log = np.where(u >= np.float32(1.0) - FLT_EPS / np.float32(2.0), -FLT_EPS / np.float32(2.0), log)
    return (np.float32(-1.0) * log).astype(np.float32)


# --------------------------------------------------------------------------- Metal kernel

_SOURCE = """
    uint j = thread_position_in_grid.x;
    if (j >= NUMEL) return;
    uint T = NTHREADS;
    uint thread_id = j % T;
    uint lane = j / T;
    uint c0 = step_seed[0];
    uint c1 = step_seed[1];
    uint c2 = thread_id;
    uint c3 = 0;
    uint k0 = step_seed[2];
    uint k1 = step_seed[3];
    for (int r = 0; r < 10; ++r) {
        uint lo0 = 0xD2511F53u * c0;
        uint hi0 = mulhi(0xD2511F53u, c0);
        uint lo1 = 0xCD9E8D57u * c2;
        uint hi1 = mulhi(0xCD9E8D57u, c2);
        uint n0 = hi1 ^ c1 ^ k0;
        uint n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    uint x = lane == 0 ? c0 : (lane == 1 ? c1 : (lane == 2 ? c2 : c3));
    float u = float(x) * 2.3283064e-10f + 1.1641532e-10f;
    float l = u >= (1.0f - 5.9604645e-08f) ? -5.9604645e-08f : metal::precise::log(u);
    out[j] = -l;
"""


@functools.lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="vllm_philox_exponential",
        input_names=["step_seed"],
        output_names=["out"],
        source=_SOURCE,
    )


def exponential_row(seed: int, step: int, numel: int, gpu: str = "h100-sxm") -> mx.array:
    """q for one decode step, as vLLM-on-CUDA would draw it. Shape [numel], float32."""
    T = n_threads(numel, gpu)
    assert numel <= 4 * T
    params = mx.array(
        [step & 0xFFFFFFFF, (step >> 32) & 0xFFFFFFFF, seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF],
        dtype=mx.uint32,
    )
    (out,) = _kernel()(
        inputs=[params],
        template=[("NUMEL", numel), ("NTHREADS", T)],
        grid=(numel, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(numel,)],
        output_dtypes=[mx.float32],
    )
    return out


def exponential_rows(seed: int, steps: mx.array, numel: int, gpu: str = "h100-sxm") -> mx.array:
    """Batched helper (one row per step) used by tests."""
    return mx.stack([exponential_row(seed, int(s), numel, gpu) for s in np.array(steps)])
