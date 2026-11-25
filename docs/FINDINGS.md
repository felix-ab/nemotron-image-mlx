# Nemotron-Labs-Image-30B-A3B on MLX: findings

Working notes, 2026-11-25. Everything here was measured or read from source; open items are marked.

## 1. What the model is

- **Generator:** `NemotronHForCausalLM`, initialised from Nemotron 3 Nano. 52 layers:
  23 Mamba2 (`M`), 23 MoE (`E`), 6 attention (`*`), pattern
  `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`. Hidden 2688.
  - **Mamba2:** 64 heads x 64, state 128, 8 groups, conv kernel 4, gated RMSNorm over 8 groups.
  - **MoE:** 128 routed ReLU^2 experts (2688 -> 1856 -> 2688), top-6 by sigmoid score + an
    fp32 correction bias, renormalised, routed output x2.5, plus one shared ReLU^2 expert (3712).
  - **Attention:** GQA 32 query / 2 KV heads of 128, **no positional encoding** (position comes from
    the Mamba layers and the row markers).
  - **Vocabulary:** 327,680 rows = 131,072 text + 3 framing (`<|img_start|>`, `<|img_end|>`,
    `<|vis_row|>`) + 196,560 visual codes + 45 padding rows.
- **Image as text:** after the prompt the model writes `<|img_start|>`, a size header in text
  (`height=H width=W`), then one `<|vis_row|>` plus W/16 visual tokens per row, then
  `<|img_end|>`. A 1024x1024 image is 64 x 65 = 4,160 grid tokens.
- **Pixel decoder (L24SQ):** a 24-layer ViT (width 1024, 2D RoPE, SwiGLU) decodes L2-normalised
  codes from a fixed 196,560-entry Leech-lattice codebook (24-d) into 16x16 patches.
- **Checkpoint:** fp32 safetensors, 128 GB; the routed experts are 29.4 B of ~32.6 B parameters.
  Experts of every layer are spread over all 32 shards.

## 2. The reference is vLLM in bf16, not the HF modeling code

The model card's generation path is vLLM 0.21 (`dtype=bfloat16`, TP=4 on H100, temperature
0.9, top-p 1, top-k off, seed 42, full-vocabulary sampling, no CFG). The port reproduces
vLLM's numerics, read from source (`nemotron_h.py`, `mamba_mixer2.py`, `layernorm_kernels.cu`,
`gate_linear.py`, `topk_topp_sampler.py`):

| Op | vLLM bf16 on H100 |
|---|---|
| Residual | bf16, each add rounded once |
| RMSNorm | fp32 reduce, `x * rsqrt(var+eps) * w` rounded once, `w` the bf16 copy |
| Router | bf16 x bf16 GEMM with **fp32 output** (cuBLAS `out_dtype`), sigmoid + bias in fp32 |
| Mamba | conv + SiLU in fp32; `A = -exp(A_log)` kept in fp32; `D`, `dt_bias` bf16; softplus/SSM in fp32 with an **fp32 SSM state** (`mamba_ssm_cache_dtype`); gated norm in fp32, rounded, then x bf16 weight |
| Logits | bf16, upcast to fp32 for sampling |
| Sampler | `argmax(softmax(logits/T) / q)`, `q ~ Exp(1)` from a per-request CUDA generator |

**Sampling noise.** With a seed, `q` comes from torch's CUDA `exponential_`. Reading
`DistributionTemplates.h` (torch 2.11) gives a closed form: for decode step *i* and vocabulary
index *j*, one Philox4x32-10 block with counter `(i, 0, j mod 270336, 0)` and key `(seed, 0)`;
lane `j div 270336`; `u = x * 2^-32 + 2^-33`; `q = -log u` (H100 SXM: 132 SMs -> 1,056 blocks of
256 threads). Implemented as a Metal kernel (0.26 ms per row). The Philox core matches the
Random123 known-answer vectors; the launch layout is transcribed from source and has **not** been
run on a CUDA device here.

## 3. Bugs and dead paths found in existing code

- **mlx-lm `nemotron_h`** computes router logits as `x @ W.T` in bf16, i.e. it rounds the router
  logits to bf16 before the sigmoid/top-k. With 128 experts and a correction bias this flips
  near-tie expert choices. It also double-rounds RMSNorm and the gated norm. The port computes
  the router as vLLM does.
- **Upstream HF modeling code (CPU paths)**: `HybridMambaAttentionDynamicCache` is broken (missing
  `conv_kernel_size`, `.device` on a list, wrong SSM-state shape), the SDPA attention class
  reshapes with `hidden_size` instead of heads x head_dim, every block forward enters a CUDA
  stream context, and `mamba_ssm`'s Triton `rmsnorm_fn` is a hard import. The torch reference
  here uses the eager attention class, a pure-torch copy of `mamba_ssm`'s `rms_norm_ref`, and a
  no-op stream context; the math is untouched.

## 4. Apple Silicon / M5 findings

- **TF32 by default on M5.** MLX 0.32.5 runs float32 GEMMs with M > 1 in TF32 on M5
  (`applegpu_g17s`): relative error 7.7e-4 vs 9e-7 with `MLX_ENABLE_TF32=0`. GEMV (M = 1) and
  the CPU are unaffected. MLX reads the variable at the first fp32 GEMM of the process, so the
  package sets it on import. bf16-valued inputs are exact in TF32, so the router is safe either
  way; it matters for the fp32 truth path and the fp32 pixel decoder.
- **bf16 GEMMs on M5** match the correctly rounded result in 99.97-100% of elements (fp32
  accumulation), like CUDA.
- **Bandwidth:** ~375-400 GB/s achievable on this M5 Max (32-core GPU); 4/5/6/8-bit quantized
  GEMV run at 350-400 GB/s.
- **Cold start is kernel JIT:** first prefill 14.3 s vs 0.38 s warm; first compiled step 4.7 s,
  then 18 ms. macOS caches compiled shaders across processes. Recurrent (step) prefill was meant
  to avoid prefill-only kernels but did not help on real weights (M4: 9.5-13.5 s vs 12.6 s), so
  parallel prefill stays the default.

## 5. Where the time and memory go (per decoded token)

| Part | Parameters read / token | bf16 bytes | Share of resident weights |
|---|---|---|---|
| 6 of 128 routed experts x 23 layers | 1.38 B | 2.75 GB | 29.4 B params (~90%) |
| Shared experts | 0.46 B | 0.92 GB | small |
| Mamba projections | 0.89 B | 1.78 GB | small |
| Attention projections | 0.14 B | 0.28 GB | small |
| lm_head (327,680 rows) | 0.88 B | 1.76 GB | small |

Memory is set by the routed experts; per-token bandwidth by everything else. So routed-expert
bits buy memory, while dense/shared/head bits buy speed. The embedding table (1.76 GB bf16) is
touched one row per step and can live in a memory map.

## 6. Validation so far

| Check | Result |
|---|---|
| Prompt token ids vs upstream tokenizer (4 prompts incl. Unicode) | identical |
| Philox4x32-10 vs Random123 KAT | identical |
| Mamba / attention / MoE block vs upstream torch, fp32 (random weights, real dims) | rel 1.1e-6 / 3.4e-7 / 4.6e-7 |
| Same, bf16, distance to fp32 truth: MLX vs torch-CPU | 6.16e-3 vs 6.25e-3; 2.68e-3 vs 2.69e-3; 2.57e-2 vs 2.57e-2 |
| MoE routing (top-6 sets) vs torch | 100% of tokens, fp32 and bf16 |
| Decode-step kernels vs full-sequence forward (fp32) | rel 1.2e-6 (Mamba), 2e-7 (attention) |
| L24SQ decode fp32 vs torch fp32 | rel 4.2e-6 (117 dB), 99.975% identical uint8 pixels |
| L24SQ decode bf16 vs torch fp32 | 40.36 dB (torch bf16 itself: 40.22 dB) |
| L24SQ encode vs torch (fp32) | 100% identical codes |
| Eager vs compiled vs screened-head decoding (random model, 3 seeds x 96 tokens) | identical token streams |
| Parallel vs recurrent prompt prefill (fp32) | rel 5.9e-7 |
| Official gallery PNG -> encode -> decode | 33.3 dB: gallery PNGs are not exact decoder outputs |

## 7. The model cannot write its own header for 880x1184 (upstream behaviour, not the port)

At the recommended size 880x1184 the original model writes the wrong size header, so the
official `infer.py` raises "Generated size ... does not match". Teacher-forced probabilities
(`parity/eval_header.json`), full 52-layer streamed forward from the Hub weights:

| Case | fp32 truth (= torch fp32 to <=1e-6 through layer 50) | bf16 | q8 |
|---|---|---|---|
| short prompt, 880x1184: `width=88` -> next digit | `4` with p = 1.000 (wants `0`) | same | same |
| gallery #023 (502-token prompt), height third digit | `2` p = 0.81 (`8` p = 0.19) | 0.90 | 0.95 |
| 1024x1024, 1248x832, 1184x880, 832x1248 | every header digit p = 1.000 correct | same | same |

Seen on two of two 880x1184 renders (headers `height=1284 width=884`, `height=1124 width=804`;
the grids that followed were 75x55 and 76x54 instead of 74x55). The pictures themselves were
fine. This looks like an induction-style copy error: after "88" the model copies what followed
the last "8" in the prompt ("1184" -> "4").

**Constrained decoding (default).** After `<|img_start|>` the header and the row/end markers
are fixed and in-row positions sample among visual codes only. Masking rescales the allowed
probabilities by a common factor, so `argmax(p/q)` over the allowed set equals the unconstrained
sample whenever that sample is allowed: every run the reference would accept is reproduced
token for token. Verified on real weights: gallery #003 and #004, seed 42, constrained vs
unconstrained, **identical for all 4,175 tokens**. Runs the reference would reject become valid
images. Forced positions skip the output head; visual positions read only the 196,560 visual rows
(M4 Pro, q8, bf16 head: 33.3 -> 37.3 tok/s).

## 8. Real-weight exactness

Streamed fp32 forward of the real checkpoint, MLX vs the upstream torch module on the CPU, same
input embeddings (`parity/eval_header.json`, `torch_check`): relative hidden-state difference
3e-7 to 1e-6 at every layer 0-50; 2.0e-4 after layer 51 (the last MoE layer; consistent with a
near-tie expert flip from fp32 summation order in one token, not a systematic error).

## 9. Speed and memory (decode, 1024x1024 = 4,174 generated tokens)

**M4 Pro 48 GB, real weights, q8 tier** (routed experts 8-bit, everything else bf16; 38 GB on
disk): 33.1-33.5 tok/s unconstrained, 37.3 tok/s constrained; peak MLX memory 39.0 GB; ~126 s
of decode per image; prompt prefill 0.8 s warm, 14.4 s cold (kernel JIT); pixel decode 0.84 s.

**M5 Max 36 GB (32-core GPU), full-size random weights** (same shapes and kernels; real-weight M5
numbers for e6d8/e4d8 are in RESULTS.md and agree within a few percent):

| Tier | Resident | ms / token, B=1 | tok/s B=1 | agg tok/s B=4 | ~s / 1024² image |
|---|---|---|---|---|---|
| q4 (experts 4-bit, rest bf16) + GPU embed | 23.0 GB | 16.0 | 62.4 | 194 | 67 |
| q5 (rest bf16) + GPU embed | 26.7 GB | 26.6 (memory pressure) | 37.6 | 180 | 111 |
| q6 (rest bf16) | ~30 GB | swaps | - | - | does not fit |
| experts 4 + dense/shared/head 8 + GPU embed | 20.8 GB | 10.9 | 91.5 | 245 | 46 |
| experts 5 + dense/shared/head 8 + GPU embed | 24.5 GB | 11.4 | 87.6 | 229 | 48 |
| experts 6 + dense/shared/head 8 (memory-mapped embed) | 26.4 GB | 12.9 | 77.6 | 204 | 54 |
| same, constrained | 26.4 GB | 11.9 | 84.2 | 216 | 50 |

Uncompiled (eager) decoding: 75 ms/token (13 tok/s) for q4 - the compiled fixed-shape step is
4.7x faster. GPU-resident embedding (+1.76 GB) lets the GPU run a step ahead (+6%).
