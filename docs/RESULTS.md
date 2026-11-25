# Results

All numbers measured 2026-11-25, MLX 0.32.5 / mlx-lm 0.32.2, Python 3.12. Raw data: `parity/`,
`results/`, `logs/`. Method and earlier findings: `FINDINGS.md`.

## Machines

| Machine | GPU | Memory | GPU working set | Role |
|---|---|---|---|---|
| MacBook Pro, Apple M5 Max | 32-core, `applegpu_g17s` (Metal 4, neural accelerators) | 36 GB | 30.15 GB | primary: real-weight renders of two tiers, fidelity eval, speed |
| Mac mini, Apple M4 Pro | `applegpu_g16s` | 48 GB | 40.2 GB | real-weight q8 tier, fidelity eval, torch reference, cross-generation |

**Tested:** M5 Max and M4 Pro, real weights. **Expected, untested:** M1-M3 and other M4/M5
variants. Nothing in the code is device-specific beyond what MLX selects itself (NAX kernels on
M5); the custom Metal kernels use only `simd_sum`, `mulhi` and `precise` math (Apple7+). M1 has no
native bf16, so MLX emulates it there: expect it to work, slower. Minimum memory follows from the
tier sizes below.

## 1. Seed parity: what is and is not achievable

- **Noise:** the sampler draws vLLM's exact CUDA Philox exponential stream for a seed
  (Random123-verified core; launch layout from torch 2.11 source; not run on a CUDA device).
- **Math:** every block equals the upstream torch module in fp32 (~1e-6 per layer on real
  weights); bf16 rounding points follow vLLM.
- **But token-for-token parity is impossible for this model, for any implementation.** bf16
  rounding alone (the reference's own precision) changes **9.65% of sampled image tokens** relative
  to the exact fp32 model, with the first change typically **~21 tokens in** (inside the first
  grid row). Image tokens at T = 0.9 over 196,560 Leech-lattice codes have many near-ties; a
  different GPU, TP degree or kernel would flip a similar share. Two seed-42 renders on different
  hardware therefore share the prompt's composition family, not pixels (`results/xtier_g003_g004_q8m4_vs_e6d8m5.jpg`).
- **Even this port is not token-identical across Apple GPU generations.** The same e6d8 tier, prompt
  and seed on the M4 Pro and the M5 Max first differ at token 18 (gallery #003) and token 61 (#004):
  the M5's neural-accelerator GEMMs round bf16 products differently from the M4's.
- **What is measurable instead** is distributional fidelity: KL between per-token sampling
  distributions, the noise-coupled flip rate, and NLL, against fp32 truth, compared with the
  reference's own bf16 floor.

## 2. Fidelity (24 real sequences = 99,564 generated tokens; 8 gallery prompts at 4 resolutions, 8 suite prompts x 2 seeds)

Truth = MLX fp32 (`MLX_ENABLE_TF32=0`), which tracks upstream torch fp32 at median 1.4e-6 per layer.
Two independent runs: M4 Pro (Hub-streamed) and M5 Max (local); they agree (bf16 KL 0.01763 vs
0.01767; flips 9.65% vs 9.62%).

| Tier (routed experts / rest) | KL vs fp32 (nats/token) | x bf16 floor | sampled-token flips vs fp32 | top-1 = fp32 | NLL - fp32 |
|---|---|---|---|---|---|
| bf16 / bf16 (= reference precision) | 0.0176 | 1.00 | 9.65% | 77.6% | +0.003 |
| **8-bit / bf16 (`q8`)** | **0.0185** | **1.05** | 9.95% | 77.4% | +0.001 |
| 6-bit / bf16 (`q6`) | 0.0210 | 1.19 | 10.3% | 76.9% | +0.008 |
| 6-bit / shared+head 8-bit | 0.0275 | 1.56 | 11.8% | 75.3% | +0.018 |
| **5-bit / bf16 (`q5`)** | **0.0282** | **1.60** | 11.9% | 75.4% | +0.017 |
| 6-bit / all dense 8-bit (`e6d8`) | 0.0290-0.0301 | 1.65-1.71 | 12.1% | 75.0% | +0.019 |
| 5-bit / shared+head 8-bit | 0.0342 | 1.94 | 13.0% | 74.0% | +0.025 |
| 5-bit / all dense 8-bit | 0.0358 | 2.03 | 13.4% | 73.9% | +0.027 |
| 4-bit / bf16 (`q4`) | 0.0549 | 3.12 | 15.8% | 71.1% | +0.049 |
| 4-bit / all dense 8-bit (`e4d8`) | 0.0618 | 3.51 | 16.8% | 69.7% | +0.056 |

Reading it:
- **q8 is indistinguishable from the reference's own precision** (KL +5%, NLL even slightly lower).
- Quality falls off smoothly to 5-bit (1.6x) and steeply at 4-bit (3.1-3.5x, NLL +0.05 nats/token).
- At the output, an **8-bit lm_head / shared expert costs more than 8-bit Mamba/attention
  projections** (+0.0065 vs +0.0008 KL at 4-bit), although Mamba projections drift more in early
  hidden states.
- **Text (Apple Vision OCR vs the quoted strings, 7 images per tier, indicative only):** phrase
  similarity q8 0.87 / 0.94 (gallery / suite), e6d8 0.84 / 0.92, e4d8 0.79 / 0.86; suite word
  recall 0.86 / 0.89 / 0.69. Same ordering as KL; 4-bit loses text.

Visual: `results/sheets/suite_tiers_labeled.jpg` (q8 | e6d8 | e4d8, seed 42, with 100% centre
crops) and a blinded version with its key. Within one seed, skin, film grain and gradients look
alike across tiers; the e4d8 lichen reads slightly more uniform. One seed per prompt is not enough to
call texture differences; the eval table is the quantitative basis.

## 3. Speed and memory

### Real weights

| Machine / tier | Batch | tok/s per image | 1024² decode | Prefill warm / cold | Pixel decode | Peak memory |
|---|---|---|---|---|---|---|
| M5 Max / e6d8 | 1 | 78-82 | ~52 s | 0.4 s / 12.9 s | 0.44 s (fp32) | 29.5 GB |
| M5 Max / e6d8 | 2 | 61 (122 agg.) | ~34 s / image | | | |
| M5 Max / e4d8 | 1 | 88-91 | ~47 s | 0.3 s / 1.7 s (shaders cached by macOS from the previous tier) | 0.45 s | 24.0 GB |
| M5 Max / e4d8 | 2 | 70-76 (~146 agg.) | ~29 s / image | | | 23.8 GB |
| M5 Max / q5 | 1 | 61-63 | ~67 s | 0.3 s / 13.6 s | | 27.9 GB |
| M4 Pro / q8 | 1 | 36.4 (33-37) | ~115 s | 1.1 s / 12.6-16.6 s | 0.85 s | 39.1 GB |
| M4 Pro / q8 | 2 | 29.8 (60 agg.) | ~70 s / image | | | 39.4 GB |
| M4 Pro / q8 | 4 | 21.1 (84 agg.) | ~50 s / image | | | |

All renders valid (0 of 72 failed) with constrained decoding.

### M5 Max, full-size random weights (identical shapes and kernels; used for tiers not rendered)

| Tier | Resident | ms/token B=1 | agg tok/s B=4 | ~s / 1024² |
|---|---|---|---|---|
| q4 (+GPU embedding) | 23.0 GB | 16.0 | 194 | 67 |
| **q5, constrained, memory-mapped embedding** | **25.0 GB** | **15.4** | 114 (B=2) | **64** |
| q6 / bf16 rest | ~30 GB | swaps on 36 GB | - | does not fit |
| 5-bit / shared+head 8 | 23.7 GB | 13.5 | 205 | 56 |
| 6-bit / shared+head 8 | 27.4 GB | 13.9 | 200 | 58 |
| e6d8, constrained | 26.4 GB | 11.9 | 216 | 50 |
| e4d8 | 20.8 GB | 10.9 | 245 | 46 |
| q4, eager (uncompiled) | 23.0 GB | 75.5 | - | 315 |

Bandwidth roofline on this M5 Max (~390 GB/s achieved): q4 with bf16 dense reads ~5.5 GB/token
-> 14.1 ms; the compiled step reaches 16.0 ms (88%). The M4 Pro q8 run reaches ~85% of its roofline.

### What each optimisation bought (M5 Max)

| Change | Effect |
|---|---|
| Compiled fixed-shape decode step (vs eager) | 75.5 -> 17.0 ms/token (4.4x) |
| GPU-resident embedding (pipelined, +1.76 GB) | 17.0 -> 16.0 ms |
| 8-bit dense/shared/head (speed lever; costs fidelity, see §2) | 16.0 -> 10.9 ms |
| Constrained decoding (visual rows of the head only) | -2 to -7% on M5 (8-bit head); +12% tok/s on M4 (bf16 head) |
| Batched candidates | 3.1x aggregate throughput at B = 4 |
| Cold start (kernel JIT) | 10-17 s on the first image in a process; macOS caches compiled shaders across processes (a second tier's first prefill: 1.7 s). Recurrent prefill did not reduce it on real weights (9.5-13.5 s vs 12.6 s), so parallel prefill stays the default |

## 4. Recommended tiers

| Mac memory | Tier | Why |
|---|---|---|
| >= 96 GB | `bf16` (65 GB) | exactly the reference precision |
| 48-64 GB | **`q8`** (38 GB) | KL 1.05x the bf16 floor; tested on M4 Pro 48 GB (peak 39 GB) |
| 36 GB | **`q5`** (27 GB on disk, 27.9 GB peak) | best fidelity that fits (1.6x); 61-63 tok/s, ~67 s per 1024² on M5 Max (real weights) |
| 36 GB, speed | `e6d8` (26.4 GB) | 1.7x, ~50 s; tight (29.4 GB peak) |
| 24-32 GB | `e4d8` (21 GB weights, 24 GB peak) | 3.5x; visible text degradation. Untested on a 24 GB machine (likely too tight there) |

Not shipped as defaults: 4-bit tiers (3x+ KL, OCR drop). Next lever if a better <= 24 GB tier is
wanted: GPTQ-style calibrated expert quantization (code from the Inkling port applies directly).
