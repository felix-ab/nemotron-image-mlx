# nemotron-image-mlx

![Renders from the 8-bit tier at seed 42](docs/images/hero.jpg)

NVIDIA's [Nemotron-Labs-Image-30B-A3B](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B) on Apple Silicon
with MLX. The model writes an image as a grid of discrete visual tokens with a hybrid Mamba2 / attention / 128-expert
MoE language model (30B total, 3B active per token), then decodes them with its L24SQ ViT. Unofficial; not affiliated
with or endorsed by NVIDIA. **Non-commercial use only** (research and evaluation), per the NVIDIA License.

```bash
pip install nemotron-image-mlx
nemotron-image "A risograph concert poster in pink and teal with the headline LOW TIDE ORCHESTRA" --size 880x1184 --seed 42 --out poster.png
```

`--model auto` (the default) downloads the best tier for your Mac. Sizes: multiples of 16; recommended 1024x1024,
1184x880, 1248x832, 880x1184. `--seeds 1,2,3,4` renders several candidates in one batch.

## Tiers

| Mac memory | `--model` | Weights | Fidelity (KL to the exact model, x the bf16 floor) | Speed, 1024x1024 |
|---|---|---|---|---|
| 48 GB+ | [`8bit`](https://huggingface.co/VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-8bit) | 38 GB | 1.05 | ~115 s, M4 Pro |
| 36 GB | [`5bit`](https://huggingface.co/VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-5bit) | 27 GB | 1.60 | ~67 s, M5 Max |
| 36 GB, faster | [`6bit`](https://huggingface.co/VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-6bit) (experts 6-bit, rest 8-bit) | 29 GB | 1.7 | ~52 s, M5 Max |

"bf16 floor" is how far bf16 itself, the official pipeline's precision, sits from the model run exactly in fp32
(0.0176 nats/token over 99,564 generated tokens). 4-bit tiers measured 3.1-3.5x and lost text accuracy; they are not
published. Your own tier: `nemotron-image-convert ./meta ./out --recipe q6 --hub nvidia/Nemotron-Labs-Image-30B-A3B`
streams the fp32 checkpoint with range requests and never stores it.

## What is in the port

- **The reference's numerics.** Rounding follows the official vLLM bf16 path (fp32 router logits from bf16 inputs,
  fp32 Mamba conv/decay/state, single-rounding RMSNorm). In fp32 every layer matches the upstream PyTorch module to
  ~1e-6 on the real weights.
- **vLLM's seeded sampler**, including the CUDA Philox exponential noise stream for a given seed, as a Metal kernel.
- **A compiled decode step**: one fixed-shape MLX graph per token (Mamba states, preallocated KV, noise), 4.4x faster
  than eager and ~85-88% of the measured bandwidth roofline.
- **Valid images by default.** At 880x1184 the original model writes `width=884` with probability 1.0, so the official
  script rejects its own output. Enforcing the image grammar after `<|img_start|>` reproduces every run the official
  pipeline accepts token for token (4,175/4,175 identical on real weights) and turns the rejected ones into valid
  images. `--unconstrained` samples exactly like the reference, failures included.
- **M5:** MLX runs float32 GEMMs in TF32 on M5 GPUs by default; the package turns that off at import.

**On seeds.** Token-for-token identity with the official outputs is not achievable by any implementation: bf16
rounding alone changes ~10% of sampled image tokens versus the exact model, starting in the first grid row, and every
later token depends on those. Even this port on an M4 and an M5 first differs at token 18 for the same tier and seed.
A seed gives a picture from the same prompt family, not the same pixels.

![Same prompt and seed: 8-bit on an M4 Pro vs 6-bit on an M5 Max](docs/images/cross_tier_same_seed.jpg)

## Validation and benchmarks

Write-up: [docs/FINDINGS.md](docs/FINDINGS.md) (architecture, numerics, bugs found, M5 notes) and
[docs/RESULTS.md](docs/RESULTS.md) (fidelity, speed, memory, tier choice). The scripts that produced them are in
`scripts/` (`eval_paths.py` is the layer-major multi-precision evaluator; `cmp_*`/`ref_*` compare against the upstream
PyTorch code, which needs `torch` and `transformers`). Tested on an M5 Max (36 GB) and an M4 Pro (48 GB); other
Apple Silicon Macs with enough memory should work (M1 more slowly: no native bf16).

## License

NVIDIA License (non-commercial: research or evaluation only), the license of the model this ports; see `LICENSE` and
`NOTICE` for derived files and the Apache-2.0 portions.
