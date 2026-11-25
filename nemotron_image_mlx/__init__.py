"""Nemotron-Labs-Image-30B-A3B on Apple Silicon (MLX)."""
import os

# On M5-class GPUs MLX runs float32 GEMMs with M > 1 in TF32 by default (rel. error ~8e-4).
# The MoE router and the fp32 pixel decoder need true fp32; MLX reads this variable at the
# first fp32 GEMM in the process, so set it before any work happens.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

__version__ = "0.1.0"
