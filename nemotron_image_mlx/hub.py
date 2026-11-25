"""Resolve --model: a local directory, a published tier name, or a Hugging Face repo id."""

from __future__ import annotations

from pathlib import Path

TIERS = {
    "8bit": "VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-8bit",
    "5bit": "VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-5bit",
    "6bit": "VisualInference/Nemotron-Labs-Image-30B-A3B-MLX-6bit",
}

# Unified memory needed to run each tier comfortably (measured peaks: 39.1, 27.9, 29.5 GB).
_MIN_MEMORY_GB = {"8bit": 44, "5bit": 32, "6bit": 34}


def auto_tier() -> str:
    """Highest-fidelity tier that fits this Mac: 8bit (KL 1.05x the bf16 floor) needs ~44 GB, 5bit (1.6x) ~32 GB."""
    import mlx.core as mx

    mem_gb = mx.device_info().get("memory_size", 0) / 1e9
    if mem_gb >= _MIN_MEMORY_GB["8bit"]:
        return "8bit"
    if mem_gb >= _MIN_MEMORY_GB["5bit"]:
        return "5bit"
    raise SystemExit(f"This Mac has {mem_gb:.0f} GB of unified memory; the smallest published tier needs ~32 GB.")


def resolve_model(spec: str) -> Path:
    p = Path(spec).expanduser()
    if p.is_dir():
        return p
    if spec == "auto":
        spec = auto_tier()
    repo = TIERS.get(spec, spec)
    if "/" not in repo:
        raise SystemExit(f"--model {spec!r}: not a directory, a tier ({', '.join(TIERS)}, auto) or a repo id")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, ignore_patterns=["code/*", "assets/*", "*.md"]))
