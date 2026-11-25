"""Probe GEMM precision on this GPU: fp32 (TF32?) and bf16 accumulation, plain and quantized."""
import numpy as np, mlx.core as mx, os, json
rng = np.random.default_rng(0)
res = {"env_MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"), "arch": mx.device_info()["architecture"]}
def bf(x): return np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
for (M, K, N) in [(1, 2688, 4096), (8, 2688, 4096), (300, 2688, 4096), (4096, 1024, 1024)]:
    a = rng.standard_normal((M, K)).astype(np.float32); b = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    c32 = np.array(mx.matmul(mx.array(a), mx.array(b)))
    ab, bb = bf(a), bf(b)
    refb = ab.astype(np.float64) @ bb.astype(np.float64)          # exact product of bf16 inputs
    cr = bf(refb.astype(np.float32))                                # correctly rounded bf16 result
    c16 = np.array(mx.matmul(mx.array(a).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16)).astype(mx.float32))
    res[f"{M}x{K}x{N}"] = {
        "fp32_rel_err": float(np.linalg.norm(c32 - ref) / np.linalg.norm(ref)),
        "bf16_rel_err_vs_exact_of_bf16_inputs": float(np.linalg.norm(c16 - refb) / np.linalg.norm(refb)),
        "bf16_frac_equal_to_correctly_rounded": float((c16 == cr).mean()),
    }
print(json.dumps(res, indent=1))
