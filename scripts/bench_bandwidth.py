"""Achievable unified-memory bandwidth and quantized GEMV throughput on this GPU."""
import time, json, mlx.core as mx, numpy as np
def t(f, n=20):
    f(); mx.synchronize()
    s = time.perf_counter()
    for _ in range(n): f()
    mx.synchronize()
    return (time.perf_counter() - s) / n
res = {"device": mx.device_info()["device_name"]}
x = mx.random.normal((1024, 1024, 256)).astype(mx.bfloat16); mx.eval(x)   # 512 MB
res["sum_read_GBs"] = x.nbytes / t(lambda: mx.eval(x.sum())) / 1e9
# GEMV bf16 (decode-like): 1 x 2688 @ 2688 x 327680 (lm_head)
w = mx.random.normal((327680, 2688)).astype(mx.bfloat16); v = mx.random.normal((1, 2688)).astype(mx.bfloat16); mx.eval(w, v)
res["gemv_bf16_lm_head_GBs"] = w.nbytes / t(lambda: mx.eval(v @ w.T)) / 1e9
for bits in (8, 6, 5, 4):
    wq, s, b = mx.quantize(w, group_size=64, bits=bits); mx.eval(wq, s, b)
    nb = wq.nbytes + s.nbytes + b.nbytes
    dt = t(lambda: mx.eval(mx.quantized_matmul(v, wq, s, b, transpose=True, group_size=64, bits=bits)))
    res[f"qmv_{bits}bit_GBs"] = nb / dt / 1e9; res[f"qmv_{bits}bit_ms"] = dt * 1e3
print(json.dumps(res, indent=1))
