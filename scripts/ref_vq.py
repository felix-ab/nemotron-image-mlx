"""Reference L24SQ decode/encode with the upstream torch module (CPU), fp32 and bf16."""
import sys, json, numpy as np, torch
from pathlib import Path
from transformers import AutoModel
from PIL import Image
tokdir, out = Path(sys.argv[1]), Path(sys.argv[2])
gallery = Path(sys.argv[3]) if len(sys.argv) > 3 else None
rng = np.random.default_rng(0)
grid = (24, 32)                                    # 384x512 test grid (non-square)
idx = rng.integers(0, 196560, size=grid[0] * grid[1])
res = {"idx": idx}
for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    m = AutoModel.from_pretrained(str(tokdir), trust_remote_code=True, dtype=dt).eval()
    with torch.inference_mode():
        img = m.pack_decode(indices_list=[torch.tensor(idx)], grid_sizes=[grid])[0]
        res[f"dec_{name}"] = img.float().numpy()
        if gallery is not None:
            im = Image.open(gallery).convert("RGB")
            x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255
            x = m.preprocess_image(x).to(dt)
            enc = m.encode(x[None])
            res[f"enc_{name}"] = enc["indices"][0].numpy()
            res["enc_grid"] = np.array(enc["grid_size"])
    del m
np.savez(out, **res)
print({k: v.shape for k, v in res.items()})
