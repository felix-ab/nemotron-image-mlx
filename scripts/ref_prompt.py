"""Reference prompt token ids via the upstream HF tokenizer (transformers + remote code)."""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
SYSTEM = ("Generate an image matching the requested resolution by detailing the color, quantity, text, shape, size, "
          "texture, spatial relationships of the objects and background.")
code = Path(sys.argv[1]); out = Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(str(code), trust_remote_code=True)
cases = [("A cinematic photograph of a red fox in a snowy pine forest at dawn.", 1024, 1024),
         ("A cinematic photograph of a red fox in a snowy pine forest at dawn.", 1184, 880),
         ('Poster with the title "Echoes of Courage" — émigré café 東京 ✓', 880, 1184),
         ("x", 1248, 832)]
res = []
for p, w, h in cases:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Generate a {w}x{h} image.\n\n{p}"}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors=None)
    if isinstance(ids, dict): ids = ids["input_ids"]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    res.append({"prompt": p, "w": w, "h": h, "ids": list(map(int, ids)), "text": text})
ids = {k: tok.convert_tokens_to_ids(k) for k in ["<|img_start|>", "<|img_end|>", "<|vis_row|>", "<|im_end|>"]}
hdr = tok.base_tokenizer.encode("height=1024 width=1024", add_special_tokens=False)
out.write_text(json.dumps({"cases": res, "special": ids, "eos": tok.base_tokenizer.eos_token_id, "hdr_example": hdr}, ensure_ascii=False, indent=1))
print(json.dumps(res[0]["text"])); print(len(res[0]["ids"]), ids)
