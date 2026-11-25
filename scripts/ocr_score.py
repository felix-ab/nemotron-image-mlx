"""Text accuracy of renders: Apple Vision OCR vs the quoted strings in each prompt.

  python ocr_score.py --cases a.json b.json --dirs tierA=dirA tierB=dirB --out ocr.json
Scores per image: word recall (fraction of requested words found, case-insensitive) and mean phrase
similarity (best fuzzy match of each quoted phrase in the OCR text, difflib ratio).
"""
import argparse, difflib, json, re
from pathlib import Path

import Quartz
import Vision
from Foundation import NSURL


def ocr(path):
    url = NSURL.fileURLWithPath_(str(path))
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    h.performRequests_error_([req], None)
    return [o.topCandidates_(1)[0].string() for o in (req.results() or [])]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("·", " "))


ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="+", required=True)
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
cases = [c for f in args.cases for c in json.loads(Path(f).read_text())]
res = {}
for spec in args.dirs:
    tier, d = spec.split("=", 1)
    rows = []
    for c in cases:
        phrases = re.findall(r'"([^"]+)"', c["prompt"])
        if not phrases:
            continue
        for s in c["seeds"]:
            p = Path(d) / f"{c['name']}_s{s}.png"
            if not p.exists():
                continue
            lines = ocr(p)
            text = norm(" ".join(lines))
            words = [w for ph in phrases for w in norm(ph).split()]
            found = sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", text))
            sims = []
            for ph in phrases:
                target = norm(ph)
                best = max((difflib.SequenceMatcher(None, target, norm(l)).ratio() for l in lines), default=0.0)
                # phrases split across lines: also compare against the full text window
                sm = difflib.SequenceMatcher(None, text, target)
                m = sm.find_longest_match(0, len(text), 0, len(target))
                best = max(best, m.size / max(len(target), 1))
                sims.append(best)
            rows.append({"image": p.name, "word_recall": found / max(len(words), 1), "phrase_sim": sum(sims) / len(sims),
                         "n_words": len(words)})
    n = len(rows)
    res[tier] = {"n_images": n, "word_recall": sum(r["word_recall"] for r in rows) / max(n, 1),
                 "phrase_sim": sum(r["phrase_sim"] for r in rows) / max(n, 1), "rows": rows}
    print(f"{tier}: {n} images, word recall {res[tier]['word_recall']:.3f}, phrase similarity {res[tier]['phrase_sim']:.3f}")
Path(args.out).write_text(json.dumps(res, indent=1))
