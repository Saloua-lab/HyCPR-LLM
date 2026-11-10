"""Conformal prediction intervals (RQ2) on the HyCPR-LLM predictions (LLM + CF fusion, hybrid.py --tag hycpr).

Same procedure as cp_probs.py for the intervals: nonconformity |r - r_hat|, split CP (30% calibration, mean over
1000 random splits) and sliding-window CP (100 previous predictions), widths clipped to the 1..5 scale. Prediction
sets need a distribution over the ratings, which the fused prediction does not have, so they are not computed here.
The LLM-only intervals of the same models are recomputed alongside for comparison (cp_probs.path).

Usage: python cp_hybrid.py [model ...]   (default: mistral llama eurollm deepseek)
       → results/cp_hybrid.{json,md}
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_probs import EPS, SCALE, SETTINGS, path  # noqa: E402
from summarize import sliding_cp, split_cp  # noqa: E402


def intervals(truth, pred):
    pairs = list(zip(truth, pred))
    out = {}
    for eps in EPS:
        sw, sc = split_cp(pairs, eps, SCALE, random.Random(0))
        lw, lc = sliding_cp(pairs, eps, SCALE)
        out[str(eps)] = {"split_width": sw, "split_cov": sc, "sliding_width": lw, "sliding_cov": lc}
    return out


def main(models):
    out = {}
    for m in models:
        for s in SETTINGS:
            hp = f"probs/hybrid/{m}_hycpr_{s}.json"
            if not os.path.exists(hp):
                continue
            h = json.load(open(hp))
            out[f"HyCPR-LLM {m} {s}"] = intervals(h["truth"], h["expected"])
            lp = path(f"{m}_clean", s)
            if os.path.exists(lp):
                d = json.load(open(lp))
                out[f"CPR-LLM (LLM only) {m} {s}"] = intervals(d["truth"], d["expected"])
    json.dump(out, open("results/cp_hybrid.json", "w"), indent=1)
    with open("results/cp_hybrid.md", "w") as f:
        f.write("| eps | model setting | split width | split cov | sliding width | sliding cov |\n|---|---|---|---|---|---|\n")
        for eps in EPS:
            for k, r in out.items():
                v = r[str(eps)]
                f.write(f"| {eps} | {k} | {v['split_width']:.2f} | {v['split_cov']:.2f} | "
                        f"{v['sliding_width']:.2f} | {v['sliding_cov']:.2f} |\n")
    print(open("results/cp_hybrid.md").read())


if __name__ == "__main__":
    main(sys.argv[1:] or ["mistral", "llama", "eurollm", "deepseek"])
