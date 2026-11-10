"""Conformal prediction (RQ2 table and prediction sets) from the probability-scored predictions in probs/.

Intervals: nonconformity |r - r_hat| with r_hat the expected rating; split CP (30% calibration, mean over 1000
random splits) and sliding-window CP (100 previous predictions), widths clipped to the 1..5 scale. These are the
functions of summarize.py.
Sets: nonconformity 1 - p(r | x), split CP; mean set size and coverage.

Usage: python cp_probs.py [tag ...]   tags such as mistral_clean llama   (default: the leak-free-trained models
of all four LLMs plus the original Mistral and Llama-2 for comparison) → results/cp_probs.{json,md}
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import CAL_RATIO, N_SPLITS, cp_quantile, sliding_cp, split_cp

SETTINGS = {"LDOS-H": "LDOS-H", "Restaurant-H": "Restaurant-H", "Depaul-H": "Depaul-H"}
EPS = [0.10, 0.05, 0.01]
SCALE = (1, 5)


def path(tag, setting):
    # original adapters were scored on the leak-free copy of the test set (_noleak); clean ones only exist there
    return f"probs/{tag}_{setting}.json" if "_clean" in tag else f"probs/{tag}_{setting}_noleak.json"


def set_cp(probs, truth, labels, eps, rng):
    probs, truth = np.array(probs), np.array(truth)
    lab = {l: i for i, l in enumerate(labels)}
    score_true = 1 - probs[np.arange(len(truth)), [lab[t] for t in truth]]
    n_cal, idx = int(len(truth) * CAL_RATIO), list(range(len(truth)))
    size = cov = 0.0
    for _ in range(N_SPLITS):
        rng.shuffle(idx)
        cal, test = idx[:n_cal], idx[n_cal:]
        tau = cp_quantile(list(score_true[cal]), eps)
        sets = (1 - probs[test]) <= tau
        size += sets.sum(1).mean()
        cov += (score_true[test] <= tau).mean()
    return size / N_SPLITS, 100 * cov / N_SPLITS


def main(tags):
    out = {}
    for tag in tags:
        for setting in SETTINGS:
            p = path(tag, setting)
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            pairs = list(zip(d["truth"], d["expected"]))
            r = {}
            for eps in EPS:
                sw, sc = split_cp(pairs, eps, SCALE, random.Random(0))
                lw, lc = sliding_cp(pairs, eps, SCALE)
                zs, zc = set_cp(d["probs"], d["truth"], d["labels"], eps, random.Random(0))
                r[str(eps)] = {"split_width": sw, "split_cov": sc, "sliding_width": lw, "sliding_cov": lc,
                               "set_size": zs, "set_cov": zc}
            out[f"{tag} {setting}"] = r
    json.dump(out, open("results/cp_probs.json", "w"), indent=1)
    with open("results/cp_probs.md", "w") as f:
        f.write("| eps | model setting | split width | split cov | sliding width | sliding cov | set size | set cov |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for eps in EPS:
            for k, r in out.items():
                v = r[str(eps)]
                f.write(f"| {eps} | {k} | {v['split_width']:.2f} | {v['split_cov']:.2f} | {v['sliding_width']:.2f} | "
                        f"{v['sliding_cov']:.2f} | {v['set_size']:.2f} | {v['set_cov']:.2f} |\n")
    print(open("results/cp_probs.md").read())


if __name__ == "__main__":
    main(sys.argv[1:] or ["mistral_clean", "llama_clean", "eurollm_clean", "deepseek_clean", "mistral", "llama"])
