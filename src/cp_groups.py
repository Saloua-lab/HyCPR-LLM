"""Conditional coverage of split CP by history length, user / item frequency and rating region, on all five -H
settings (review 4: "no conditional coverage by user history length, rating scale region, or by user/item
frequency"). HyCPR-LLM (Mistral), eps = 0.1, 30% calibration, 1,000 random splits (as cp_adaptive.py), for the
absolute residual and for the CQR-style score of cp_adaptive.py.

Groups of the test interactions:
  history   number of history items in the test prompt (after leak removal for the small datasets):
            0-2 / 3-4 / 5+ (small datasets: at most 5; MovieLens-1M and Amazon-Books: at most 10)
  user, item  tertiles of the number of training ratings (segments.py)
  rating    true rating 1-2 / 3 / 4-5
Usage: python cp_groups.py   → results/cp_groups.{json,md}
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_adaptive import K, q, quantile_of  # noqa: E402
from segments import counts_large, counts_small, tertiles  # noqa: E402

EPS = 0.1
SETTINGS = {"LDOS-H": ("small", "LDOS"), "Depaul-H": ("small", "Depaul"), "Restaurant-H": ("small", "Tijuana"),
            "MovieLens-H": ("large", "movielens"), "AmazonBooks-H": ("large", "amazon")}


def hist_len(ex):
    body = ex.split("### Response")[0]
    return len(re.findall(r"out of 5", body))


def main():
    from datasets import load_from_disk
    out, md = {}, []
    for s, (kind, key) in SETTINGS.items():
        H = json.load(open(f"probs/hybrid/mistral_hycpr_{s}.json"))
        L = json.load(open(f"probs/variants/mistral_v0_{s}_test.json" if kind == "small" else f"probs/mistral_full_{s}.json"))
        t, hy = np.array(H["truth"], float), np.array(H["expected"], float)
        P, mu = np.array(L["probs"], float), np.array(L["expected"], float)
        assert np.array_equal(t, np.array(L["truth"], float))
        ex = load_from_disk(L["data"])["test"]["example"]
        hl = np.array([hist_len(e) for e in ex])
        uc, ic = counts_small(key) if kind == "small" else counts_large(key)
        groups = {"history": np.where(hl <= 2, "0-2", np.where(hl <= 4, "3-4", "5+")),
                  "user": tertiles(uc)[0], "item": tertiles(ic)[0],
                  "rating": np.where(t <= 2, "1-2", np.where(t == 3, "3", "4-5"))}
        qlo, qhi = quantile_of(P, EPS / 2) + (hy - mu), quantile_of(P, 1 - EPS / 2) + (hy - mu)
        scores = {"abs": np.abs(t - hy), "cqr": np.maximum(qlo - t, t - qhi)}
        sets = {"abs": lambda tau: (K[None] >= hy[:, None] - tau - 1e-9) & (K[None] <= hy[:, None] + tau + 1e-9),
                "cqr": lambda tau: (K[None] >= qlo[:, None] - tau - 1e-9) & (K[None] <= qhi[:, None] + tau + 1e-9)}
        n, ncal = len(t), int(0.3 * len(t))
        rec = {}
        for sc in ("abs", "cqr"):
            rng = np.random.default_rng(0)
            hits = {(g, v): [] for g, lab in groups.items() for v in np.unique(lab)}
            marg, size = [], []
            for _ in range(1000):
                perm = rng.permutation(n)
                cal, te = perm[:ncal], perm[ncal:]
                S = sets[sc](q(scores[sc][cal], EPS))[te]
                hit = S[np.arange(len(te)), (t[te] - 1).astype(int)]
                marg.append(hit.mean())
                size.append(S.sum(1).mean())
                for g, lab in groups.items():
                    for v in np.unique(lab):
                        m = lab[te] == v
                        if m.any():
                            hits[(g, v)].append(hit[m].mean())
            rec[sc] = {"marginal": 100 * float(np.mean(marg)), "size": float(np.mean(size)),
                       "groups": {f"{g}: {v}": {"n": int((groups[g] == v).sum()), "coverage": 100 * float(np.mean(h))}
                                  for (g, v), h in hits.items()}}
        out[s] = rec
        md.append(f"## {s} (history length: min {hl.min()}, max {hl.max()})")
        for sc in ("abs", "cqr"):
            r = rec[sc]
            md.append(f"- {sc}: marginal {r['marginal']:.1f}%, size {r['size']:.2f}; " + "; ".join(
                f"{k} {v['coverage']:.0f}% (n={v['n']})" for k, v in r["groups"].items()))
        print("\n".join(md[-3:]), flush=True)
    json.dump(out, open("results/cp_groups.json", "w"), indent=1)
    open("results/cp_groups.md", "w").write("\n".join(md))


if __name__ == "__main__":
    main()
