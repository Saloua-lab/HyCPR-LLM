"""Actionability of the conformal sets (review 4: "intervals very wide, limiting actionability"): with the locally
adaptive (normalized) score, the set size varies with the LLM's uncertainty; share of test predictions whose set has
at most 2 admissible ratings, and the coverage on them, HyCPR-LLM (Mistral), split CP, 1,000 splits, eps = 0.1 and 0.2.
Usage: python cp_actionable.py   → results/cp_actionable.json
"""
import json
import numpy as np
from cp_adaptive import K, load, q

out = {}
for s in ["LDOS-H", "Depaul-H", "Restaurant-H", "MovieLens-H", "AmazonBooks-H"]:
    t, hy, P, mu = load("mistral", s)
    sig = np.sqrt((P * (K[None] - mu[:, None]) ** 2).sum(1)) + 0.1
    for eps in (0.1, 0.2):
        for name, sc, half in (("abs", np.abs(t - hy), lambda tau: tau + 0 * hy), ("normalized", np.abs(t - hy) / sig, lambda tau: tau * sig)):
            rng = np.random.default_rng(0)
            n, ncal = len(t), int(0.3 * len(t))
            share, cov_small, cov = [], [], []
            for _ in range(1000):
                perm = rng.permutation(n); cal, te = perm[:ncal], perm[ncal:]
                h = half(q(sc[cal], eps))[te]
                S = (K[None] >= hy[te, None] - h[:, None] - 1e-9) & (K[None] <= hy[te, None] + h[:, None] + 1e-9)
                hit = S[np.arange(len(te)), (t[te] - 1).astype(int)]
                small = S.sum(1) <= 2
                share.append(small.mean()); cov.append(hit.mean())
                if small.any(): cov_small.append(hit[small].mean())
            out[f"{s} {eps} {name}"] = {"share_le2": 100 * float(np.mean(share)), "cov_le2": 100 * float(np.mean(cov_small)) if cov_small else None, "coverage": 100 * float(np.mean(cov))}
            r = out[f"{s} {eps} {name}"]
            print(f"{s} eps={eps} {name}: sets<=2 ratings {r['share_le2']:.1f}% (coverage on them {r['cov_le2'] if r['cov_le2'] is None else round(r['cov_le2'],1)}%), marginal {r['coverage']:.1f}%")
json.dump(out, open("results/cp_actionable.json", "w"), indent=1)
