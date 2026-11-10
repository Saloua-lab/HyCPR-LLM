"""Conformalized quantile regression with learned quantile regressors, and conformal baselines without the LLM
(review 6: "strong hybrid/quantile baselines for uncertainty (e.g., conformalized quantile regression)").

Same split protocol as cp_localized.py (30% of the test predictions as calibration data, split in two halves A and B,
200 random splits, eps = 0.1), so that every learned variant uses A to fit and B to conformalize:
  abs_hycpr   absolute residual |r - r_hat| of HyCPR-LLM, calibrated on the whole 30% (the method of the paper)
  cqr_hycpr   CQR (Romano et al. 2019): gradient-boosted quantile regressors (levels eps/2 and 1 - eps/2) fitted on A,
              features: fused rating, CF rating, LLM mean / std / entropy, log(1 + #training ratings of the user),
              log(1 + #training ratings of the item); score max(q_lo - r, r - q_hi) calibrated on B
  abs_cf      absolute residual of the CF model of the fusion alone (no LLM), calibrated on the whole 30%
  cqr_cf      CQR as above with the CF-only features (CF rating, user and item counts): a non-LLM quantile baseline
The CF rating is recovered from the fusion, b = (r_hat - w * LLM mean) / (1 - w) (w < 1 in every setting).
Sets are reported as admissible integer ratings, as in cp_adaptive.py.

Usage: python cp_cqr.py → results/cp_cqr.{json,md}
"""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cp_adaptive import K, LLMS, SETTINGS, load, q
from cp_localized import summarize
from segments import counts_large, counts_small

EPS, SPLITS = 0.10, 200
CNT = {"LDOS-H": ("small", "LDOS"), "Depaul-H": ("small", "Depaul"), "Restaurant-H": ("small", "Tijuana"),
       "MovieLens-H": ("large", "movielens"), "AmazonBooks-H": ("large", "amazon")}


def qreg(X, y, a):
    return HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=10, random_state=0).fit(X, y)


def main():
    out, md = {}, ["| LLM | setting | method | coverage | admissible ratings | coverage 1-2 | sets <= 2 ratings (%) | coverage of these |",
                   "|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        kind, name = CNT[s]
        uc, ic = counts_small(name) if kind == "small" else counts_large(name)
        for m in LLMS:
            try:
                t, hy, P, mu = load(m, s)
            except FileNotFoundError:
                continue
            w = json.load(open(f"probs/hybrid/{m}_hycpr_{s}.json"))["w"]
            b = (hy - w * mu) / (1 - w)
            sig = np.sqrt((P * (K[None] - mu[:, None]) ** 2).sum(1))
            ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
            cnt = np.c_[np.log1p(uc), np.log1p(ic)]
            feats = {"cqr_hycpr": (np.c_[hy, b, mu, sig, ent, cnt], hy), "cqr_cf": (np.c_[b, cnt], b)}
            n, ncal = len(t), int(0.3 * len(t))
            rng = np.random.default_rng(0)
            acc = {k: {"hit": [], "size": [], "t": []} for k in ("abs_hycpr", "cqr_hycpr", "abs_cf", "cqr_cf")}
            for _ in range(SPLITS):
                perm = rng.permutation(n)
                cal, te = perm[:ncal], perm[ncal:]
                A, B = cal[: ncal // 2], cal[ncal // 2:]
                lohi = {"abs_hycpr": (hy - q(np.abs(t - hy)[cal], EPS), hy + q(np.abs(t - hy)[cal], EPS)),
                        "abs_cf": (b - q(np.abs(t - b)[cal], EPS), b + q(np.abs(t - b)[cal], EPS))}
                for k, (X, center) in feats.items():
                    lo = qreg(X[A], t[A], EPS / 2).predict(X)
                    hi = qreg(X[A], t[A], 1 - EPS / 2).predict(X)
                    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                    tau = q(np.maximum(lo - t, t - hi)[B], EPS)
                    lohi[k] = (lo - tau, hi + tau)
                for k, (lo, hi) in lohi.items():
                    S = (K[None] >= lo[te][:, None] - 1e-9) & (K[None] <= hi[te][:, None] + 1e-9)
                    acc[k]["hit"].append(S[np.arange(len(te)), (t[te] - 1).astype(int)])
                    acc[k]["size"].append(S.sum(1))
                    acc[k]["t"].append(t[te])
            res = {k: summarize(np.concatenate(v["t"]), np.concatenate(v["hit"]), np.concatenate(v["size"]))
                   for k, v in acc.items()}
            out[f"{m} {s}"] = res
            for k, r in res.items():
                md.append(f"| {m} | {s} | {k} | {r['coverage']:.1f} | {r['set_size']:.2f} | {r['low_coverage']:.0f} | "
                          f"{r['small_share']:.0f} | {r['small_coverage']:.0f} |")
            print("\n".join(md[-4:]), flush=True)
    json.dump(out, open("results/cp_cqr.json", "w"), indent=1)
    open("results/cp_cqr.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
