"""Localized conformal prediction with a random-forest localizer (review 5: "stronger connections to localized CP (e.g.,
LCP-RF) ... could suggest ways to narrow intervals and improve adaptiveness"; Guan 2023, Amoukou & Brunel 2023).

Score: absolute residual |r - r_hat| of the fused HyCPR-LLM rating. Split CP with 30% of the test predictions as
calibration data, as in cp_adaptive.py; the calibration part is split in two halves:
  A  fits a random forest (100 trees, min. 10 samples per leaf) that predicts the score from features known at test
     time: the fused rating, the mean, standard deviation and entropy of the LLM's distribution over the 5 ratings,
     and the absolute gap between the fused rating and the LLM's mean (i.e. how much the CF model moved it);
  B  gives the calibration scores, weighted for a test point x by the forest's localizer
     w_i(x) = (1/T) sum_t 1[leaf_t(x_i) = leaf_t(x)] / (n_t(x) + 1), with self-weight (1/T) sum_t 1 / (n_t(x) + 1)
     put on +infinity (weighted split CP, Tibshirani et al. 2019; Guan 2023), so that the threshold tau(x) is the
     quantile of the weighted scores and the set is {k : |k - r_hat(x)| <= tau(x)}. localized_rf_raw uses the level
     1 - eps; localized_rf (reported) uses Guan's calibrated level: the smallest level at which the leave-one-out
     coverage of the B points (each B point in turn playing the test point) reaches (1 - eps)(1 + 1/|B|).
The absolute and normalized scores of cp_adaptive.py are recomputed on the same random splits (calibration = the whole
30%) for comparison. 200 random splits (the forest is refitted on each); eps = 0.1.
Reported: coverage, mean number of admissible ratings, coverage of the ratings 1-2, share of predictions with at most
two admissible ratings, and coverage of those.

Usage: python cp_localized.py → results/cp_localized.{json,md}
"""
import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from cp_adaptive import K, LLMS, SETTINGS, load, q

EPS, SPLITS = 0.10, 200


def leaf_weights(leaf_a, leaf_B):
    """(na, nB) localizer weights of the B points for each row of leaf_a, and the self-weights (na,)."""
    T = leaf_B.shape[1]
    W = np.zeros((len(leaf_a), len(leaf_B)))
    selfw = np.zeros(len(leaf_a))
    for t in range(T):
        same = leaf_a[:, t][:, None] == leaf_B[:, t][None]
        n = same.sum(1)
        W += same / (n + 1)[:, None]
        selfw += 1 / (n + 1)
    return W / T, selfw / T


def tau_at(W, selfw, sB, level):
    """Weighted quantile at `level` of sum_i W_ji delta_{s_i} + selfw_j delta_{inf}, for each row j."""
    order = np.argsort(sB)
    cw = np.cumsum(W[:, order], 1) / (W.sum(1) + selfw)[:, None]
    k = (cw < level - 1e-12).sum(1)
    s_sorted = np.r_[sB[order], np.inf]
    return s_sorted[np.minimum(k, len(sB))]


def calibrated_level(leaf_B, sB, eps, grid=np.linspace(0.80, 0.999, 200)):
    """Guan (2023): smallest level whose leave-one-out coverage on the B points reaches (1 - eps)(1 + 1/nB)."""
    W, selfw = leaf_weights(leaf_B, leaf_B)
    selfw = selfw.copy()
    np.fill_diagonal(W, 0)     # point i plays the role of the test point: its own mass is the self-weight
    target = min(1.0, (1 - eps) * (1 + 1 / len(sB)))
    for lev in grid:
        if np.mean(sB <= tau_at(W, selfw, sB, lev)) >= target:
            return lev
    return 1.0


def summarize(t, hits, sizes):
    small = sizes <= 2
    return {"coverage": 100 * hits.mean(), "set_size": float(sizes.mean()),
            "low_coverage": 100 * hits[t <= 2].mean() if (t <= 2).any() else float("nan"),
            "small_share": 100 * small.mean(), "small_coverage": 100 * hits[small].mean() if small.any() else float("nan")}


def main():
    out, md = {}, ["| LLM | setting | method | coverage | admissible ratings | coverage 1-2 | sets <= 2 ratings (%) | coverage of these |",
                   "|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        for m in LLMS:
            try:
                t, hy, P, mu = load(m, s)
            except FileNotFoundError:
                continue
            sig = np.sqrt((P * (K[None] - mu[:, None]) ** 2).sum(1))
            ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
            X = np.c_[hy, mu, sig, ent, np.abs(hy - mu)]
            score = np.abs(t - hy)
            n, ncal = len(t), int(0.3 * len(t))
            rng = np.random.default_rng(0)
            acc = {k: {"hit": [], "size": [], "t": []} for k in ("abs", "normalized", "localized_rf_raw", "localized_rf")}
            for _ in range(SPLITS):
                perm = rng.permutation(n)
                cal, te = perm[:ncal], perm[ncal:]
                A, B = cal[: ncal // 2], cal[ncal // 2:]
                taus = {"abs": np.full(len(te), q(score[cal], EPS)),
                        "normalized": q(score[cal] / (sig[cal] + 0.1), EPS) * (sig[te] + 0.1)}
                rf = RandomForestRegressor(100, min_samples_leaf=10, random_state=0, n_jobs=8).fit(X[A], score[A])
                lB, lt = rf.apply(X[B]), rf.apply(X[te])
                W, selfw = leaf_weights(lt, lB)
                taus["localized_rf_raw"] = tau_at(W, selfw, score[B], 1 - EPS)
                taus["localized_rf"] = tau_at(W, selfw, score[B], calibrated_level(lB, score[B], EPS))
                for k, tau in taus.items():
                    S = np.abs(K[None] - hy[te][:, None]) <= tau[:, None] + 1e-9
                    acc[k]["hit"].append(S[np.arange(len(te)), (t[te] - 1).astype(int)])
                    acc[k]["size"].append(S.sum(1))
                    acc[k]["t"].append(t[te])
            res = {k: summarize(np.concatenate(v["t"]), np.concatenate(v["hit"]), np.concatenate(v["size"])) for k, v in acc.items()}
            out[f"{m} {s}"] = res
            for k, r in res.items():
                md.append(f"| {m} | {s} | {k} | {r['coverage']:.1f} | {r['set_size']:.2f} | {r['low_coverage']:.0f} | "
                          f"{r['small_share']:.0f} | {r['small_coverage']:.0f} |")
            print("\n".join(md[-4:]), flush=True)
    json.dump(out, open("results/cp_localized.json", "w"), indent=1)
    open("results/cp_localized.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
