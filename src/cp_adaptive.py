"""Ordinal and locally adaptive conformal prediction (review 3: "global absolute-residual score ... ordinal CP
(ordinal APS) or locally adaptive CP (CQR) not explored").

All variants: split CP with 30% of the test predictions as calibration set, finite-sample quantile, 1,000 random
splits; prediction sets are reported as the set of admissible integer ratings, so that every variant is compared on
the same scale (coverage, mean number of admissible ratings, coverage of the ratings 1-2).

  abs        |r - r_hat| on the fused HyCPR-LLM rating (the method of the paper)
  normalized |r - r_hat| / (sigma + 0.1), sigma = standard deviation of the LLM's distribution over the 5 ratings
             (locally adaptive CP, Lei et al. 2018); interval r_hat +- tau (sigma + 0.1)
  cqr        CQR-style: q_lo, q_hi = eps/2 and 1-eps/2 quantiles of the LLM's distribution (linear interpolation of
             its CDF), shifted by (r_hat - LLM mean) so that they surround the fused rating; score
             max(q_lo - r, r - q_hi); interval [q_lo - tau, q_hi + tau] (Romano et al. 2019)
  ordinal_aps  ordinal APS on the LLM's distribution (LLM alone): from the most probable rating, the contiguous set
             is grown by adding the neighbour with the larger probability; the score is the mass of the smallest
             such set containing the true rating (Lu et al. 2022)
The LLM distribution is that of the LLM inside the fusion (small datasets: the v0 adapter fine-tuned on 90% of the
training data, probs/variants; large datasets: probs/<m>_full_<Name>-H.json).

Usage: python cp_adaptive.py   → results/cp_adaptive.{json,md}
"""
import json
import math
import os

import numpy as np

LLMS = ["mistral", "eurollm", "llama", "deepseek"]
SETTINGS = ["LDOS-H", "Depaul-H", "Restaurant-H", "MovieLens-H", "AmazonBooks-H"]
EPS = [0.10, 0.05]
K = np.arange(1, 6)


def q(scores, eps):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - eps))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def quantile_of(P, a):
    """a-quantile of each categorical distribution over 1..5, linear interpolation of the CDF on [0.5, 5.5]."""
    cdf = np.cumsum(P, 1)
    xs = np.r_[0.5, K + 0.5]
    out = []
    for c in cdf:
        out.append(np.interp(a, np.r_[0, c], xs))
    return np.clip(np.array(out), 1, 5)


def aps_sets(P):
    """For each distribution: the order in which ratings enter the contiguous set grown from the mode."""
    orders = []
    for p in P:
        lo = hi = int(np.argmax(p))
        order = [lo]
        while len(order) < 5:
            left = p[lo - 1] if lo > 0 else -1
            right = p[hi + 1] if hi < 4 else -1
            if left >= right:
                lo -= 1
                order.append(lo)
            else:
                hi += 1
                order.append(hi)
        orders.append(order)
    return np.array(orders)


def load(m, s):
    name = s.replace("-H", "")
    if name in ("MovieLens", "AmazonBooks"):
        L = json.load(open(f"probs/{m}_full_{s}.json"))
    else:
        L = json.load(open(f"probs/variants/{m}_v0_{s}_test.json"))
    H = json.load(open(f"probs/hybrid/{m}_hycpr_{s}.json"))
    t = np.array(H["truth"], float)
    assert np.array_equal(t, np.array(L["truth"], float))
    return t, np.array(H["expected"], float), np.array(L["probs"], float), np.array(L["expected"], float)


def evaluate(t, scores, sets_fn, eps, rng):
    """Generic split CP: scores (n,), sets_fn(tau, idx) -> boolean (len(idx), 5) admissible ratings."""
    n = len(t)
    ncal = int(0.3 * n)
    cov = size = low = 0.0
    lowmask = t <= 2
    nl = 0
    for _ in range(1000):
        perm = rng.permutation(n)
        cal, te = perm[:ncal], perm[ncal:]
        tau = q(scores[cal], eps)
        S = sets_fn(tau, te)
        hit = S[np.arange(len(te)), (t[te] - 1).astype(int)]
        cov += hit.mean()
        size += S.sum(1).mean()
        if lowmask[te].any():
            low += hit[lowmask[te]].mean()
            nl += 1
    return {"coverage": 100 * cov / 1000, "set_size": size / 1000, "low_coverage": 100 * low / max(nl, 1)}


def main():
    out, md = {}, ["# Ordinal and locally adaptive CP (split CP, 1,000 splits): coverage % / admissible ratings / coverage of ratings 1-2 %\n",
                   "| LLM | setting | eps | abs (HyCPR) | normalized (HyCPR) | CQR-style (HyCPR) | ordinal APS (LLM alone) | abs (LLM alone) |",
                   "|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        for m in LLMS:
            try:
                t, hy, P, mu = load(m, s)
            except FileNotFoundError:
                continue
            sig = np.sqrt((P * (K[None] - mu[:, None]) ** 2).sum(1)) + 0.1
            for eps in EPS:
                rng = lambda: np.random.default_rng(0)
                within = lambda lo, hi, idx: (K[None] >= lo[idx][:, None] - 1e-9) & (K[None] <= hi[idx][:, None] + 1e-9)
                res = {}
                res["abs"] = evaluate(t, np.abs(t - hy), lambda tau, i: within(hy - tau, hy + tau, i), eps, rng())
                res["normalized"] = evaluate(t, np.abs(t - hy) / sig,
                                             lambda tau, i: within(hy - tau * sig, hy + tau * sig, i), eps, rng())
                qlo, qhi = quantile_of(P, eps / 2) + (hy - mu), quantile_of(P, 1 - eps / 2) + (hy - mu)
                res["cqr"] = evaluate(t, np.maximum(qlo - t, t - qhi), lambda tau, i: within(qlo - tau, qhi + tau, i),
                                      eps, rng())
                order = aps_sets(P)
                rank = np.argsort(order, 1)  # position of each rating in the growth order
                Psorted = np.take_along_axis(P, order, 1).cumsum(1)
                aps_score = Psorted[np.arange(len(t)), rank[np.arange(len(t)), (t - 1).astype(int)]]
                res["ordinal_aps"] = evaluate(t, aps_score, lambda tau, i: np.take_along_axis(
                    Psorted[i] <= tau + 1e-12, rank[i], 1) | (rank[i] == 0), eps, rng())
                res["abs_llm"] = evaluate(t, np.abs(t - mu), lambda tau, i: within(mu - tau, mu + tau, i), eps, rng())
                out[f"{m} {s} {eps}"] = res
                f = lambda r: f"{r['coverage']:.1f} / {r['set_size']:.2f} / {r['low_coverage']:.0f}"
                md.append(f"| {m} | {s} | {eps} | " + " | ".join(f(res[k]) for k in ("abs", "normalized", "cqr", "ordinal_aps", "abs_llm")) + " |")
    json.dump(out, open("results/cp_adaptive.json", "w"), indent=1)
    open("results/cp_adaptive.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
