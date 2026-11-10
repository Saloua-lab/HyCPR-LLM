"""Conformal prediction: ordinal view, conditional coverage and group-conditional (Mondrian) CP
(second review: "CP under-specified ... ordinal adjustments for a 1-5 scale"; "no analysis of conditional coverage").

Base procedure (as cp_hybrid.py): score s = |r - r_hat| on the fused HyCPR-LLM rating, split CP with 30% of the
test predictions as calibration set, finite-sample quantile ceil((n+1)(1-eps))/n, 1,000 random splits.

1. Ordinal view. True ratings are integers, so the interval [r_hat - tau, r_hat + tau] is equivalent to the set of
   admissible ratings S = {k in 1..5 : |k - r_hat| <= tau}; coverage is identical, and |S| (number of ratings) is
   reported next to the clipped width.
2. Conditional coverage of the marginal (split) CP within groups of the test interactions:
     user activity   number of training ratings of the user (low / high: below / above the median)
     item popularity number of training ratings of the item (low / high)
     context         each value of the first context variable with >= 30 test interactions
     true rating     low (1-2), mid (3), high (4-5)
3. Mondrian CP: the same, but tau is computed separately within each group of the calibration set
   (group-conditional coverage guarantee, Vovk et al. 2005), for groupings known at test time (user activity,
   item popularity, predicted rating <3 / 3-4 / >=4); the true rating cannot be used as a Mondrian taxonomy because it is unknown at test time.

Usage: python cp_extended.py   → results/cp_extended.{json,md}
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LLMS = ["mistral", "eurollm", "llama", "deepseek"]
SETTINGS = {"LDOS-H": "LDOS", "Depaul-H": "Depaul", "Restaurant-H": "Tijuana"}
EPS = [0.10, 0.05]
N_SPLITS, CAL = 1000, 0.3


def q(scores, eps):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - eps))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def set_size(pred, tau):
    ks = np.arange(1, 6)
    return (np.abs(ks[None, :] - pred[:, None]) <= np.asarray(tau)[:, None] + 1e-12).sum(1)


def width(pred, tau):
    return np.minimum(5, pred + tau) - np.maximum(1, pred - tau)


def groups_of(name):
    """Group labels of the test interactions (in the order of the probs files) from the training data."""
    from baselines_same_split import SMALL, recover_users
    tr, te, _ = recover_users(*SMALL[name])
    tr["u"], te["u"] = tr.u.astype(str), te.u.astype(str)
    ucount = te.u.map(tr.u.value_counts()).fillna(0).to_numpy()
    icount = te.i.map(tr.i.value_counts()).fillna(0).to_numpy()
    ctx = sorted(c for c in te.columns if c.startswith("c_"))
    c0 = te[ctx[0]].fillna("NA").astype(str).to_numpy() if ctx else np.array(["all"] * len(te))
    r = te.r.astype(float).to_numpy()
    lab = lambda x: np.where(x <= np.median(x), "low", "high")
    return {"user activity": lab(ucount), "item popularity": lab(icount), f"context ({ctx[0][2:]})": c0,
            "true rating": np.where(r <= 2, "1-2", np.where(r == 3, "3", "4-5"))}, r


def run(truth, pred, groups, eps, mondrian_key=None):
    n = len(truth)
    s = np.abs(truth - pred)
    rng = np.random.default_rng(0)
    ncal = int(n * CAL)
    cov_tot = wid = size = 0.0
    gcov = {k: {} for k in groups}
    for _ in range(N_SPLITS):
        perm = rng.permutation(n)
        cal, te = perm[:ncal], perm[ncal:]
        if mondrian_key is None:
            tau = np.full(len(te), q(s[cal], eps))
        else:
            g = groups[mondrian_key]
            tau = np.array([q(s[cal][g[cal] == g[j]], eps) for j in te])
        covered = s[te] <= tau + 1e-12
        cov_tot += covered.mean()
        wid += width(pred[te], np.minimum(tau, 4)).mean()
        size += set_size(pred[te], np.minimum(tau, 4)).mean()
        for k, g in groups.items():
            for v in np.unique(g):
                m = g[te] == v
                if m.sum():
                    gcov[k].setdefault(v, []).append(covered[m].mean())
    return {"coverage": 100 * cov_tot / N_SPLITS, "width": wid / N_SPLITS, "set_size": size / N_SPLITS,
            "group_coverage": {k: {v: 100 * float(np.mean(x)) for v, x in d.items()} for k, d in gcov.items()},
            "group_n": {k: {v: int((g == v).sum()) for v in np.unique(g)} for k, g in groups.items()}}


def main():
    out, md = {}, ["# Conformal prediction: ordinal sets, conditional coverage, Mondrian CP (split CP, 1,000 splits)\n"]
    for setting, name in SETTINGS.items():
        groups, r = groups_of(name)
        for m in LLMS:
            p = f"probs/hybrid/{m}_hycpr_{setting}.json"
            if not os.path.exists(p):
                continue
            h = json.load(open(p))
            truth, pred = np.array(h["truth"], float), np.array(h["expected"], float)
            assert np.array_equal(truth, r), p
            groups = dict(groups, **{"predicted rating": np.where(pred < 3, "<3", np.where(pred < 4, "3-4", ">=4"))})
            for eps in EPS:
                key = f"{m} {setting} eps={eps}"
                res = {"marginal": run(truth, pred, groups, eps)}
                for mk in ("user activity", "item popularity", "predicted rating"):
                    res[f"mondrian ({mk})"] = run(truth, pred, groups, eps, mk)
                out[key] = res
                mg = res["marginal"]
                md.append(f"## {key}: coverage {mg['coverage']:.1f}%, width {mg['width']:.2f}, set size {mg['set_size']:.2f} ratings")
                for g, d in mg["group_coverage"].items():
                    md.append(f"- {g}: " + ", ".join(f"{v} {c:.1f}% (n={mg['group_n'][g][v]})" for v, c in d.items()))
                for mk in ("user activity", "item popularity", "predicted rating"):
                    r_ = res[f"mondrian ({mk})"]
                    md.append(f"- Mondrian on {mk}: coverage {r_['coverage']:.1f}%, width {r_['width']:.2f}, set size "
                              f"{r_['set_size']:.2f}; per group: " + ", ".join(
                                  f"{v} {c:.1f}%" for v, c in r_["group_coverage"][mk].items())
                              + "; by true rating: " + ", ".join(f"{v} {c:.1f}%" for v, c in r_["group_coverage"]["true rating"].items()))
                md.append("")
    json.dump(out, open("results/cp_extended.json", "w"), indent=1)
    open("results/cp_extended.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
