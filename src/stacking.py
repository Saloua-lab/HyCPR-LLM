"""Learned fusion (stacking) vs the convex fusion of HyCPR-LLM (second review: "fusion is limited to a single convex
weight; stronger stacking (meta-learner, non-linear fusion) deserves comparison").

Same data as the convex fusion: the meta-learners are fitted on the validation set only (LLM predictions and the
predictions of every CF model of the pool fitted without the validation part), and applied on the test set to the
LLM predictions and to the CF models refitted on the whole training data (as for the convex fusion).

  convex   w * LLM + (1 - w) * b, b and w chosen on validation RMSE (the HyCPR-LLM of the paper)
  NNLS     non-negative least squares on [LLM, all CF models] + intercept (Breiman's stacked regression)
  ridge    ridge regression on [LLM, all CF models], alpha chosen by 5-fold CV on the validation set
  GBM      gradient-boosted trees (sklearn HistGradientBoostingRegressor) on [LLM expected rating, LLM probabilities
           of ratings 1..5, all CF models]; depth, learning rate and number of iterations chosen by 5-fold CV
All predictions are clipped to [1, 5]. Paired bootstrap (10,000) of each meta-learner minus the convex fusion.

Usage: python stacking.py   → results/stacking.{json,md}
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap import collect, mae_rmse  # noqa: E402
from hybrid import fit_baselines  # noqa: E402

LLMS = ["mistral", "eurollm", "llama", "deepseek"]
SMALL = ["LDOS-WH", "Depaul-WH", "Tijuana-WH", "LDOS-H", "Depaul-H", "Restaurant-H"]
LARGE = ["MovieLens", "AmazonBooks"]
B = 10000


def stackers(Xv, yv, Xt, Pv, Pt):
    """Xv/Xt: [LLM, CF models] (n, k); Pv/Pt: LLM probabilities (n, 5). Returns {name: test predictions}."""
    out = {}
    A = np.c_[np.ones(len(Xv)), Xv]
    coef, _ = nnls(A, yv)
    out["NNLS"] = np.c_[np.ones(len(Xt)), Xt] @ coef
    r = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5).fit(Xv, yv)
    out["ridge"] = r.predict(Xt)
    grid = {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "max_iter": [50, 150]}
    g = GridSearchCV(HistGradientBoostingRegressor(random_state=0), grid, cv=KFold(5, shuffle=True, random_state=0),
                     scoring="neg_mean_squared_error").fit(np.c_[Xv, Pv], yv)
    out["GBM"] = g.predict(np.c_[Xt, Pt])
    return {k: np.clip(v, 1, 5) for k, v in out.items()}


def paired(a, b, t):
    idx = np.random.default_rng(0).integers(0, len(t), (B, len(t)))
    res = {}
    for k, name in enumerate(("mae", "rmse")):
        d = mae_rmse((a - t)[idx])[k] - mae_rmse((b - t)[idx])[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[name] = [float(mae_rmse(a - t)[k] - mae_rmse(b - t)[k]), float(lo), float(hi), bool(lo > 0 or hi < 0)]
    return res


def main():
    cols = collect()
    report, cache = {}, {}
    for s in SMALL:
        key = s.replace("-WH", "").replace("-H", "").replace("Restaurant", "Tijuana")
        for m in LLMS:
            ipath = f"data/variants/{m}_{s}_interactions.parquet"
            hp = f"probs/hybrid/{m}_hycpr_{s}.json"
            if not (os.path.exists(ipath) and os.path.exists(hp)):
                continue
            df = pd.read_parquet(ipath)
            if key not in cache:
                cache[key] = fit_baselines(df, key, neural=True)
            base = cache[key]
            lv, lt = (json.load(open(f"probs/variants/{m}_v0_{s}_{sp}.json")) for sp in ("validation", "test"))
            tv, tt = np.array(lv["truth"], float), np.array(lt["truth"], float)
            col = cols[s]
            assert np.array_equal(col["truth"], tt)
            names = sorted(base["validation"])
            Xv = np.c_[lv["expected"], np.column_stack([base["validation"][b] for b in names])]
            Xt = np.c_[lt["expected"], np.column_stack([col["models"][b] for b in names])]
            Pv, Pt = np.array(lv["probs"], float), np.array(lt["probs"], float)
            preds = stackers(Xv, tv, Xt, Pv, Pt)
            conv = np.array(json.load(open(hp))["expected"], float)
            report[f"{m} {s}"] = rec(conv, preds, tt, len(tv))
    for name in LARGE:
        b = json.load(open(f"results/large_baselines_{name}.json"))
        t, tv = np.array(b["truth"], float), np.array(b["val_truth"], float)
        names = sorted(k for k in b["models"] if not k.endswith("-50k"))
        for m in LLMS:
            lt = json.load(open(f"probs/{m}_full_{name}-H.json"))
            lv = json.load(open(f"probs/{m}_full_{name}-H_val.json"))
            assert np.array_equal(np.array(lv["truth"], float), tv)
            Xv = np.c_[lv["expected"], np.column_stack([b["val_models"][k] for k in names])]
            Xt = np.c_[lt["expected"], np.column_stack([b["models"][k] for k in names])]
            preds = stackers(Xv, tv, Xt, np.array(lv["probs"], float), np.array(lt["probs"], float))
            conv = np.array(json.load(open(f"probs/hybrid/{m}_hycpr_{name}-H.json"))["expected"], float)
            report[f"{m} {name}-H"] = rec(conv, preds, t, len(tv))
    json.dump(report, open("results/stacking.json", "w"), indent=1)
    lines = ["| LLM setting | n val | convex MAE/RMSE | " + " | ".join(f"{k} MAE/RMSE (vs convex)" for k in ("NNLS", "ridge", "GBM")) + " |",
             "|---|---|---|---|---|---|"]
    for k, r in report.items():
        c = r["convex"]
        cells = []
        for st in ("NNLS", "ridge", "GBM"):
            x = r[st]
            sig = lambda v: f"{v[0]:+.3f}{'*' if v[3] else ''}"
            cells.append(f"{x['mae']:.3f}/{x['rmse']:.3f} ({sig(x['vs_convex']['mae'])}, {sig(x['vs_convex']['rmse'])})")
        lines.append(f"| {k} | {r['n_val']} | {c['mae']:.3f}/{c['rmse']:.3f} | " + " | ".join(cells) + " |")
    open("results/stacking.md", "w").write("# Stacking vs convex fusion (test; * = paired bootstrap 95% CI excludes 0)\n\n" + "\n".join(lines))
    print("\n".join(lines))


def rec(conv, preds, t, nv):
    r = {"n_val": nv, "convex": dict(zip(("mae", "rmse"), map(float, mae_rmse(conv - t))))}
    for k, p in preds.items():
        r[k] = dict(zip(("mae", "rmse"), map(float, mae_rmse(p - t))), vs_convex=paired(p, conv, t))
    return r


if __name__ == "__main__":
    main()
