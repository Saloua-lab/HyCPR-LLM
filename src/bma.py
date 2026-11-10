"""Bayesian model averaging over the fusion pool vs the convex fusion of HyCPR-LLM (review 5: "user- or
instance-adaptive blending (meta-learning, gating, Bayesian model averaging)").

Same data as stacking.py: validation predictions of the LLM and of every CF model of the pool (LLM fine-tuned on 90% of
the training data and CF models fitted on the same 90% for the small datasets; the last 5% of the training period for
MovieLens-1M and Amazon-Books), test predictions of the LLM and of the refitted CF models. Each member m is a Gaussian
predictive model y ~ N(x_m, sigma_m^2), sigma_m^2 = its validation MSE.
  BMA          posterior weights proportional to the validation likelihood (uniform prior): w_m ∝ exp(sum_i log N(y_i; x_im, sigma_m^2))
  pseudo-BMA+  Yao et al. (2018): weights from the pointwise log-likelihoods, stabilized by the Bayesian bootstrap
               (1,000 Dirichlet(1) draws), w = mean_b softmax_m(n sum_i alpha_bi l_im)
Prediction: sum_m w_m x_m. Paired bootstrap (10,000) against the convex fusion.

Usage: python bma.py → results/bma.{json,md}
"""
import json
import os

import numpy as np
import pandas as pd

from bootstrap import collect, mae_rmse
from hybrid import fit_baselines
from stacking import LARGE, LLMS, SMALL, paired


def weights(Xv, yv):
    s2 = ((Xv - yv[:, None]) ** 2).mean(0)
    L = -0.5 * np.log(2 * np.pi * s2)[None] - (Xv - yv[:, None]) ** 2 / (2 * s2[None])   # (n, m) pointwise
    z = L.sum(0)
    bma = np.exp(z - z.max())
    bma /= bma.sum()
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(len(yv)), 1000)                # (B, n)
    Z = len(yv) * a @ L                                      # (B, m)
    Z = np.exp(Z - Z.max(1, keepdims=True))
    pb = (Z / Z.sum(1, keepdims=True)).mean(0)
    return {"BMA": bma, "pseudo-BMA+": pb}


def main():
    cols = collect()
    report, cache, jobs = {}, {}, []
    for s in SMALL:
        key = s.replace("-WH", "").replace("-H", "").replace("Restaurant", "Tijuana")
        for m in LLMS:
            ipath, hp = f"data/variants/{m}_{s}_interactions.parquet", f"probs/hybrid/{m}_hycpr_{s}.json"
            if not (os.path.exists(ipath) and os.path.exists(hp)):
                continue
            if key not in cache:
                cache[key] = fit_baselines(pd.read_parquet(ipath), key, neural=True)
            base = cache[key]
            lv, lt = (json.load(open(f"probs/variants/{m}_v0_{s}_{sp}.json")) for sp in ("validation", "test"))
            names = sorted(base["validation"])
            jobs.append((f"{m} {s}", ["LLM"] + names, np.array(lv["truth"], float), np.array(lt["truth"], float),
                         np.c_[lv["expected"], np.column_stack([base["validation"][b] for b in names])],
                         np.c_[lt["expected"], np.column_stack([cols[s]["models"][b] for b in names])], hp))
    for name in LARGE:
        b = json.load(open(f"results/large_baselines_{name}.json"))
        names = sorted(k for k in b["models"] if not k.endswith("-50k"))
        for m in LLMS:
            lt, lv = json.load(open(f"probs/{m}_full_{name}-H.json")), json.load(open(f"probs/{m}_full_{name}-H_val.json"))
            jobs.append((f"{m} {name}-H", ["LLM"] + names, np.array(b["val_truth"], float), np.array(b["truth"], float),
                         np.c_[lv["expected"], np.column_stack([b["val_models"][k] for k in names])],
                         np.c_[lt["expected"], np.column_stack([b["models"][k] for k in names])],
                         f"probs/hybrid/{m}_hycpr_{name}-H.json"))
    lines = ["| LLM setting | n val | convex MAE/RMSE | BMA (vs convex) | top BMA weights | pseudo-BMA+ (vs convex) | top pseudo-BMA+ weights |",
             "|---|---|---|---|---|---|---|"]
    for key, names, yv, yt, Xv, Xt, hp in jobs:
        conv = np.array(json.load(open(hp))["expected"], float)
        r = {"n_val": len(yv), "convex": dict(zip(("mae", "rmse"), map(float, mae_rmse(conv - yt))))}
        cells = []
        for k, w in weights(Xv, yv).items():
            p = np.clip(Xt @ w, 1, 5)
            top = sorted(zip(names, w), key=lambda x: -x[1])[:3]
            r[k] = dict(zip(("mae", "rmse"), map(float, mae_rmse(p - yt))), vs_convex=paired(p, conv, yt),
                        weights={n: float(x) for n, x in zip(names, w)})
            sig = lambda v: f"{v[0]:+.3f}{'*' if v[3] else ''}"
            cells += [f"{r[k]['mae']:.3f}/{r[k]['rmse']:.3f} ({sig(r[k]['vs_convex']['mae'])}, {sig(r[k]['vs_convex']['rmse'])})",
                      ", ".join(f"{n} {x:.2f}" for n, x in top)]
        report[key] = r
        lines.append(f"| {key} | {len(yv)} | {r['convex']['mae']:.3f}/{r['convex']['rmse']:.3f} | " + " | ".join(cells) + " |")
        print(lines[-1], flush=True)
    json.dump(report, open("results/bma.json", "w"), indent=1)
    open("results/bma.md", "w").write("# BMA over the fusion pool vs convex fusion (test; * = paired bootstrap 95% CI excludes 0)\n\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
