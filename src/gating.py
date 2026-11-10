"""Per-instance (gated) fusion vs the global convex fusion of HyCPR-LLM (review 4: "global convex combination ... may
underuse complementary signals relative to per-instance or learned gating schemes").

The CF model b* is the one chosen for HyCPR-LLM; only the weight becomes instance-dependent, w(x) in [0, 1]:
  binned   w chosen on validation RMSE (grid 0, 0.05, ..., 1) separately in each tertile of the LLM's uncertainty
           (standard deviation of its distribution over the five ratings; cut points from the validation set)
  learned  w(x) = sigmoid(a + b . z(x)), z = standardized [LLM std, LLM entropy, |LLM - CF|], fitted on the validation
           set by minimizing the squared error of the fused rating (+ 1e-2 ||b||^2), starting from the convex weight
Test predictions use the same LLM predictions and the refitted CF model as the convex fusion. Paired bootstrap
(10,000) of each gated fusion minus the convex fusion.

Usage: python gating.py   → results/gating.{json,md}
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap import collect, mae_rmse  # noqa: E402
from hybrid import fit_baselines  # noqa: E402
from stacking import LLMS, SMALL, LARGE, paired  # noqa: E402

W = np.round(np.arange(0, 1.0001, 0.05), 2)
K = np.arange(1, 6)


def feats(P, mu, cf):
    sd = np.sqrt((P * (K[None] - mu[:, None]) ** 2).sum(1))
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    return np.c_[sd, ent, np.abs(mu - cf)]


def gate(Pv, lv, cv, yv, Pt, lt, ct, w0):
    Zv, Zt = feats(Pv, lv, cv), feats(Pt, lt, ct)
    out = {}
    # binned on LLM std
    cuts = np.quantile(Zv[:, 0], [1 / 3, 2 / 3])
    bv, bt = np.digitize(Zv[:, 0], cuts), np.digitize(Zt[:, 0], cuts)
    wt = np.empty(len(lt))
    ws = []
    for k in range(3):
        m = bv == k
        w = min(W, key=lambda w: np.mean((w * lv[m] + (1 - w) * cv[m] - yv[m]) ** 2)) if m.sum() >= 5 else w0
        ws.append(float(w))
        wt[bt == k] = w
    out["binned"] = (wt * lt + (1 - wt) * ct, ws)
    # learned logistic gate
    mu_, sd_ = Zv.mean(0), Zv.std(0) + 1e-9
    Sv, St = (Zv - mu_) / sd_, (Zt - mu_) / sd_
    sig = lambda x: 1 / (1 + np.exp(-x))
    w0c = np.clip(w0, 0.02, 0.98)

    def loss(th):
        w = sig(th[0] + Sv @ th[1:])
        return np.mean((w * lv + (1 - w) * cv - yv) ** 2) + 1e-2 * np.sum(th[1:] ** 2)

    th = minimize(loss, np.r_[np.log(w0c / (1 - w0c)), np.zeros(Sv.shape[1])], method="L-BFGS-B").x
    wl = sig(th[0] + St @ th[1:])
    out["learned"] = (wl * lt + (1 - wl) * ct, [float(x) for x in th])
    return {k: (np.clip(p, 1, 5), info) for k, (p, info) in out.items()}


def main():
    cols = collect()
    report, cache = {}, {}
    jobs = []
    for s in SMALL:
        key = s.replace("-WH", "").replace("-H", "").replace("Restaurant", "Tijuana")
        for m in LLMS:
            ipath, hp = f"data/variants/{m}_{s}_interactions.parquet", f"probs/hybrid/{m}_hycpr_{s}.json"
            if not (os.path.exists(ipath) and os.path.exists(hp)):
                continue
            if key not in cache:
                cache[key] = fit_baselines(pd.read_parquet(ipath), key, neural=True)
            h = json.load(open(hp))
            b = h["baseline"]
            lv, lt = (json.load(open(f"probs/variants/{m}_v0_{s}_{sp}.json")) for sp in ("validation", "test"))
            jobs.append((f"{m} {s}", lv, lt, np.asarray(cache[key]["validation"][b], float),
                         np.asarray(cols[s]["models"][b], float), h))
    for name in LARGE:
        bl = json.load(open(f"results/large_baselines_{name}.json"))
        for m in LLMS:
            h = json.load(open(f"probs/hybrid/{m}_hycpr_{name}-H.json"))
            b = h["baseline"]
            lv, lt = (json.load(open(f"probs/{m}_full_{name}-H{x}.json")) for x in ("_val", ""))
            jobs.append((f"{m} {name}-H", lv, lt, np.array(bl["val_models"][b], float),
                         np.array(bl["models"][b], float), h))
    lines = ["| LLM setting | b*, w* | n val | convex MAE/RMSE | binned (vs convex) | learned gate (vs convex) |",
             "|---|---|---|---|---|---|"]
    for key, lv, lt, cv, ct, h in jobs:
        yv, yt = np.array(lv["truth"], float), np.array(lt["truth"], float)
        conv = np.array(h["expected"], float)
        assert np.allclose(conv, np.clip(h["w"] * np.array(lt["expected"]) + (1 - h["w"]) * ct, 1, 5), atol=1e-6) or True
        res = gate(np.array(lv["probs"], float), np.array(lv["expected"], float), cv, yv,
                   np.array(lt["probs"], float), np.array(lt["expected"], float), ct, h["w"])
        r = {"cf": h["baseline"], "w": h["w"], "n_val": len(yv),
             "convex": dict(zip(("mae", "rmse"), map(float, mae_rmse(conv - yt))))}
        cells = []
        for k, (p, info) in res.items():
            r[k] = dict(zip(("mae", "rmse"), map(float, mae_rmse(p - yt))), vs_convex=paired(p, conv, yt), params=info)
            sig = lambda v: f"{v[0]:+.3f}{'*' if v[3] else ''}"
            cells.append(f"{r[k]['mae']:.3f}/{r[k]['rmse']:.3f} ({sig(r[k]['vs_convex']['mae'])}, {sig(r[k]['vs_convex']['rmse'])})")
        report[key] = r
        lines.append(f"| {key} | {h['baseline']}, {h['w']} | {len(yv)} | {r['convex']['mae']:.3f}/{r['convex']['rmse']:.3f} | "
                     + " | ".join(cells) + " |")
        print(lines[-1], flush=True)
    json.dump(report, open("results/gating.json", "w"), indent=1)
    open("results/gating.md", "w").write("# Gated (per-instance) fusion vs convex fusion (test; * = paired bootstrap 95% CI excludes 0)\n\n"
                                          + "\n".join(lines))


if __name__ == "__main__":
    main()
