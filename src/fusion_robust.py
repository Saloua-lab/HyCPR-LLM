"""Robustness of the fusion weight (review 6: "it is unclear how robust weight selection is across domains").

Same validation / test predictions as bma.py (LLM and every CF model of the pool). For each LLM x -H setting:
  * the selection of HyCPR-LLM (CF model b* and w on the grid {0, 0.05, ..., 1} minimizing the validation RMSE) is
    redone and checked against probs/hybrid;
  * oracle: the w minimizing the TEST RMSE for the same b*, and the regret RMSE_test(w_val) - RMSE_test(w_oracle);
  * flatness: the largest increase of the test RMSE for |w - w_val| <= 0.1;
  * stability: 1,000 bootstrap resamples of the validation set, the selection (b and w) redone on each; reported:
    share of resamples selecting the same b*, 5-95% range of w, and the mean and 95th percentile of the test RMSE of the
    resampled selections minus that of HyCPR-LLM.

Usage: python fusion_robust.py → results/fusion_robust.{json,md}
"""
import json
import os

import numpy as np
import pandas as pd

from bootstrap import collect
from hybrid import fit_baselines
from stacking import LARGE, LLMS, SMALL

W = np.round(np.arange(0, 1.0001, 0.05), 2)


def rmse(p, y):
    return np.sqrt(((p - y[..., None, :]) ** 2).mean(-1)) if p.ndim > y.ndim else np.sqrt(((p - y) ** 2).mean(-1))


def select(Lv, Bv, yv):
    """Lv (n,), Bv (m, n), yv (n,) -> (index of b, w) minimizing the validation RMSE over the grid."""
    F = W[None, :, None] * Lv[None, None] + (1 - W[None, :, None]) * Bv[:, None]      # (m, |W|, n)
    R = np.sqrt(((F - yv) ** 2).mean(-1))
    j, k = np.unravel_index(np.argmin(R), R.shape)
    return j, W[k]


def jobs():
    cols, cache = collect(), {}
    for s in SMALL:
        if not s.endswith("-H"):
            continue
        key = s.replace("-H", "").replace("Restaurant", "Tijuana")
        for m in LLMS:
            ipath, hp = f"data/variants/{m}_{s}_interactions.parquet", f"probs/hybrid/{m}_hycpr_{s}.json"
            if not (os.path.exists(ipath) and os.path.exists(hp)):
                continue
            if key not in cache:
                cache[key] = fit_baselines(pd.read_parquet(ipath), key, neural=True)
            base = cache[key]
            lv, lt = (json.load(open(f"probs/variants/{m}_v0_{s}_{sp}.json")) for sp in ("validation", "test"))
            names = sorted(base["validation"])
            yield (m, s, names, np.array(lv["truth"], float), np.array(lt["truth"], float), np.array(lv["expected"], float),
                   np.array(lt["expected"], float), np.array([base["validation"][b] for b in names], float),
                   np.array([cols[s]["models"][b] for b in names], float), hp)
    for name in LARGE:
        b = json.load(open(f"results/large_baselines_{name}.json"))
        names = sorted(k for k in b["models"] if not k.endswith("-50k"))
        for m in LLMS:
            lt, lv = json.load(open(f"probs/{m}_full_{name}-H.json")), json.load(open(f"probs/{m}_full_{name}-H_val.json"))
            yield (m, f"{name}-H", names, np.array(b["val_truth"], float), np.array(b["truth"], float),
                   np.array(lv["expected"], float), np.array(lt["expected"], float),
                   np.array([b["val_models"][k] for k in names], float), np.array([b["models"][k] for k in names], float),
                   f"probs/hybrid/{m}_hycpr_{name}-H.json")


def main():
    out, md = {}, ["| LLM | setting | b* | w | oracle w (test) | regret RMSE | max dRMSE, abs(w - w*) <= 0.1 | same b* (%) | w 5-95% | dRMSE bootstrap mean / 95th |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for m, s, names, yv, yt, Lv, Lt, Bv, Bt, hp in jobs():
        h = json.load(open(hp))
        j, w = select(Lv, Bv, yv)
        bstar = names[j]
        ok = bstar == h["baseline"] and abs(w - h["w"]) < 1e-9
        test_curve = np.array([rmse(x * Lt + (1 - x) * Bt[j], yt) for x in W])
        r0 = test_curve[np.argmin(np.abs(W - w))]
        near = np.abs(W - w) <= 0.1 + 1e-9
        same, ws, dr = 0, [], []
        for _ in range(1000):
            i = rng.integers(0, len(yv), len(yv))
            jb, wb = select(Lv[i], Bv[:, i], yv[i])
            same += jb == j
            ws.append(wb)
            dr.append(rmse(wb * Lt + (1 - wb) * Bt[jb], yt) - r0)
        dr = np.array(dr)
        rec = {"b": bstar, "w": float(w), "matches_hycpr": bool(ok), "n_val": len(yv),
               "oracle_w": float(W[np.argmin(test_curve)]), "regret": float(r0 - test_curve.min()),
               "flat": float(test_curve[near].max() - r0), "same_b": same / 10, "w_5_95": [float(np.percentile(ws, 5)), float(np.percentile(ws, 95))],
               "boot_d_mean": float(dr.mean()), "boot_d_95": float(np.percentile(dr, 95)), "test_curve": test_curve.tolist()}
        out[f"{m} {s}"] = rec
        md.append(f"| {m} | {s} | {bstar}{'' if ok else ' (!)'} | {w:.2f} | {rec['oracle_w']:.2f} | {rec['regret']:.4f} | {rec['flat']:.4f} | "
                  f"{rec['same_b']:.0f} | {rec['w_5_95'][0]:.2f}-{rec['w_5_95'][1]:.2f} | {rec['boot_d_mean']:+.4f} / {rec['boot_d_95']:+.4f} |")
        print(md[-1], flush=True)
    json.dump(out, open("results/fusion_robust.json", "w"), indent=1)
    open("results/fusion_robust.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
