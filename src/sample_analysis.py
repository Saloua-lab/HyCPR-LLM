"""Analysis of the sampling-based (black-box) predictions of sample_eval.py (review 2, point 2).

For each setting (Mistral): MAE / RMSE of the logit-based expected rating (probs/...) and of the mean of the first n
sampled ratings (n = 1, 3, 5, 10, 20), paired bootstrap of the n = 20 mean minus the logit-based prediction. On
MovieLens-1M and Amazon-Books, the fusion is re-run without any logit: the CF model and w are chosen on the sampled
validation predictions (n = 20) exactly as in large_baselines.hybrid, and the fused test prediction is compared with
the logit-based HyCPR-LLM. Split CP width at eps = 0.1 for the logit-based and the sampled (n = 20) LLM predictions.

Usage: python sample_analysis.py   → results/sampling.{json,md}
"""
import json
import os
import random

import numpy as np

from bootstrap import mae_rmse
from summarize import split_cp

SETTINGS = {"LDOS-H": "probs/mistral_clean_LDOS-H.json", "Depaul-H": "probs/mistral_clean_Depaul-H.json",
            "Restaurant-H": "probs/mistral_clean_Restaurant-H.json", "MovieLens-H": "probs/mistral_full_MovieLens-H.json",
            "AmazonBooks-H": "probs/mistral_full_AmazonBooks-H.json"}
NS = [1, 3, 5, 10, 20]
W = np.round(np.arange(0, 1.0001, 0.05), 2)


def sampled_mean(samples, n, fallback):
    out, empty = [], 0
    for row in samples:
        v = [x for x in row if x is not None][:n]
        if not v:
            empty += 1
        out.append(np.mean(v) if v else fallback)
    return np.array(out), empty


def paired(a, b, t, B=10000):
    idx = np.random.default_rng(0).integers(0, len(t), (B, len(t)))
    res = {}
    for k, name in enumerate(("mae", "rmse")):
        d = mae_rmse((a - t)[idx])[k] - mae_rmse((b - t)[idx])[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[name] = [float(mae_rmse(a - t)[k] - mae_rmse(b - t)[k]), float(lo), float(hi), bool(lo > 0 or hi < 0)]
    return res


def main():
    out, md = {}, ["# Sampling-based (black-box) vs logit-based rating prediction (Mistral)\n",
                   "| setting | logit MAE/RMSE | " + " | ".join(f"n={n}" for n in NS) + " | n=20 − logit: MAE, RMSE | valid samples | CP width 90%: logit / n=20 |",
                   "|---|---|" + "---|" * len(NS) + "---|---|---|"]
    for s, lp in SETTINGS.items():
        sp = f"probs/sampled/{s}.json"
        if not os.path.exists(sp):
            continue
        S, L = json.load(open(sp)), json.load(open(lp))
        t = np.array(S["truth"], float)
        assert np.array_equal(t, np.array(L["truth"], float)), s
        logit = np.array(L["expected"], float)
        fb = float(np.mean(t))  # only used if a prompt has no valid sample (counted)
        rec = {"logit": dict(zip(("mae", "rmse"), map(float, mae_rmse(logit - t))))}
        cells = []
        for n in NS:
            p, empty = sampled_mean(S["samples"], n, fb)
            rec[f"n{n}"] = dict(zip(("mae", "rmse"), map(float, mae_rmse(p - t))), empty=empty)
            cells.append(f"{rec[f'n{n}']['mae']:.3f}/{rec[f'n{n}']['rmse']:.3f}")
        p20, _ = sampled_mean(S["samples"], 20, fb)
        rec["n20_vs_logit"] = paired(p20, logit, t)
        valid = sum(x is not None for r in S["samples"] for x in r) / (len(S["samples"]) * S["n"])
        wl, _ = split_cp(list(zip(t, logit)), 0.1, (1, 5), random.Random(0))
        ws, _ = split_cp(list(zip(t, p20)), 0.1, (1, 5), random.Random(0))
        rec.update(valid=valid, cp_width_logit=wl, cp_width_n20=ws)
        name = s.replace("-H", "")
        if name in ("MovieLens", "AmazonBooks") and os.path.exists(f"probs/sampled/{s}_val.json"):
            b = json.load(open(f"results/large_baselines_{name}.json"))
            V = json.load(open(f"probs/sampled/{s}_val.json"))
            tv = np.array(b["val_truth"], float)
            assert np.array_equal(tv, np.array(V["truth"], float))
            pv, _ = sampled_mean(V["samples"], 20, fb)
            rm = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
            best = None
            for m in b["models"]:
                bv = np.array(b["val_models"][m])
                w = min(W, key=lambda w: rm(w * pv + (1 - w) * bv, tv))
                r = rm(w * pv + (1 - w) * bv, tv)
                if best is None or r < best[0]:
                    best = (r, m, float(w))
            _, m, w = best
            fused = w * p20 + (1 - w) * np.array(b["models"][m])
            hy = np.array(json.load(open(f"probs/hybrid/mistral_hycpr_{s}.json"))["expected"], float)
            rec["fusion_blackbox"] = {"cf": m, "w": w, **dict(zip(("mae", "rmse"), map(float, mae_rmse(fused - t)))),
                                      "vs_logit_hycpr": paired(fused, hy, t),
                                      "logit_hycpr": dict(zip(("mae", "rmse"), map(float, mae_rmse(hy - t))))}
        out[s] = rec
        d = rec["n20_vs_logit"]
        f = lambda v: f"{v[0]:+.3f}{'*' if v[3] else ''}"
        md.append(f"| {s} | {rec['logit']['mae']:.3f}/{rec['logit']['rmse']:.3f} | " + " | ".join(cells)
                  + f" | {f(d['mae'])}, {f(d['rmse'])} | {valid:.1%} | {wl:.2f} / {ws:.2f} |")
    for s, rec in out.items():
        if "fusion_blackbox" in rec:
            fb_ = rec["fusion_blackbox"]
            d = fb_["vs_logit_hycpr"]
            md.append(f"\n{s}: black-box fusion with {fb_['cf']} (w={fb_['w']}): {fb_['mae']:.4f}/{fb_['rmse']:.4f} vs "
                      f"logit HyCPR-LLM {fb_['logit_hycpr']['mae']:.4f}/{fb_['logit_hycpr']['rmse']:.4f}; diff MAE "
                      f"{d['mae'][0]:+.4f} [{d['mae'][1]:+.3f},{d['mae'][2]:+.3f}], RMSE {d['rmse'][0]:+.4f} "
                      f"[{d['rmse'][1]:+.3f},{d['rmse'][2]:+.3f}]")
    json.dump(out, open("results/sampling.json", "w"), indent=1)
    open("results/sampling.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
