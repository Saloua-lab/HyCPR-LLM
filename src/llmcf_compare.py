"""Paired comparison of the input-level LLM-CF methods adapted to rating prediction (Mistral-7B, -H, MovieLens-1M and
Amazon-Books): CoLLM (collm.py --official), LLaRA (llara_rating.py --lr 2e-4) and A-LLMRec (allmrec_rating.py),
against CPR-LLM and HyCPR-LLM (Table tab:baselines-user-rating and paragraph "Input-level versus prediction-level
fusion" of the paper).

Paired bootstrap (holm.test, 10,000 replicates; AUC 2,000), diff = metric(method) - metric(reference); Holm correction
over the 18 comparisons with HyCPR-LLM (3 methods x 3 metrics x 2 datasets).
Usage: python llmcf_compare.py → results/llmcf.json
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from holm import holm, test

METHODS = {"CoLLM": "collm_official", "LLaRA": "llara_lr0.0002", "A-LLMRec": "allmrec"}
out, fam = {}, []
for name in ("MovieLens", "AmazonBooks"):
    L = json.load(open(f"probs/mistral_full_{name}-H.json"))
    H = json.load(open(f"probs/hybrid/mistral_hycpr_{name}-H.json"))
    t = np.array(L["truth"], float)
    refs = {"CPR-LLM": np.array(L["expected"], float), "HyCPR-LLM": np.array(H["expected"], float)}
    rec = {}
    for k, v in refs.items():
        rec[k] = {"rmse": float(np.sqrt(np.mean((v - t) ** 2))), "mae": float(np.mean(np.abs(v - t))),
                  "auc": float(roc_auc_score(t >= 3, v))}
    for meth, tag in METHODS.items():
        X = json.load(open(f"probs/{tag}_mistral_{name}-H.json"))
        assert np.array_equal(t, np.array(X["truth"], float)), tag
        c = np.array(X["expected"], float)
        r = {"rmse": float(np.sqrt(np.mean((c - t) ** 2))), "mae": float(np.mean(np.abs(c - t))),
             "auc": float(roc_auc_score(t >= 3, c))}
        for rk, o in refs.items():
            r["vs " + rk] = {m: test(c, o, t, m) for m in ("rmse", "mae", "auc")}
        for m, v in r["vs HyCPR-LLM"].items():
            fam.append({"dataset": name, "method": meth, "metric": m, **v})
        rec[meth] = r
    out[name] = rec
holm(fam)
out["holm_vs_HyCPR-LLM"] = fam
json.dump(out, open("results/llmcf.json", "w"), indent=1)
for name in ("MovieLens", "AmazonBooks"):
    print(f"== {name}")
    for k, v in out[name].items():
        print(f"  {k:10s} RMSE {v['rmse']:.4f} MAE {v['mae']:.4f} AUC {v['auc']:.4f}")
print("vs HyCPR-LLM (Holm over 18 tests):")
for x in fam:
    print(f"  {x['dataset']:12s} {x['method']:9s} {x['metric']:4s} diff {x['diff']:+.4f} "
          f"[{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] p {x['p']:.4f} Holm {x['p_holm']:.4f}")
