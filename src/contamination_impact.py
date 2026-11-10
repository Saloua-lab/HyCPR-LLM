"""Does pretraining memorization explain the results on MovieLens-1M / Amazon-Books?

For each LLM, the test interactions are split by the probes of contamination.py on its own pretrained base model:
  MovieLens  item memorized (title of the MovieID reproduced exactly) vs not; and rating line memorized
             (exact rating AND exact timestamp reproduced) vs not
  Amazon     item memorized (title of the ASIN reproduced, fuzzy match) vs not
In each group: RMSE / MAE of the LLM alone (probs/<m>_full_<Name>-H.json), of HyCPR-LLM
(probs/hybrid/<m>_hycpr_<Name>-H.json) and of the strongest CF baseline (BiasedMF on MovieLens, MLP on Amazon),
and the paired bootstrap 95% CI of HyCPR-LLM minus the baseline.

Usage: python contamination_impact.py   → results/contamination_impact.{md,json}
"""
import json
import os

import numpy as np

from contamination import test_split

LLMS = ["mistral", "eurollm", "llama", "deepseek"]
BASE_OF = {"MovieLens": "BiasedMF", "AmazonBooks": "MLP"}
B = 10000


def boot(a, b, t, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(t), (B, len(t)))
    out = {}
    for name, f in (("rmse", lambda p, j: np.sqrt(((p[j] - t[j]) ** 2).mean(-1))),
                    ("mae", lambda p, j: np.abs(p[j] - t[j]).mean(-1))):
        d = f(a, idx) - f(b, idx)
        lo, hi = np.percentile(d, [2.5, 97.5])
        full = np.arange(len(t))[None]
        out[name] = [float(f(a, full)[0] - f(b, full)[0]), float(lo), float(hi), bool(lo > 0 or hi < 0)]
    return out


def metrics(p, t):
    return float(np.sqrt(((p - t) ** 2).mean())), float(np.abs(p - t).mean())


splits = {"MovieLens": test_split("movielens"), "AmazonBooks": test_split("amazon")}
out, md = {}, ["# Contamination probes and their effect on the results\n"]
md += ["## Probe results (pretrained base models)\n",
       "| LLM | ML items exact | ML items no year | ML rating exact (mode ctrl) | ML timestamp exact (prev-line ctrl) | AZ items exact | AZ items fuzzy |",
       "|---|---|---|---|---|---|---|"]
probes = {}
for m in LLMS:
    p = f"results/contamination/{m}.json"
    if not os.path.exists(p):
        continue
    c = probes[m] = json.load(open(p))
    r = c["ml_ratings"]
    md.append(f"| {m} | {c['ml_items']['exact']:.1%} | {c['ml_items']['title_without_year']:.1%} | "
              f"{r['rating_exact']:.1%} ({r['mode_exact']:.1%}) | {r['ts_exact']:.1%} ({r['prev_ts_exact']:.1%}) | "
              f"{c['az_items']['exact']:.1%} | {c['az_items']['fuzzy']:.1%} |")
md.append("")
for name, (df, te) in splits.items():
    b = json.load(open(f"results/large_baselines_{name}.json"))
    t = np.array(b["truth"], float)
    base = np.array(b["models"][BASE_OF[name]], float)
    md += [f"## {name}: test interactions split by memorization\n",
           f"| LLM | group | n | LLM alone RMSE / MAE | HyCPR-LLM RMSE / MAE | {BASE_OF[name]} RMSE / MAE | HyCPR − {BASE_OF[name]}: RMSE | MAE |",
           "|---|---|---|---|---|---|---|---|"]
    out[name] = {}
    for m in LLMS:
        if m not in probes:
            continue
        llm = np.array(json.load(open(f"probs/{m}_full_{name}-H.json"))["expected"], float)
        hy = json.load(open(f"probs/hybrid/{m}_hycpr_{name}-H.json"))
        assert np.array_equal(np.array(hy["truth"], float), t)
        hy = np.array(hy["expected"], float)
        c = probes[m]
        groupings = []
        if name == "MovieLens":
            items = np.array([c["ml_items"]["flags"][str(int(df.at[i, "item"]))]["exact"] for i in te])
            per = c["ml_ratings"]["per_interaction"]
            line = np.array([p["rating_exact"] and p["ts_exact"] for p in per])
            groupings = [("item memorized", items), ("rating line memorized", line)]
        else:
            fl = c["az_items"]["flags"]
            items = np.array([fl.get(str(df.at[i, "item"]), {"fuzzy": False})["fuzzy"] for i in te])
            groupings = [("item memorized", items)]
        out[name][m] = {}
        for gname, mask in groupings:
            for label, sel in ((f"{gname}: yes", mask), (f"{gname}: no", ~mask)):
                n = int(sel.sum())
                if n < 20:
                    md.append(f"| {m} | {label} | {n} | (too few) | | | | |")
                    continue
                d = boot(hy[sel], base[sel], t[sel])
                rec = {"n": n, "llm": metrics(llm[sel], t[sel]), "hycpr": metrics(hy[sel], t[sel]),
                       "base": metrics(base[sel], t[sel]), "hycpr_minus_base": d}
                out[name][m][label] = rec
                f = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]{' *' if v[3] else ''}"
                md.append(f"| {m} | {label} | {n} | {rec['llm'][0]:.3f} / {rec['llm'][1]:.3f} | "
                          f"{rec['hycpr'][0]:.3f} / {rec['hycpr'][1]:.3f} | {rec['base'][0]:.3f} / {rec['base'][1]:.3f} | "
                          f"{f(d['rmse'])} | {f(d['mae'])} |")
    md.append("")
json.dump({"probes": {m: {k: {kk: vv for kk, vv in v.items() if kk not in ("flags", "per_interaction")}
                          for k, v in c.items() if isinstance(v, dict)} for m, c in probes.items()},
           "impact": out}, open("results/contamination_impact.json", "w"), indent=1)
open("results/contamination_impact.md", "w").write("\n".join(md))
print("\n".join(md))
