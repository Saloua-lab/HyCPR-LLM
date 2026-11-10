"""Expected rating from the token probabilities vs rating parsed from generated text, same adapter and prompts
(review 3: "expected-value decoding vs text parsing not clearly quantified").

Pairs results/<name>.json (greedy generation, rating parsed with the reference regex list; runs.full) with
probs/<name>.json (evaluate_probs.py). Reports the share of prompts with a usable parsed rating, the MAE / RMSE of the
parsed rating on those prompts, and of the expected rating on the same prompts and on all prompts.
Usage: python gen_vs_prob.py   → results/gen_vs_prob.{json,md}
"""
import glob
import json
import os

import numpy as np

rows, out = [], {}
for g in sorted(glob.glob("results/*_*_*.json")):
    name = os.path.basename(g)[:-5]
    p = f"probs/{name}.json"
    if not os.path.exists(p):
        continue
    try:
        r = json.load(open(g))["runs"]["full"]
    except (KeyError, TypeError):
        continue
    P = json.load(open(p))
    tg, tp = r["truth"], P["truth"]
    if len(tg) != len(tp) or any(a != b for a, b in zip(tg, tp) if a is not None):
        continue
    t = np.array(tp, float)
    e = np.array(P["expected"], float)
    pred = r["pred"]
    ok = np.array([x is not None and 1 <= x <= 5 for x in pred])
    pp = np.array([x if x is not None else np.nan for x in pred], float)
    m = lambda a, b: (float(np.abs(a - b).mean()), float(np.sqrt(((a - b) ** 2).mean())))
    rec = {"n": len(t), "usable": float(ok.mean()), "parsed_usable": m(pp[ok], t[ok]) if ok.any() else None,
           "expected_usable": m(e[ok], t[ok]) if ok.any() else None, "expected_all": m(e, t)}
    out[name] = rec
    rows.append(rec)
json.dump(out, open("results/gen_vs_prob.json", "w"), indent=1)
lines = ["| run | n | usable parsed | parsed MAE/RMSE (usable) | expected MAE/RMSE (same prompts) | expected (all prompts) |", "|---|---|---|---|---|---|"]
for k, r in out.items():
    pu, eu, ea = r["parsed_usable"], r["expected_usable"], r["expected_all"]
    lines.append(f"| {k} | {r['n']} | {100*r['usable']:.0f}% | " + (f"{pu[0]:.3f}/{pu[1]:.3f} | {eu[0]:.3f}/{eu[1]:.3f}" if pu else "– | –") + f" | {ea[0]:.3f}/{ea[1]:.3f} |")
open("results/gen_vs_prob.md", "w").write("\n".join(lines))
print("\n".join(lines))
