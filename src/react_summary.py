"""Summary of the ReAct-with-tools experiment (react_tools.py): coverage, MAE / RMSE with 95% bootstrap CIs,
paired differences with the full tool set, and tool usage.

Metrics are computed on every test interaction: an episode without a usable rating gets the global mean rating
of the training data (counted in "usable"), so that all configurations are compared on the same interactions.

Usage: python react_summary.py   → results/react_summary.{md,json}, results/react_table.tex
"""
import json
import os

import numpy as np

DS = ["LDOS", "Depaul", "Tijuana"]
CONFIGS = [("direct", "Direct prompt (no ReAct)"), ("direct_tools", "Direct prompt + the four tool outputs"), ("react_notools", "ReAct, self-generated observations"),
           ("react_all", "ReAct with the four tools"), ("react_minus_user", "\\quad without user\\_ratings"),
           ("react_minus_item", "\\quad without item\\_ratings"), ("react_minus_similar", "\\quad without similar\\_items"),
           ("react_minus_context", "\\quad without context\\_ratings")]
B = 10000


def load(ds, c):
    p = f"results/react/{ds}_{c}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    t = np.array(d["truth"], float)
    raw = d["pred"]
    pred = np.array([d["global_mean"] if x is None else x for x in raw], float)
    return {"t": t, "p": pred, "usable": sum(x is not None for x in raw), "n": len(raw), "steps": d["steps"],
            "calls": d["calls"]}


def mm(e):
    return np.abs(e).mean(-1), np.sqrt((e ** 2).mean(-1))


out, md, tex = {}, [], {}
for ds in DS:
    runs = {c: load(ds, c) for c, _ in CONFIGS}
    runs = {c: r for c, r in runs.items() if r}
    if not runs:
        continue
    t = next(iter(runs.values()))["t"]
    idx = np.random.default_rng(0).integers(0, len(t), (B, len(t)))
    md += [f"## {ds} ({len(t)} test interactions)", "",
           "| config | usable | MAE [CI] | RMSE [CI] | vs react_all: MAE | RMSE | tools/episode |", "|---|---|---|---|---|---|---|"]
    out[ds] = {}
    for c, label in CONFIGS:
        if c not in runs:
            continue
        r = runs[c]
        assert np.array_equal(r["t"], t)
        e = r["p"] - t
        (mae, rmse), (bm, br) = mm(e), mm(e[idx])
        rec = {"usable": r["usable"], "n": r["n"], "mae": float(mae), "rmse": float(rmse),
               "mae_ci": np.percentile(bm, [2.5, 97.5]).tolist(), "rmse_ci": np.percentile(br, [2.5, 97.5]).tolist()}
        calls = {}
        for cc in r["calls"]:
            for k, v in cc.items():
                calls[k] = calls.get(k, 0) + v
        real = ("user_ratings", "item_ratings", "similar_items", "context_ratings")
        grouped = {k: calls.get(k, 0) for k in real if k in calls}
        grouped["invented"] = sum(v for k, v in calls.items() if k not in real and k != "Finish")
        rec["calls_per_episode"] = {k: v / r["n"] for k, v in grouped.items() if v or k != "invented"}
        rec["episodes_with_invented_tool"] = sum(any(k not in real and k != "Finish" for k in cc) for cc in r["calls"]) / r["n"]
        rec["mean_steps"] = float(np.mean(r["steps"]))
        diff = ""
        if c != "react_all" and "react_all" in runs:
            ea = runs["react_all"]["p"] - t
            d = {}
            for k, name in enumerate(("mae", "rmse")):
                dd = mm(e[idx])[k] - mm(ea[idx])[k]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                d[name] = [float(mm(e)[k] - mm(ea)[k]), float(lo), float(hi), bool(lo > 0 or hi < 0)]
            rec["vs_react_all"] = d
            diff = " | ".join(f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]{' *' if v[3] else ''}" for v in d.values())
        else:
            diff = " | "
        tools = ", ".join(f"{k} {v:.2f}" for k, v in rec["calls_per_episode"].items())
        if c.startswith("react_") and c != "react_notools":
            tools += f"; episodes with an invented tool {rec['episodes_with_invented_tool']:.0%}"
        md.append(f"| {c} | {r['usable']}/{r['n']} | {mae:.3f} [{rec['mae_ci'][0]:.3f}, {rec['mae_ci'][1]:.3f}] | "
                  f"{rmse:.3f} [{rec['rmse_ci'][0]:.3f}, {rec['rmse_ci'][1]:.3f}] | {diff} | {tools} |")
        out[ds][c] = rec
        sig = lambda k: "^{*}" if rec.get("vs_react_all", {}).get(k, [0, 0, 0, False])[3] else ""
        tex.setdefault(c, {"label": label})[ds] = f"{mae:.3f}${sig('mae')}$ / {rmse:.3f}${sig('rmse')}$ ({r['usable']})"
    md.append("")
json.dump(out, open("results/react_summary.json", "w"), indent=1)
open("results/react_summary.md", "w").write("# ReAct with tools (Mistral-7B-Instruct, zero-shot)\n\n" + "\n".join(md))
rows = [f"{v['label']} & " + " & ".join(v.get(ds, "--") for ds in DS) + " \\\\" for c, v in tex.items()]
open("results/react_table.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(md))
