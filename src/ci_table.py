"""Appendix table of the paper (tab:ci): 95% bootstrap intervals of MAE and RMSE for the CPR-LLM models of Table 4
and the best baseline of each column, from results/bootstrap.json (run bootstrap.py first).

HyCPR-LLM rows from probs/hybrid/<llm>_hycpr_<setting>.json.

Usage: python3 ci_table.py > results/ci_table.tex
"""
import json
import os

import numpy as np

from bootstrap import ci, mae_rmse

from table4 import COLS, NAMES, ROW_ORDER, llm_tag

BASELINES = ["GAR", "IAR", "UAR", "ItemKNN", "UserKNN", "SVD++", "BiasedMF", "FM", "CAMF_CI", "CAMF_CU", "CAMF_C",
             "CAMF_CUCI", "NFM", "DeepFM", "AutoInt"]
d = json.load(open("results/bootstrap.json"))["columns"]
cell = lambda v, m: f"{v[m]:.3f} [{v[m + '_ci'][0]:.3f}, {v[m + '_ci'][1]:.3f}]"
lines = []
for col in COLS:
    models = d[col]["models"]
    rows = [(f"\\llmonly{{}} {NAMES[l]}", models[llm_tag(l, col)]) for l in ROW_ORDER if llm_tag(l, col) in models]
    for l in ROW_ORDER:  # HyCPR-LLM rows: fused predictions of hybrid.py / large_baselines.py
        hp = f"probs/hybrid/{l}_hycpr_{col}.json"
        if os.path.exists(hp):
            h = json.load(open(hp))
            err = np.array(h["expected"]) - np.array(h["truth"], float)
            idx = np.random.default_rng(0).integers(0, len(err), (10000, len(err)))
            bm, br = mae_rmse(err[idx])
            mae, rmse = mae_rmse(err)
            rows.append((f"\\ours{{}} {NAMES[l]}", {"mae": float(mae), "rmse": float(rmse), "mae_ci": ci(bm), "rmse_ci": ci(br)}))
    base = [b for b in BASELINES if b in models]
    best = min(base, key=lambda b: models[b]["mae"])
    rows.append((f"{best.replace('_', chr(92) + '_')} (best baseline)", models[best]))
    for i, (name, v) in enumerate(rows):
        first = f"\\multirow{{{len(rows)}}}{{*}}{{{col}}}" if i == 0 else ""
        lines.append(f"{first} & {name} & {cell(v, 'mae')} & {cell(v, 'rmse')} \\\\")
    lines.append("\\midrule")
print("\n".join(lines[:-1]))
