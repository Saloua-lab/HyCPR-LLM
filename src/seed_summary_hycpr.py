"""HyCPR-LLM and the LLM alone (v0, 90% training) over 3 training seeds, for every LLM and small-dataset setting.

Inputs: results/improve_<m>_hycpr[_seed1|_seed2].json (hybrid.py <m> --only v0 --tag hycpr[_seed<k>] [--seed <k>]).
Output: results/seed_summary_hycpr.{md,json}: mean ± sd (over seeds) of test MAE / RMSE.
"""
import json

import numpy as np

MODELS = ["mistral", "eurollm", "llama", "deepseek"]
SETTINGS = ["LDOS-WH", "Depaul-WH", "Tijuana-WH", "LDOS-H", "Depaul-H", "Restaurant-H"]
out, lines = {}, ["| LLM | setting | LLM alone MAE | LLM alone RMSE | HyCPR-LLM MAE | HyCPR-LLM RMSE | RMSE gain > 0 (seeds) |",
                  "|---|---|---|---|---|---|---|"]
for m in MODELS:
    runs = [json.load(open(f"results/improve_{m}_hycpr{t}.json")) for t in ("", "_seed1", "_seed2")]
    for s in SETTINGS:
        llm = np.array([[r[s]["candidates"]["v0"]["test_mae"], r[s]["candidates"]["v0"]["test_rmse"]] for r in runs])
        hy = np.array([[r[s]["candidates"][r[s]["chosen"]]["test_mae"], r[s]["candidates"][r[s]["chosen"]]["test_rmse"]] for r in runs])
        out[f"{m} {s}"] = {"llm": llm.tolist(), "hycpr": hy.tolist(), "chosen": [r[s]["chosen"] for r in runs],
                           "baseline": [r[s]["candidates"][r[s]["chosen"]].get("baseline") for r in runs],
                           "w": [r[s]["candidates"][r[s]["chosen"]].get("w") for r in runs]}
        f = lambda a, k: f"{a[:, k].mean():.3f} ± {a[:, k].std(ddof=1):.3f}"
        lines.append(f"| {m} | {s} | {f(llm, 0)} | {f(llm, 1)} | {f(hy, 0)} | {f(hy, 1)} | {(hy[:, 1] < llm[:, 1]).sum()}/3 |")
json.dump(out, open("results/seed_summary_hycpr.json", "w"), indent=1)
open("results/seed_summary_hycpr.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# LaTeX body of tab:seeds (main_hycpr.tex)
NAMES = {"mistral": "Mistral", "eurollm": "EuroLLM", "llama": "Llama", "deepseek": "DeepSeek"}
tex = []
for m in MODELS:
    for i, s in enumerate(SETTINGS):
        llm, hy = np.array(out[f"{m} {s}"]["llm"]), np.array(out[f"{m} {s}"]["hycpr"])
        c = lambda a: f"{a[:, 0].mean():.3f}$\\pm${a[:, 0].std(ddof=1):.3f} / {a[:, 1].mean():.3f}$\\pm${a[:, 1].std(ddof=1):.3f}"
        tex.append(f"{f'\\multirow{{6}}{{*}}{{{NAMES[m]}}}' if i == 0 else ''} & {s} & {c(llm)} & {c(hy)} \\\\")
    tex.append("\\midrule")
open("results/seed_table.tex", "w").write("\n".join(tex[:-1]) + "\n")
