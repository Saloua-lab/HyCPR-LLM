"""Body of Table 4 (tab:mae_rmse_rotated_mindH) from saved predictions, with significance marks.

Values: baselines from results/baselines_*.json; CPR-LLM -WH from probs/<llm>_<setting>.json (original adapters),
-H from probs/<llm>_clean_<setting>.json (models trained and tested on leak-free histories), Yelp from
probs/<llm>_YELP-*.json (no leakage by construction). Cells without saved predictions (STATIC) are printed with a
dagger and take no part in the tests.

Marks, per column and metric: bold = best value; underlined = not significantly different from the best value
(paired bootstrap over the test interactions, B=10000, 95% interval of the difference contains 0).

HyCPR-LLM rows: probs/hybrid/<llm>_hycpr_<setting>.json (fused predictions), included in the tests.

Usage: python table4.py > results/table4_body.tex
"""
import json
import os

import numpy as np

from bootstrap import BASELINES, LLMS, collect, mae_rmse

COLS = ["LDOS-WH", "Depaul-WH", "Tijuana-WH", "LDOS-H", "Depaul-H", "Restaurant-H", "YELP-WH", "YELP-H"]
NAMES = {"mistral": "Mistral", "eurollm": "EuroLLM", "llama": "Llama", "deepseek": "DeepSeek"}
ROW_ORDER = ["mistral", "eurollm", "llama", "deepseek"]
GROUPS = [("Heuristic", ["GAR", "IAR", "UAR"]), ("Traditional", ["ItemKNN", "UserKNN", "SVD++", "BiasedMF"]),
          ("CA", ["FM", "CAMF_CI", "CAMF_CU", "CAMF_C", "CAMF_CUCI"]),
          ("Neural CA", ["NFM", "DeepFM", "AutoInt"])]
# CPR-LLM cells whose adapters are lost (DeepSeek Yelp: values of the manuscript, from the original evaluation)
STATIC = {("deepseek", "YELP-WH"): (0.875, 1.089), ("deepseek", "YELP-H"): (0.799, 1.018)}
B, SEED = 10000, 0


def llm_tag(llm, col):
    return f"{llm}_clean" if col.endswith("-H") and not col.startswith("YELP") else llm


def main():
    cols = collect()
    for (l, c) in list(STATIC):  # a re-run with saved predictions replaces the manuscript value
        if c in cols and l in cols[c]["models"]:
            del STATIC[(l, c)]
    cells, marks = {}, {}  # (row, col) -> (mae, rmse); (row, col, metric) -> "best"/"tie"
    for col in COLS:
        c = cols[col]
        rows = {m: c["models"][m] for m in BASELINES if m in c["models"]}
        for llm in LLMS:
            if llm_tag(llm, col) in c["models"]:
                rows[llm] = c["models"][llm_tag(llm, col)]
            hp = f"probs/hybrid/{llm}_hycpr_{col}.json"  # HyCPR-LLM: LLM + CF fusion (hybrid.py / large_baselines.py)
            if os.path.exists(hp):
                h = json.load(open(hp))
                assert np.array_equal(np.array(h["truth"], float), c["truth"]), hp
                rows[llm + "_hycpr"] = np.array(h["expected"], float)
        err = {r: p - c["truth"] for r, p in rows.items()}
        idx = np.random.default_rng(SEED).integers(0, len(c["truth"]), size=(B, len(c["truth"])))
        boot = {r: mae_rmse(e[idx]) for r, e in err.items()}
        for r, e in err.items():
            cells[(r, col)] = mae_rmse(e)
        for k, metric in enumerate(["mae", "rmse"]):
            vals = {r: cells[(r, col)][k] for r in rows}
            static = [v[k] for (l, cc), v in STATIC.items() if cc == col]
            best = min(vals, key=vals.get)
            if static and min(static) < vals[best]:
                raise SystemExit(f"{col} {metric}: a static cell is best; mark it by hand")
            for r in rows:
                if r == best:
                    marks[(r, col, metric)] = "best"
                    continue
                d = boot[r][k] - boot[best][k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                if lo <= 0 <= hi:
                    marks[(r, col, metric)] = "tie"
    for (llm, col), v in STATIC.items():
        cells[(llm, col)] = v

    def fmt(r, col, k):
        if (r, col) not in cells:
            return "n/a" if r.split("_")[0] in LLMS else "--"
        s = f"{cells[(r, col)][k]:.3f}"
        if (r, col) in STATIC:
            return s + "$^\\dagger$"
        m = marks.get((r, col, ["mae", "rmse"][k]))
        return f"\\textbf{{{s}}}" if m == "best" else f"\\underline{{{s}}}" if m == "tie" else s

    def row(r, label):
        return f"& {label:10s}& " + " & ".join(fmt(r, col, k) for col in COLS for k in (0, 1)) + " \\\\"

    out = []
    for group, models in GROUPS:
        out.append(f"\\multirow{{{len(models)}}}{{*}}{{{group}}}")
        out += [row(m, m.replace("_", "\\_")) for m in models]
        out.append("\\midrule")
    out.append(f"\\multirow{{{len(ROW_ORDER)}}}{{*}}{{\\llmonly{{}}}}")
    out += [row(l, NAMES[l]) for l in ROW_ORDER]
    out.append("\\midrule")
    out.append(f"\\multirow{{{len(ROW_ORDER)}}}{{*}}{{\\ours{{}}}}")
    out += [row(l + "_hycpr", NAMES[l]) for l in ROW_ORDER]
    print("\n".join(out))


if __name__ == "__main__":
    main()
