"""Multiple-comparison correction of the paper's main significance claims (second review: "details about test types,
correction for multiple comparisons, and effect sizes are missing").

Test: paired bootstrap over the test interactions (B = 10,000; AUC: 2,000), two-sided p = 2 min(P(d<=0), P(d>=0)),
floored at 1/B. Holm-Bonferroni correction within each family of claims:
  F1  HyCPR-LLM vs the best baseline of the column (per metric), Table 5: 4 LLMs x 8 columns x 2 metrics
  F2  HyCPR-LLM vs the LLM alone (fusion ablation), Table 5: 4 LLMs x 8 columns x 2 metrics
  F3  HyCPR-LLM vs the strongest baseline of our split (per metric), Table 6: 4 LLMs x 2 datasets x 3 metrics
Effect size: the difference of the metric (and relative difference in %), with its 95% bootstrap interval.

Usage: python holm.py   → results/holm.{json,md}
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from bootstrap import BASELINES, collect, mae_rmse
from table4 import COLS, llm_tag

LLMS = ["mistral", "eurollm", "llama", "deepseek"]


def test(a, b, t, metric, B=10000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(t), (B if metric != "auc" else 2000, len(t)))
    if metric == "auc":
        f = lambda p, j: np.array([roc_auc_score(t[r] >= 3, p[r]) for r in j])
        full = roc_auc_score(t >= 3, a) - roc_auc_score(t >= 3, b)
    else:
        k = {"mae": 0, "rmse": 1}[metric]
        f = lambda p, j: mae_rmse(p[j] - t[j])[k]
        full = mae_rmse(a - t)[k] - mae_rmse(b - t)[k]
    d = f(a, idx) - f(b, idx)
    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))
    lo, hi = np.percentile(d, [2.5, 97.5])
    ref = (mae_rmse(b - t)[{"mae": 0, "rmse": 1}[metric]] if metric != "auc" else roc_auc_score(t >= 3, b))
    return {"diff": float(full), "rel": float(100 * full / ref), "ci": [float(lo), float(hi)], "p": float(min(p, 1))}


def holm(tests):
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    m, prev = len(tests), 0.0
    for r, i in enumerate(order):
        adj = min(1.0, max(prev, (m - r) * tests[i]["p"]))
        tests[i]["p_holm"], prev = adj, adj
        tests[i]["sig_raw"], tests[i]["sig_holm"] = tests[i]["p"] < 0.05, adj < 0.05
    return tests


def main():
    cols = collect()
    fam = {"F1": [], "F2": [], "F3": []}
    for col in COLS:
        c = cols[col]
        t = c["truth"]
        base = {b: c["models"][b] for b in BASELINES if b in c["models"]}
        for m in LLMS:
            hy = np.array(json.load(open(f"probs/hybrid/{m}_hycpr_{col}.json"))["expected"], float)
            for metric, k in (("mae", 0), ("rmse", 1)):
                best = min(base, key=lambda b: mae_rmse(base[b] - t)[k])
                fam["F1"].append({"llm": m, "col": col, "metric": metric, "vs": best, **test(hy, base[best], t, metric)})
                if llm_tag(m, col) in c["models"]:
                    fam["F2"].append({"llm": m, "col": col, "metric": metric, "vs": "LLM alone",
                                      **test(hy, c["models"][llm_tag(m, col)], t, metric)})
    for name in ("MovieLens", "AmazonBooks"):
        b = json.load(open(f"results/large_baselines_{name}.json"))
        t = np.array(b["truth"], float)
        P = {k: np.array(v, float) for k, v in b["models"].items() if k != "GAR"}
        for m in LLMS:
            hy = np.array(json.load(open(f"probs/hybrid/{m}_hycpr_{name}-H.json"))["expected"], float)
            for metric in ("rmse", "mae", "auc"):
                if metric == "auc":
                    best = max(P, key=lambda k: roc_auc_score(t >= 3, P[k]))
                else:
                    best = min(P, key=lambda k: mae_rmse(P[k] - t)[{"mae": 0, "rmse": 1}[metric]])
                fam["F3"].append({"llm": m, "col": name, "metric": metric, "vs": best, **test(hy, P[best], t, metric)})
    md = ["# Holm-Bonferroni correction of the main claims\n"]
    for f, tests in fam.items():
        holm(tests)
        raw = sum(x["sig_raw"] for x in tests)
        kept = sum(x["sig_holm"] for x in tests)
        lost = [f"{x['llm']} {x['col']} {x['metric']} vs {x['vs']} ({x['diff']:+.3f}, p={x['p']:.4f})"
                for x in tests if x["sig_raw"] and not x["sig_holm"]]
        md.append(f"## {f}: {len(tests)} tests, significant before / after Holm: {raw} / {kept}")
        md += [f"- lost: {l}" for l in lost] + [""]
        md.append("| LLM | setting | metric | vs | diff [95% CI] | rel. % | p | p Holm |")
        md.append("|---|---|---|---|---|---|---|---|")
        md += [f"| {x['llm']} | {x['col']} | {x['metric']} | {x['vs']} | {x['diff']:+.4f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | "
               f"{x['rel']:+.1f} | {x['p']:.4f} | {x['p_holm']:.4f} |" for x in tests]
        md.append("")
    json.dump(fam, open("results/holm.json", "w"), indent=1)
    open("results/holm.md", "w").write("\n".join(md))
    print("\n".join(l for l in md if l.startswith("#") or l.startswith("- lost")))


if __name__ == "__main__":
    main()
