"""LaTeX rows of the MovieLens / Amazon-Books table, block "on our split": baselines of large_baselines.py, CPR-LLM
(LLM only) and HyCPR-LLM (fusion chosen on validation, results/table6_hybrid.json). Bold = best of the block,
underlined = not significantly different from it (paired bootstrap 95%, 10,000 replicates; AUC 2,000).
Usage: python table6_rows.py"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score

ROWS = [("GAR", "GAR"), ("IAR", "IAR"), ("UAR", "UAR"), ("UAR-hist", "UAR (history in the prompt)"),
        ("BiasedMF-50k", "BiasedMF ($50$k train.)"), ("BiasedMF", "BiasedMF"), ("MLP", "MLP"),
        ("NFM", "NFM"), ("DeepFM", "DeepFM"), ("AutoInt", "AutoInt"),
        ]
LLMS = [("mistral", "Mistral-7B"), ("eurollm", "EuroLLM-9B"), ("llama", "Llama-2-7B"), ("deepseek", "DeepSeek-7B")]
ROWS += [(f"LLM_{m}", f"\\llmonly{{}} ({n}, -H)") for m, n in LLMS]
ROWS += [(f"HY_{m}", f"\\ours{{}} ({n}, -H)") for m, n in LLMS]
hy = json.load(open("results/table6_hybrid.json"))
preds, truth = {}, {}
for name in ("MovieLens", "AmazonBooks"):
    b = json.load(open(f"results/large_baselines_{name}.json"))
    t = np.array(b["truth"])
    P = {k: np.array(v) for k, v in b["models"].items()}
    for m, _ in LLMS:
        l = json.load(open(f"probs/{m}_full_{name}-H.json"))
        h = json.load(open(f"probs/hybrid/{m}_hycpr_{name}-H.json"))  # saved by large_baselines.py hybrid
        assert np.array_equal(np.array(l["truth"], float), t) and np.array_equal(np.array(h["truth"], float), t)
        P[f"LLM_{m}"], P[f"HY_{m}"] = np.array(l["expected"]), np.array(h["expected"])
        print(f"% {name} {m}: fusion with {h['baseline']}, w = {h['w']}")
    preds[name], truth[name] = P, t
rng = np.random.default_rng(0)
cells = {}
for name, P in preds.items():
    t = truth[name]; idx = rng.integers(0, len(t), (10000, len(t)))
    f = {"rmse": lambda p, j: np.sqrt(((p[j] - t[j]) ** 2).mean(-1)), "mae": lambda p, j: np.abs(p[j] - t[j]).mean(-1)}
    val = {(m, k): (fn(P[m], np.arange(len(t))[None])[0]) for m, _ in ROWS for k, fn in f.items()}
    for m, _ in ROWS:
        val[(m, "auc")] = roc_auc_score(t >= 3, P[m]) if m != "GAR" else 0.5
    for k in ("rmse", "mae", "auc"):
        sign = -1 if k == "auc" else 1
        best = min((m for m, _ in ROWS), key=lambda m: sign * val[(m, k)])
        for m, _ in ROWS:
            if m == best:
                mark = "best"
            elif m == "GAR" and k == "auc":
                mark = ""
            else:
                if k == "auc":
                    d = np.array([roc_auc_score(t[j] >= 3, P[m][j]) - roc_auc_score(t[j] >= 3, P[best][j]) for j in idx[:2000]])
                else:
                    d = f[k](P[m], idx) - f[k](P[best], idx)
                lo, hi = np.percentile(d, [2.5, 97.5])
                mark = "same" if lo <= 0 <= hi else ""
            v = f"{val[(m, k)]:.4f}"
            cells[(name, m, k)] = "\\best{" + v + "}" if mark == "best" else ("\\underline{" + v + "}" if mark == "same" else v)
for m, label in ROWS:
    if m == "LLM_mistral" or m == "HY_mistral":
        print("\\midrule")
    print(f"{label} & " + " & ".join(cells[(n, m, k)] for n in ("MovieLens", "AmazonBooks") for k in ("rmse", "mae", "auc")) + " \\\\")
