"""When does the fusion beat the best baseline? (review 5: "improvements on small CARS datasets are mixed ... leaving
questions about when the hybrid approach provides clear wins").

For every LLM x setting (6 context-aware columns, Yelp-WH/-H, MovieLens-1M, Amazon-Books; 40 cases): the RMSE (and
MAE) gap of the LLM inside the fusion to the best baseline of the column (lowest test error), the gap of HyCPR-LLM to
the same baseline, and the fusion weight w. Spearman correlations over the 40 cases.

Usage: python when_wins.py → results/when_wins.json
"""
import json

import numpy as np
from scipy.stats import spearmanr

from bootstrap import collect

SETTINGS = ["LDOS-WH", "Depaul-WH", "Tijuana-WH", "LDOS-H", "Depaul-H", "Restaurant-H", "YELP-WH", "YELP-H",
            "MovieLens-H", "AmazonBooks-H"]
LLMS = ["mistral", "eurollm", "llama", "deepseek"]


def main():
    cols = collect()
    rows = []
    for s in SETTINGS:
        for m in LLMS:
            h = json.load(open(f"probs/hybrid/{m}_hycpr_{s}.json"))
            t, hy = np.array(h["truth"], float), np.array(h["expected"], float)
            if s in cols:
                base = {k: v for k, v in cols[s]["models"].items() if not any(x in k for x in LLMS)}
                lp = f"probs/{m}_{s}.json" if s.startswith("YELP") else f"probs/variants/{m}_v0_{s}_test.json"
            else:
                base = {k: np.array(v, float) for k, v in json.load(open(f"results/large_baselines_{s[:-2]}.json"))["models"].items()
                        if not k.endswith("-50k")}
                lp = f"probs/{m}_full_{s}.json"
            llm = np.array(json.load(open(lp))["expected"], float)
            mae = lambda x: float(np.mean(np.abs(x - t)))
            rmse = lambda x: float(np.sqrt(np.mean((x - t) ** 2)))
            b = min(base, key=lambda k: rmse(base[k]))
            rows.append({"setting": s, "llm": m, "w": h.get("w"), "best": b,
                         "llm_gap_rmse": rmse(llm) - rmse(base[b]), "hycpr_gap_rmse": rmse(hy) - rmse(base[b]),
                         "llm_gap_mae": mae(llm) - mae(base[min(base, key=lambda k: mae(base[k]))]),
                         "hycpr_gap_mae": mae(hy) - mae(base[min(base, key=lambda k: mae(base[k]))])})
    g = lambda k: np.array([r[k] for r in rows], float)
    close = g("llm_gap_rmse") <= 0.05
    summary = {"n": len(rows),
               "spearman_rmse": spearmanr(g("llm_gap_rmse"), g("hycpr_gap_rmse"))[0],
               "spearman_mae": spearmanr(g("llm_gap_mae"), g("hycpr_gap_mae"))[0],
               "spearman_gap_w": spearmanr(g("llm_gap_rmse"), g("w"))[0],
               "close": {"n": int(close.sum()), "hycpr_below_best": float((g("hycpr_gap_rmse")[close] < 0).mean())},
               "far": {"n": int((~close).sum()), "hycpr_below_best": float((g("hycpr_gap_rmse")[~close] < 0).mean()),
                       "median_w": float(np.median(g("w")[~close]))}}
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "rows": rows}, open("results/when_wins.json", "w"), indent=1)


if __name__ == "__main__":
    main()
