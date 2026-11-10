"""Cold-start-aware fusion (review 5: user-adaptive blending; cold-start regimes). For a test user without any
training rating, the CF model has no information about the user (its prediction reduces to item or global effects),
so the fused rating falls back on the LLM alone (w = 1); every other user keeps the weight chosen on validation.
No parameter is tuned. Paired bootstrap (10,000) against the convex fusion, on all test interactions and on the
cold-start users. Settings with cold-start users: Yelp-WH, Yelp-H, MovieLens-1M, Amazon-Books (the context-aware
datasets have at most 1% of such users). Usage: python cold_aware.py → results/cold_aware.json"""
import json
import numpy as np
from segments import counts_large
from stacking import paired

out = {}
for s, d in (("YELP-WH", "yelp"), ("YELP-H", "yelp"), ("MovieLens-H", "movielens"), ("AmazonBooks-H", "amazon")):
    uc, _ = counts_large(d)
    for m in ("mistral", "eurollm", "llama", "deepseek"):
        h = json.load(open(f"probs/hybrid/{m}_hycpr_{s}.json"))
        lp = f"probs/{m}_{s}.json" if s.startswith("YELP") else f"probs/{m}_full_{s}.json"
        t, hy = np.array(h["truth"], float), np.array(h["expected"], float)
        llm = np.array(json.load(open(lp))["expected"], float)
        cold = uc == 0
        ca = np.where(cold, llm, hy)
        mae = lambda p, k: float(np.mean(np.abs(p[k] - t[k])))
        rmse = lambda p, k: float(np.sqrt(np.mean((p[k] - t[k]) ** 2)))
        allm = np.ones(len(t), bool)
        out[f"{m} {s}"] = {"n_cold": int(cold.sum()), "w": h["w"],
                           "convex": [mae(hy, allm), rmse(hy, allm)], "cold_aware": [mae(ca, allm), rmse(ca, allm)],
                           "all_vs_convex": paired(ca, hy, t), "cold_vs_convex": paired(ca[cold], hy[cold], t[cold])}
        r = out[f"{m} {s}"]
        print(f"{m} {s} cold {cold.sum()} w {h['w']}: all {r['convex'][0]:.4f}/{r['convex'][1]:.4f} -> {r['cold_aware'][0]:.4f}/{r['cold_aware'][1]:.4f}"
              f" (MAE {r['all_vs_convex']['mae'][0]:+.4f}{'*' if r['all_vs_convex']['mae'][3] else ''}, RMSE {r['all_vs_convex']['rmse'][0]:+.4f}{'*' if r['all_vs_convex']['rmse'][3] else ''});"
              f" cold users MAE {r['cold_vs_convex']['mae'][0]:+.3f}{'*' if r['cold_vs_convex']['mae'][3] else ''}", flush=True)
json.dump(out, open("results/cold_aware.json", "w"), indent=1)
