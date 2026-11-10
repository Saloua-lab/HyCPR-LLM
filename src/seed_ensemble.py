"""3-run seed ensemble of the Table 4 Mistral model (Table 4 run + probs/seeds/*_seed1, *_seed2).

The prediction is the mean of the three expected ratings. Compared on the test set (paired bootstrap, 10,000
replicates) with the Table 4 Mistral model and with the best Table 4 baseline.
Usage: python seed_ensemble.py → results/seed_ensemble_mistral.{json,md}
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap import collect, mae_rmse  # noqa: E402
from hybrid import SETTINGS, paired  # noqa: E402

table4 = collect()
report = {}
for s in SETTINGS:
    tag = "mistral_clean" if s.endswith("-H") else "mistral"
    seeds = [f"probs/seeds/{'mistral_clean' if s.endswith('-H') else 'mistral_full'}_{s}_seed{k}.json" for k in (1, 2)]
    col = table4[s]
    t = col["truth"]
    runs = [col["models"][tag]]
    for p in seeds:
        d = json.load(open(p))
        assert np.array_equal(np.array(d["truth"], float), t), p
        runs.append(np.array(d["expected"]))
    avg = np.mean(runs, axis=0)
    base = [m for m in col["models"] if not any(x in m for x in ("mistral", "llama", "eurollm", "deepseek"))]
    best_b = min(base, key=lambda m: mae_rmse(col["models"][m] - t)[0])
    report[s] = {
        "runs": [list(map(float, mae_rmse(r - t))) for r in runs],
        "ensemble": list(map(float, mae_rmse(avg - t))),
        "best_baseline": {"name": best_b, "mae_rmse": list(map(float, mae_rmse(col["models"][best_b] - t)))},
        "vs_table4": paired(avg, col["models"][tag], t),
        "vs_best_baseline": paired(avg, col["models"][best_b], t),
    }
json.dump(report, open("results/seed_ensemble_mistral.json", "w"), indent=1)
f = lambda x: f"{x[0]:.3f} / {x[1]:.3f}"
g = lambda p: ", ".join(f"{k.upper()} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]{' *' if v['significant'] else ''}"
                        for k, v in p.items())
lines = ["# Mistral 3-run seed ensemble (test MAE / RMSE)\n",
         "| setting | Table 4 run | seed 1 | seed 2 | ensemble | vs Table 4 | best baseline | vs best baseline |",
         "|---|---|---|---|---|---|---|---|"]
for s, r in report.items():
    lines.append(f"| {s} | {f(r['runs'][0])} | {f(r['runs'][1])} | {f(r['runs'][2])} | **{f(r['ensemble'])}** | "
                 f"{g(r['vs_table4'])} | {r['best_baseline']['name']} {f(r['best_baseline']['mae_rmse'])} | {g(r['vs_best_baseline'])} |")
open("results/seed_ensemble_mistral.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
