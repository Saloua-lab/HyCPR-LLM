"""When does HyCPR-LLM gain or lose against the best baseline? (review 3: "analysis of when/why the method
underperforms is limited").

For each -H setting, the test interactions are split by user activity and item popularity (number of training
ratings, tertiles), and by the true rating (1-2, 3, 4-5). In each segment: MAE of HyCPR-LLM (Mistral) and of the best
baseline of the setting, and the paired bootstrap 95% interval of the difference (10,000 replicates). The weight w of
the LLM in the fusion is reported too.

Usage: python segments.py   → results/segments.{json,md}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap import collect, mae_rmse  # noqa: E402

BEST = {"LDOS-H": "DeepFM", "Depaul-H": "CAMF_CU", "Restaurant-H": "DeepFM", "YELP-H": "BiasedMF",
        "MovieLens-H": "BiasedMF", "AmazonBooks-H": "MLP"}


def counts_small(name):
    from baselines_same_split import SMALL, recover_users
    tr, te, _ = recover_users(*SMALL[name])
    tr["u"], te["u"] = tr.u.astype(str), te.u.astype(str)
    return te.u.map(tr.u.value_counts()).fillna(0).to_numpy(), te.i.map(tr.i.value_counts()).fillna(0).to_numpy()


def counts_large(dataset):
    if dataset == "yelp":
        from baselines_same_split import yelp_split
        df, _, te, cut = yelp_split()
    else:
        from contamination import test_split
        df, te = test_split(dataset)
        cut = int(len(df) * 0.9)
    tr = df.iloc[:cut]
    u = df.loc[te, "user"].map(tr["user"].value_counts()).fillna(0).to_numpy()
    i = df.loc[te, "item"].map(tr["item"].value_counts()).fillna(0).to_numpy()
    return u, i


def tertiles(x):
    a, b = np.quantile(x, [1 / 3, 2 / 3])
    lab = np.where(x <= a, "low", np.where(x <= b, "mid", "high"))
    return lab, (a, b)


def boot(a, b, t, B=10000):
    idx = np.random.default_rng(0).integers(0, len(t), (B, len(t)))
    d = mae_rmse((a - t)[idx])[0] - mae_rmse((b - t)[idx])[0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(mae_rmse(a - t)[0] - mae_rmse(b - t)[0]), float(lo), float(hi)


def main():
    cols = collect()
    out, md = {}, ["# HyCPR-LLM (Mistral) minus best baseline, MAE, by segment (paired bootstrap 95% CI; * excludes 0)\n"]
    for s, best in BEST.items():
        h = json.load(open(f"probs/hybrid/mistral_hycpr_{s}.json"))
        t, hy = np.array(h["truth"], float), np.array(h["expected"], float)
        if s in ("MovieLens-H", "AmazonBooks-H"):
            b = json.load(open(f"results/large_baselines_{s[:-2]}.json"))
            bp = np.array(b["models"][best], float)
            uc, ic = counts_large({"MovieLens-H": "movielens", "AmazonBooks-H": "amazon"}[s])
        else:
            c = cols[s]
            assert np.array_equal(c["truth"], t)
            bp = c["models"][best]
            if s == "YELP-H":
                uc, ic = counts_large("yelp")
            else:
                uc, ic = counts_small({"LDOS-H": "LDOS", "Depaul-H": "Depaul", "Restaurant-H": "Tijuana"}[s])
        ul, ucut = tertiles(uc)
        il, icut = tertiles(ic)
        rl = np.where(t <= 2, "1-2", np.where(t == 3, "3", "4-5"))
        rec = {"best": best, "w": h.get("w"), "cf": h.get("baseline"), "all": boot(hy, bp, t), "segments": {}}
        line = [f"## {s}: vs {best}; fusion with {h.get('baseline')}, w = {h.get('w')}; all: {rec['all'][0]:+.3f} [{rec['all'][1]:+.3f}, {rec['all'][2]:+.3f}]"]
        for gname, lab, cuts in (("user activity", ul, ucut), ("item popularity", il, icut), ("true rating", rl, None)):
            parts = []
            for v in (["low", "mid", "high"] if cuts is not None else ["1-2", "3", "4-5"]):
                m = lab == v
                if m.sum() < 15:
                    continue
                d = boot(hy[m], bp[m], t[m])
                rec["segments"][f"{gname}: {v}"] = {"n": int(m.sum()), "diff": d,
                                                    "hy_mae": float(mae_rmse(hy[m] - t[m])[0]),
                                                    "base_mae": float(mae_rmse(bp[m] - t[m])[0])}
                sig = "*" if d[1] > 0 or d[2] < 0 else ""
                parts.append(f"{v} (n={m.sum()}): {d[0]:+.3f}{sig}")
            cut_txt = f" [cuts {cuts[0]:.0f}/{cuts[1]:.0f} training ratings]" if cuts is not None else ""
            line.append(f"- {gname}{cut_txt}: " + "; ".join(parts))
        out[s] = rec
        md += line + [""]
        print("\n".join(line), flush=True)
    json.dump(out, open("results/segments.json", "w"), indent=1)
    open("results/segments.md", "w").write("\n".join(md))


if __name__ == "__main__":
    main()
