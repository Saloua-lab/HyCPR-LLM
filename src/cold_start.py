"""Cold-start regimes (review 5: "whether per-user cold-start regimes were evaluated explicitly").

For each -H setting, the test interactions are grouped by the number of training ratings of their user (0 = strict
user cold start, 1-5, 6-20, > 20) and by whether their item has a training rating (0 = strict item cold start).
In each group: MAE / RMSE of HyCPR-LLM (Mistral), CPR-LLM (Mistral, the LLM of the fusion), the best baseline of the
setting (as in segments.py), and Text-KNN (text_knn.py), with the paired bootstrap 95% interval of HyCPR-LLM minus the
best baseline (MAE). Groups with fewer than 30 test interactions are not reported. Also reported: the mean number of
history items in the -H test prompts of the group (on the chronological datasets a user without training ratings can
have earlier test-period interactions in the prompt history, which the baselines do not use).

Usage: python cold_start.py → results/cold_start.{json,md}
"""
import json
import re

import numpy as np

from segments import BEST, boot, counts_large, counts_small
from bootstrap import collect

LARGE = {"YELP-H": "yelp", "MovieLens-H": "movielens", "AmazonBooks-H": "amazon"}
SMALL = {"LDOS-H": "LDOS", "Depaul-H": "Depaul", "Restaurant-H": "Tijuana"}
DATA = {"LDOS-H": "data/variants/mistral_LDOS-H_val", "Depaul-H": "data/variants/mistral_Depaul-H_val",
        "Restaurant-H": "data/variants/mistral_Restaurant-H_val", "YELP-H": "data/YELP-H",
        "MovieLens-H": "data/MovieLens-H", "AmazonBooks-H": "data/AmazonBooks-H"}


def n_hist(setting):
    from datasets import load_from_disk
    try:
        ex = load_from_disk(DATA[setting])["test"]["example"]
    except Exception:
        return None
    return np.array([len(re.findall(r"rated (?:it|this \w+) as: \d", e.split("### Response")[0])) for e in ex])


def main():
    cols = collect()
    out, md = {}, ["| Setting | Group | n | hist. items | HyCPR-LLM | CPR-LLM | best baseline | Text-KNN | HyCPR - best (MAE) |",
                   "|---|---|---|---|---|---|---|---|---|"]
    for s, best in BEST.items():
        h = json.load(open(f"probs/hybrid/mistral_hycpr_{s}.json"))
        t, hy = np.array(h["truth"], float), np.array(h["expected"], float)
        if s in ("MovieLens-H", "AmazonBooks-H"):
            bp = np.array(json.load(open(f"results/large_baselines_{s[:-2]}.json"))["models"][best], float)
            llm = np.array(json.load(open(f"probs/mistral_full_{s}.json"))["expected"], float)
        else:
            bp = cols[s]["models"][best]
            llm = np.array(json.load(open(f"probs/variants/mistral_v0_{s}_test.json" if s in SMALL else f"probs/mistral_{s}.json"))["expected"], float)
        tk = np.array(json.load(open(f"probs/textknn_{s}.json"))["expected"], float)
        uc, ic = counts_large(LARGE[s]) if s in LARGE else counts_small(SMALL[s])
        nh = n_hist(s)
        if nh is not None and len(nh) != len(t):
            nh = None
        groups = {"users: 0": uc == 0, "users: 1-5": (uc >= 1) & (uc <= 5), "users: 6-20": (uc >= 6) & (uc <= 20),
                  "users: >20": uc > 20, "items: 0": ic == 0, "items: >=1": ic >= 1, "all": np.ones(len(t), bool)}
        rec = {"best": best}
        for g, m in groups.items():
            if m.sum() < 30:
                continue
            e = lambda p: [float(np.mean(np.abs(p[m] - t[m]))), float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))]
            d = boot(hy[m], bp[m], t[m])
            rec[g] = {"n": int(m.sum()), "hist": None if nh is None else float(nh[m].mean()), "hycpr": e(hy), "cpr": e(llm),
                      "best": e(bp), "textknn": e(tk), "hycpr_minus_best_mae": d}
            f = lambda x: f"{x[0]:.3f} / {x[1]:.3f}"
            sig = "*" if d[1] > 0 or d[2] < 0 else ""
            md.append(f"| {s} | {g} | {m.sum()} | {'' if nh is None else f'{nh[m].mean():.1f}'} | {f(rec[g]['hycpr'])} | {f(rec[g]['cpr'])} | "
                      f"{best} {f(rec[g]['best'])} | {f(rec[g]['textknn'])} | {d[0]:+.3f}{sig} [{d[1]:+.3f}, {d[2]:+.3f}] |")
            print(md[-1], flush=True)
        out[s] = rec
    json.dump(out, open("results/cold_start.json", "w"), indent=1)
    open("results/cold_start.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
