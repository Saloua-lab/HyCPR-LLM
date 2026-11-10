"""Bootstrap confidence intervals and paired bootstrap tests over the test interactions (reviewer remark 3).

For every test set (column of Table 4) all models are evaluated on the same test interactions in the same
order (checked below), so a model comparison can be paired: each bootstrap replicate draws the test
interactions with replacement once and computes the metric of both models on that same draw.

- CI: 95% percentile interval of MAE and RMSE over B replicates.
- Paired test of A vs B: 95% interval of MAE(A) - MAE(B) (and RMSE) and a two-sided bootstrap p-value
  p = 2 * min(P(d <= 0), P(d >= 0)); "significant" = the interval excludes 0.

Inputs: probs/<model>[_clean]_<setting>[_noleak].json (LLMs, field `expected`) and
results/baselines_small.json, results/baselines_yelp.json (field `pred`). Missing files are skipped, so the
script can be re-run as new runs finish.

Usage: python bootstrap.py [--B 10000]   → results/bootstrap.json, results/bootstrap.md
"""
import argparse
import json
import os

import numpy as np

COLUMNS = {  # column -> (baseline dataset key, LLM setting name in probs/, is_history)
    "LDOS-WH": ("LDOS", "LDOS-WH", False), "Depaul-WH": ("Depaul", "Depaul-WH", False),
    "Tijuana-WH": ("Tijuana", "Tijuana-WH", False), "LDOS-H": ("LDOS", "LDOS-H", True),
    "Depaul-H": ("Depaul", "Depaul-H", True), "Restaurant-H": ("Tijuana", "Restaurant-H", True),
    "YELP-WH": ("YELP", "YELP-WH", False), "YELP-H": ("YELP", "YELP-H", True),
}
WH_OF = {"LDOS-H": "LDOS-WH", "Depaul-H": "Depaul-WH", "Restaurant-H": "Tijuana-WH", "YELP-H": "YELP-WH"}
LLMS = ["mistral", "deepseek", "llama", "eurollm"]
BASELINES = ["GAR", "IAR", "UAR", "ItemKNN", "UserKNN", "SVD++", "BiasedMF", "FM",
             "CAMF_CI", "CAMF_CU", "CAMF_C", "CAMF_CUCI", "NFM", "DeepFM", "AutoInt"]


def load_probs(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return np.array(d["truth"], float), np.array(d["expected"], float)


def collect():
    """{column: {"truth": array, "models": {name: pred array}}}. LLM names: <model> (original adapter),
    <model>_clean (retrained on leak-free histories), <model>_noit (no instruction tuning)."""
    base = {}
    for f in ["results/baselines_small.json", "results/baselines_yelp.json",
              "results/baselines_neural_small.json", "results/baselines_neural_yelp.json"]:
        if os.path.exists(f) and os.path.getsize(f):
            for k, v in json.load(open(f)).items():  # neural files add models to the same columns
                if k in base:
                    assert v["truth"] == base[k]["truth"], f
                    base[k]["models"].update(v["models"])
                else:
                    base[k] = v
    cols = {}
    for col, (bkey, setting, hist) in COLUMNS.items():
        truth, models = None, {}
        if bkey in base:
            truth = np.array(base[bkey]["truth"], float)
            for m, v in base[bkey]["models"].items():
                models[m] = np.array(v["pred"], float)
        suffix = "_noleak" if hist and not setting.startswith("YELP") else ""
        for llm in LLMS:
            for tag, path in [(llm, f"probs/{llm}_{setting}{suffix}.json"),
                              (llm + "_clean", f"probs/{llm}_clean_{setting}.json"),
                              (llm + "_noit", f"probs/{llm}_noit_{setting}{suffix}.json"),
                              (llm + "_noitclean", f"probs/{llm}_noitclean_{setting}.json")]:
                r = load_probs(path)
                if r is None:
                    continue
                t, p = r
                if truth is None:
                    truth = t
                assert np.array_equal(t, truth), f"{path}: test interactions not in the baseline order"
                models[tag] = p
        if truth is not None:
            cols[col] = {"truth": truth, "models": models}
    return cols


def mae_rmse(err):
    """err: (B, n) or (n,) errors -> (mae, rmse) along the last axis."""
    return np.abs(err).mean(-1), np.sqrt((err ** 2).mean(-1))


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cols = collect()
    out = {"B": args.B, "seed": args.seed, "columns": {}, "paired": []}
    idx_of = {}
    for col, c in cols.items():
        n = len(c["truth"])
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, n, size=(args.B, n))  # same draws for every model of the column -> paired
        idx_of[col] = idx
        res = {}
        for m, p in c["models"].items():
            e = p - c["truth"]
            mae, rmse = mae_rmse(e)
            bm, br = mae_rmse(e[idx])
            res[m] = {"mae": float(mae), "rmse": float(rmse), "mae_ci": ci(bm), "rmse_ci": ci(br)}
        out["columns"][col] = {"n": n, "models": res}

    def paired(col_a, a, col_b, b, why):
        ca, cb = cols[col_a], cols[col_b]
        if a not in ca["models"] or b not in cb["models"]:
            return
        assert np.array_equal(ca["truth"], cb["truth"])  # -WH and -H of a dataset share the test interactions
        idx = idx_of[col_a]
        ea, eb = ca["models"][a] - ca["truth"], cb["models"][b] - cb["truth"]
        (ma, ra), (mb, rb) = mae_rmse(ea), mae_rmse(eb)
        (bma, bra), (bmb, brb) = mae_rmse(ea[idx]), mae_rmse(eb[idx])
        rec = {"why": why, "a": f"{a} ({col_a})", "b": f"{b} ({col_b})"}
        for name, d, dobs in [("mae", bma - bmb, ma - mb), ("rmse", bra - brb, ra - rb)]:
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            lo, hi = ci(d)
            rec[name] = {"diff": float(dobs), "ci": [lo, hi], "p": float(min(p, 1.0)),
                         "significant": bool(lo > 0 or hi < 0)}
        out["paired"].append(rec)

    for col, c in cols.items():
        base_models = [m for m in BASELINES if m in c["models"]]
        llm_models = [m for m in c["models"] if m not in BASELINES]
        if not base_models or not llm_models:
            continue
        best_base = min(base_models, key=lambda m: out["columns"][col]["models"][m]["mae"])
        best_base_r = min(base_models, key=lambda m: out["columns"][col]["models"][m]["rmse"])
        for m in llm_models:
            paired(col, m, col, best_base, "LLM vs best baseline (MAE)")
            if best_base_r != best_base:
                paired(col, m, col, best_base_r, "LLM vs best baseline (RMSE)")
        main_tag = "_clean" if col.endswith("-H") and not col.startswith("YELP") else ""  # main -H = leak-free training
        main_llms = [llm + main_tag for llm in LLMS if llm + main_tag in llm_models]
        ranked = sorted(main_llms, key=lambda m: out["columns"][col]["models"][m]["mae"])
        for m in ranked[1:]:
            paired(col, ranked[0], col, m, "best LLM vs other LLM")
        for llm in LLMS:
            paired(col, llm + "_clean", col, llm, "leak-free retraining vs original training")
            paired(col, llm, col, llm + "_noit", "instruction-tuned vs base")
            paired(col, llm + "_clean", col, llm + "_noitclean", "instruction-tuned vs base (leak-free training)")
            if col in WH_OF and WH_OF[col] in cols:
                for tag in (llm, llm + "_clean"):
                    paired(col, tag, WH_OF[col], llm, "with vs without history")

    json.dump(out, open("results/bootstrap.json", "w"), indent=1)
    with open("results/bootstrap.md", "w") as f:
        f.write(f"# Bootstrap over test interactions (B={args.B}, seed {args.seed})\n\n")
        for col, c in out["columns"].items():
            f.write(f"## {col} (n={c['n']})\n\n| model | MAE [95% CI] | RMSE [95% CI] |\n|---|---|---|\n")
            for m, v in sorted(c["models"].items(), key=lambda kv: kv[1]["mae"]):
                f.write(f"| {m} | {v['mae']:.3f} [{v['mae_ci'][0]:.3f}, {v['mae_ci'][1]:.3f}] | "
                        f"{v['rmse']:.3f} [{v['rmse_ci'][0]:.3f}, {v['rmse_ci'][1]:.3f}] |\n")
            f.write("\n")
        f.write("## Paired comparisons (A - B; * = 95% interval excludes 0)\n\n"
                "| comparison | A | B | ΔMAE [95% CI] p | ΔRMSE [95% CI] p |\n|---|---|---|---|---|\n")
        for r in out["paired"]:
            cell = lambda d: (f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] "
                              f"p={d['p']:.3f}{' *' if d['significant'] else ''}")
            f.write(f"| {r['why']} | {r['a']} | {r['b']} | {cell(r['mae'])} | {cell(r['rmse'])} |\n")
    print(f"{len(out['columns'])} columns, {len(out['paired'])} paired comparisons → results/bootstrap.md")


if __name__ == "__main__":
    main()
