"""Improvement experiments: LLM variants, LLM + baseline hybrid, and selection on the validation split.

For each setting (data/variants/<model>_<setting>_interactions.parquet, made by make_variants.py):
  1. every baseline of Table 4 is fitted on the training part (90%) and predicts validation and test;
  2. for each LLM variant (probs/variants/<model>_<variant>_<setting>_{validation,test}.json):
       - LLM alone;
       - hybrid  w * LLM + (1 - w) * baseline, with the baseline and w in {0, 0.05, ..., 1} chosen on validation
         (lowest validation RMSE);
  3. the final method of the setting is the variant/hybrid with the lowest validation RMSE;
  4. on the test set it is compared, with the paired bootstrap, with the best Table 4 baseline (fitted on the
     whole training data, so slightly stronger than the 90% fits) and with the main CPR-LLM model of Table 4.
Nothing is chosen on the test set.

Usage: python hybrid.py [model] [--only v0] [--tag hycpr] [--seed 1] [--neural] [--refit]   → results/improve_<model>.json, results/improve_<model>.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines_same_split import heuristics  # noqa: E402
from bootstrap import collect, mae_rmse  # noqa: E402

SETTINGS = ["LDOS-WH", "Depaul-WH", "Tijuana-WH", "LDOS-H", "Depaul-H", "Restaurant-H"]
VARIANTS = {"v0": "original recipe (90% train)", "vr": "rating-only loss + early stopping",
            "vrcf": "rating-only loss + early stopping + CF hints"}
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
B = 10000


def fit_baselines(df, key=None, neural=False):
    from baselines import MODELS, factor_model, knn
    ctx = sorted(c for c in df.columns if c.startswith("c_"))
    T = df[(df.split == "train") & (df.u != "None")]
    out = {}
    for sp in ("validation", "test"):
        P = df[df.split == sp]
        preds = heuristics(T, P)
        for m in MODELS:
            preds[m] = np.asarray(knn(T, P, ctx, by="item" if m == "ItemKNN" else "user") if m.endswith("KNN")
                                  else factor_model(T, P, ctx, m), float)
        out[sp] = preds
    if neural:  # NFM, DeepFM, AutoInt fitted on the same 90% part (cached: the split is the same for every LLM)
        import neural_cars as nc
        cpath = f"data/variants/neural_{key}.json"
        if os.path.exists(cpath):
            cached = json.load(open(cpath))
            assert cached["n"] == [int((df.split == sp).sum()) for sp in ("train", "validation", "test")]
        else:
            V, Te = df[df.split == "validation"], df[df.split == "test"]
            both = pd.concat([V, Te])
            cached = {"n": [int((df.split == sp).sum()) for sp in ("train", "validation", "test")], "models": {}}
            for m in nc.MODELS:
                p, h = nc.fit_predict(T.assign(r=T.r.astype(float)), both, ["u", "i"] + ctx, m)
                cached["models"][m] = {"validation": p[:len(V)].tolist(), "test": p[len(V):].tolist(), "hyper": h}
            json.dump(cached, open(cpath, "w"))
        for m, v in cached["models"].items():
            out["validation"][m] = np.array(v["validation"], float)
            out["test"][m] = np.array(v["test"], float)
    return out


def paired(a, b, t, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(t), (B, len(t)))
    res = {}
    for k, name in enumerate(["mae", "rmse"]):
        d = mae_rmse((a - t)[idx])[k] - mae_rmse((b - t)[idx])[k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[name] = {"diff": float(mae_rmse(a - t)[k] - mae_rmse(b - t)[k]), "ci": [float(lo), float(hi)],
                     "significant": bool(lo > 0 or hi < 0)}
    return res


def main(model, only=None, tag_out=None, seed=None, neural=False, refit=False):
    """only: restrict candidates (e.g. ['v0'] = HyCPR-LLM, the same procedure for every LLM); tag_out: output suffix."""
    out_name = f"{model}{'_' + tag_out if tag_out else ''}"
    table4 = collect()
    report, cache = {}, {}
    for s in SETTINGS:
        ipath = f"data/variants/{model}_{s}_interactions.parquet"
        if not os.path.exists(ipath):
            continue
        df = pd.read_parquet(ipath)
        key = s.replace("-WH", "").replace("-H", "").replace("Restaurant", "Tijuana")
        if key not in cache:
            cache[key] = fit_baselines(df, key, neural)
        base = cache[key]
        tv = df[df.split == "validation"]["r"].to_numpy(float)
        tt = df[df.split == "test"]["r"].to_numpy(float)
        cands = {}
        sources = {v: [v] for v in VARIANTS}
        sources["v0ens"] = ["v0", "v0_seed1", "v0_seed2"]  # mean of 3 training seeds of the v0 recipe
        if seed:  # same procedure with the LLM of another training seed (v0 -> v0_seed<k>)
            sources["v0"] = [f"v0_seed{seed}"]
        for v, runs in sources.items():
            if only and v not in only:
                continue
            # run "v0" → mistral_v0_<s>_<split>.json, run "v0_seed1" → mistral_v0_<s>_seed1_<split>.json
            paths = [[f"probs/variants/{model}_{r.split('_')[0]}_{s}{'_' + r.split('_')[1] if '_' in r else ''}_{sp}.json"
                      for sp in ("validation", "test")] for r in runs]
            if not all(os.path.exists(p) for pp in paths for p in pp):
                continue
            lvs, lts = [], []
            for pv, pt in paths:
                lv, lt = (json.load(open(p)) for p in (pv, pt))
                assert np.array_equal(np.array(lv["truth"], float), tv) and np.array_equal(np.array(lt["truth"], float), tt)
                lvs.append(np.array(lv["expected"])); lts.append(np.array(lt["expected"]))
            lv, lt = np.mean(lvs, axis=0), np.mean(lts, axis=0)
            cands[v] = {"val": lv, "test": lt}
            best = None
            for bname, bv in base["validation"].items():
                for w in WEIGHTS:
                    r = mae_rmse(w * lv + (1 - w) * bv - tv)[1]
                    if best is None or r < best[0]:
                        best = (r, bname, w)
            _, bname, w = best
            cands[f"{v}+hybrid"] = {"val": w * lv + (1 - w) * base["validation"][bname],
                                    "test": w * lt + (1 - w) * base["test"][bname], "baseline": bname, "w": float(w)}
        if not cands:
            continue
        for c in cands.values():
            c["val_mae"], c["val_rmse"] = map(float, mae_rmse(c["val"] - tv))
            c["test_mae"], c["test_rmse"] = map(float, mae_rmse(c["test"] - tt))
        col = table4[s]
        assert np.array_equal(col["truth"], tt)
        if refit:  # test side of the fusion: the chosen CF model refitted on the whole training split (= Table 5 row)
            for c in cands.values():
                if "baseline" in c:
                    b = c["baseline"]  # w * LLM + (1 - w) * CF(90%)  ->  w * LLM + (1 - w) * CF(100%)
                    c["test"] = c["test"] + (1 - c["w"]) * (col["models"][b] - base["test"][b])
                    c["test_mae"], c["test_rmse"] = map(float, mae_rmse(c["test"] - tt))
        chosen = min(cands, key=lambda k: cands[k]["val_rmse"])
        names_b = [m for m in col["models"] if m in base["test"]]
        best_b = min(names_b, key=lambda m: mae_rmse(col["models"][m] - tt)[0])
        tag = f"{model}_clean" if s.endswith("-H") else model
        report[s] = {
            "candidates": {k: {kk: vv for kk, vv in c.items() if kk not in ("val", "test")} for k, c in cands.items()},
            "chosen": chosen,
            "best_table4_baseline": {"name": best_b, "mae": float(mae_rmse(col["models"][best_b] - tt)[0]),
                                     "rmse": float(mae_rmse(col["models"][best_b] - tt)[1])},
            "table4_llm": {"mae": float(mae_rmse(col["models"][tag] - tt)[0]), "rmse": float(mae_rmse(col["models"][tag] - tt)[1])},
            "chosen_vs_best_baseline": paired(cands[chosen]["test"], col["models"][best_b], tt),
            "chosen_vs_table4_llm": paired(cands[chosen]["test"], col["models"][tag], tt),
        }
        os.makedirs("probs/hybrid", exist_ok=True)  # per-prompt fused test predictions (conformal prediction, CIs)
        json.dump({"truth": tt.tolist(), "expected": cands[chosen]["test"].tolist(), "chosen": chosen,
                   "baseline": cands[chosen].get("baseline"), "w": cands[chosen].get("w")},
                  open(f"probs/hybrid/{out_name}_{s}.json", "w"))
    json.dump(report, open(f"results/improve_{out_name}.json", "w"), indent=1)
    with open(f"results/improve_{out_name}.md", "w") as f:
        f.write(f"# Improvement experiments, {model} (selection on validation, numbers on test)\n\n")
        for s, r in report.items():
            f.write(f"## {s}\n\nTable 4: {model} {r['table4_llm']['mae']:.3f} / {r['table4_llm']['rmse']:.3f}; "
                    f"best baseline {r['best_table4_baseline']['name']} {r['best_table4_baseline']['mae']:.3f} / "
                    f"{r['best_table4_baseline']['rmse']:.3f}\n\n| candidate | val MAE / RMSE | test MAE / RMSE | hybrid |\n|---|---|---|---|\n")
            for k, c in r["candidates"].items():
                h = f"{c['baseline']} w={c['w']}" if "baseline" in c else ""
                mark = " **chosen**" if k == r["chosen"] else ""
                f.write(f"| {k}{mark} | {c['val_mae']:.3f} / {c['val_rmse']:.3f} | {c['test_mae']:.3f} / {c['test_rmse']:.3f} | {h} |\n")
            for lab, key in [("vs best baseline", "chosen_vs_best_baseline"), ("vs Table 4 LLM", "chosen_vs_table4_llm")]:
                d = r[key]
                f.write(f"\nchosen {lab}: MAE {d['mae']['diff']:+.3f} [{d['mae']['ci'][0]:+.3f}, {d['mae']['ci'][1]:+.3f}]"
                        f"{' *' if d['mae']['significant'] else ''}, RMSE {d['rmse']['diff']:+.3f} "
                        f"[{d['rmse']['ci'][0]:+.3f}, {d['rmse']['ci'][1]:+.3f}]{' *' if d['rmse']['significant'] else ''}\n")
            f.write("\n")
    print(open(f"results/improve_{out_name}.md").read())


if __name__ == "__main__":
    # hybrid.py <model> [--only v0 [...]] [--tag name]
    a = sys.argv[1:]
    only = a[a.index("--only") + 1:] if "--only" in a else None
    if only and "--tag" in only:
        only = only[:only.index("--tag")]
    if only and "--seed" in only:
        only = only[:only.index("--seed")]
    tag_out = a[a.index("--tag") + 1] if "--tag" in a else None
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else None
    neural = "--neural" in a  # add NFM, DeepFM, AutoInt to the pool of the fusion
    refit = "--refit" in a  # chosen CF model refitted on the whole training split for the test predictions
    if only:
        only = [o for o in only if o not in ("--neural", "--refit")]
    main(a[0] if a and not a[0].startswith("--") else "mistral", only, tag_out, seed, neural, refit)
