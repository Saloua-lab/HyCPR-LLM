"""Baselines for the MovieLens / Amazon-Books table on exactly the CPR-LLM split, and paired comparison.

  fit      python large_baselines.py fit {movielens,amazon}
           Rebuilds the split of prepare_large.py (same seed, same filtering), checks it against the saved test
           prompts (user id prefix and rating), and fits on the training part (first 90% of interactions):
             GAR       global mean rating
             IAR       item mean (GAR if unseen)
             UAR       user mean (GAR if unseen)
             UAR-hist  mean of the ratings in the test prompt's history (the value used so far in the paper)
             BiasedMF  mu + b_u + b_i + p_u.q_i                    (dim 32, Adam, MSE)
             MLP       MLP([e_u, e_i]) with user/item embeddings   (dim 32, hidden 64-32, Adam, MSE)
             NFM, DeepFM, AutoInt  neural_cars.py over (user, item); embedding size 32/64 chosen on validation
           MF and MLP use the full training part (as in Kang et al. 2023) and also, as a same-data-budget
           check, the 50k training interactions the LLM was fine-tuned on (suffix "-50k"). Early stopping on
           the chronologically last 5% of the training part; the validation predictions come from that fit, and
           the test predictions from a refit on the whole training part for the chosen number of epochs (2026-11-06).
           The test set is never used for any choice.
           → results/large_baselines_<Name>.json  (truth + predictions on the 2,000 test interactions)

  compare  python large_baselines.py compare
           CPR-LLM = expected rating of probs/mistral_full_<Name>-H.json (evaluate_probs.py). RMSE, MAE, AUC
           (ratings >= 3 positive) with paired bootstrap 95% CIs (10,000 replicates) of LLM minus each baseline.
           → results/table6_recomputed.{json,md}
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NAMES = {"movielens": "MovieLens", "amazon": "AmazonBooks", "yelp": "YELP"}
VARIANTS = {"MovieLens": ["H"], "AmazonBooks": ["H"], "YELP": ["WH", "H"]}  # prompt versions with a CPR-LLM model


LLMS = {"YELP": ["mistral", "deepseek", "llama", "eurollm"],  # LLMs with models per dataset (default: Mistral only)
        "MovieLens": ["mistral", "mistral_seed1", "deepseek", "llama", "eurollm"],
        "AmazonBooks": ["mistral", "mistral_seed1", "deepseek", "llama", "eurollm"]}


def llm_path(name, v, val=False, m="mistral"):
    """CPR-LLM probability file of a dataset / prompt version and LLM, test or validation."""
    stem = f"{m}_{name}-{v}" if name == "YELP" else f"{m}_full_{name}-{v}"
    return os.path.join(HERE, "probs", f"{stem}{'_val' if val else ''}.json")
torch.set_num_threads(int(os.environ.get("THREADS", 32)))


def rebuild(dataset):
    import prepare_large as pl
    cap = {}
    pl.save = lambda name, df, tr, te, prompt, k, render, meta: cap.update(df=df, tr=tr, te=te, prompt=prompt, k=k,
                                                                            render=render)
    getattr(pl, dataset)(SimpleNamespace(n_train=50000, n_test=2000, core=20))
    return cap


def validation_set(name, cap, n=2000, seed=1):
    """2,000 interactions of the last 5% of the training period (the early-stopping period of MF/MLP), none of them
    in the LLM's 50k training sample; written as data/<name>-H_val (split "test", test-prompt layout) so that
    evaluate_probs.py scores them. Used only to choose the hybrid baseline and weight."""
    import random
    from datasets import Dataset, DatasetDict
    import prepare_large as pl
    df, cut = cap["df"], int(len(cap["df"]) * 0.9)
    vcut = int(cut * 0.95)
    pool = sorted(set(range(vcut, cut)) - set(cap["tr"]))
    val = sorted(random.Random(seed).sample(pool, n))
    hist = pl.histories(df, val, cap["k"], cap["render"])
    ex = {v: [f"{df.at[i, 'user']}<s>[INST] {cap['prompt'](df.loc[i], hist[i] if v == 'H' else None)}[/INST]</s>"
              for i in val] for v in ("H", "WH")}
    for v in VARIANTS[name]:
        out = os.path.join(HERE, "data", f"{name}-{v}_val")
        if not os.path.exists(out):
            DatasetDict({"test": Dataset.from_dict({"example": ex[v]})}).save_to_disk(out)
    return np.array(val), ex["H"]


def check(name, df, te):
    from datasets import load_from_disk
    ex = load_from_disk(os.path.join(HERE, "data", f"{name}-H"))["test"]["example"]
    assert len(ex) == len(te)
    for e, i in zip(ex, te):
        assert e.split("<s>", 1)[0] == str(df.at[i, "user"]), i
        assert e.split("### Response:\n")[-1].strip()[0] == str(df.at[i, "label"]), i


def fit_torch(kind, u, i, r, val, nu, ni, seed=0, dim=32, max_epochs=40, patience=3):
    """MF / MLP with early stopping on `val`: returns (predict, best val RMSE, best epoch). With val=None, trains
    for exactly max_epochs (refit on the whole training part) and returns (predict, None, max_epochs)."""
    if kind in ("NFM", "DeepFM", "AutoInt"):
        return fit_neural(kind, u, i, r, val, nu, ni, seed, max_epochs, patience)
    torch.manual_seed(seed)
    mu = float(r.mean())
    if kind == "MF":
        P, Q = torch.nn.Embedding(nu, dim), torch.nn.Embedding(ni, dim)
        bu, bi = torch.nn.Embedding(nu, 1), torch.nn.Embedding(ni, 1)
        for e in (P, Q):
            torch.nn.init.normal_(e.weight, std=0.01)
        for e in (bu, bi):
            torch.nn.init.zeros_(e.weight)
        mods = torch.nn.ModuleList([P, Q, bu, bi])
        f = lambda uu, ii: mu + bu(uu)[:, 0] + bi(ii)[:, 0] + (P(uu) * Q(ii)).sum(1)
        wd = 1e-5
    else:
        P, Q = torch.nn.Embedding(nu, dim), torch.nn.Embedding(ni, dim)
        net = torch.nn.Sequential(torch.nn.Linear(2 * dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32),
                                  torch.nn.ReLU(), torch.nn.Linear(32, 1))
        mods = torch.nn.ModuleList([P, Q, net])
        f = lambda uu, ii: mu + net(torch.cat([P(uu), Q(ii)], 1))[:, 0]
        wd = 1e-6
    opt = torch.optim.Adam(mods.parameters(), lr=1e-3, weight_decay=wd)
    U, I, R = map(torch.as_tensor, (u, i, r.astype(np.float32)))
    if val is not None:
        vu, vi, vr = map(torch.as_tensor, (val[0], val[1], val[2].astype(np.float32)))
    bs = 4096 if len(r) < 2e5 else 16384
    best, best_state, bad, best_ep = np.inf, None, 0, max_epochs
    for ep in range(max_epochs):
        perm = torch.randperm(len(R))
        for s in range(0, len(R), bs):
            b = perm[s:s + bs]
            loss = ((f(U[b], I[b]) - R[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is None:
            continue
        with torch.no_grad():
            v = float(((f(vu, vi).clamp(1, 5) - vr) ** 2).mean().sqrt())
        print(f"  {kind} epoch {ep + 1}: val RMSE {v:.4f}", flush=True)
        if v < best - 1e-4:
            best, bad, best_ep, best_state = v, 0, ep + 1, {k: x.clone() for k, x in mods.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if val is not None:
        mods.load_state_dict(best_state)
    return (lambda uu, ii: f(torch.as_tensor(uu), torch.as_tensor(ii)).detach().clamp(1, 5).numpy(),
            (best if val is not None else None), best_ep)


def fit_neural(kind, u, i, r, val, nu, ni, seed=0, max_epochs=40, patience=3):
    """NFM / DeepFM / AutoInt of neural_cars.py over the fields (user, item), on the GPU when available; the
    embedding size (32 or 64) is chosen on `val`. Same return convention as fit_torch."""
    import neural_cars as nc
    X, Xv = np.stack([u, i], 1), (np.stack([val[0], val[1]], 1) if val is not None else None)
    sizes = [int(nu), int(ni)]
    mu = float(r.mean())

    def predictor(net):
        net.mu = mu
        return lambda uu, ii: nc.predict(net, np.stack([uu, ii], 1).astype(np.int64))

    if val is None:
        cfg = fit_neural.cfg[kind]
        return predictor(nc.train_net(kind, X, r, sizes, cfg, max_epochs, seed, batch=8192)), None, max_epochs
    best = None
    for dim in (32, 64):
        cfg = {"dim": dim, "wd": 1e-6, "dropout": 0.0, "lr": 1e-3}
        v, ep = nc.train_net(kind, X, r, sizes, cfg, max_epochs, seed, Xv, val[2], patience=patience, batch=8192)
        print(f"  {kind} dim {dim}: val RMSE {v:.4f} at epoch {ep}", flush=True)
        if best is None or v < best[0]:
            best = (v, ep, cfg)
    v, ep, cfg = best
    fit_neural.cfg[kind] = cfg
    net = nc.train_net(kind, X, r, sizes, cfg, ep, seed, batch=8192)  # same data, best epoch count
    return predictor(net), v, ep


fit_neural.cfg = {}


def fit(dataset):
    import pandas as pd
    name = NAMES[dataset]
    cap = rebuild(dataset)
    df, tr_sample, te = cap["df"], cap["tr"], cap["te"]
    check(name, df, te)
    print(f"{name}: split rebuilt and checked against the {len(te)} test prompts", flush=True)
    va, va_ex = validation_set(name, cap)
    cut = int(len(df) * 0.9)
    uid = {x: k for k, x in enumerate(df["user"].unique())}
    iid = {x: k for k, x in enumerate(df["item"].unique())}
    U = df["user"].map(uid).to_numpy(np.int64)
    I = df["item"].map(iid).to_numpy(np.int64)
    R = df["label"].to_numpy(np.float64)
    te = np.array(te)
    t = R[te]
    preds, vpreds = {}, {}
    vcut = int(cut * 0.95)
    from datasets import load_from_disk
    import re
    ex = load_from_disk(os.path.join(HERE, "data", f"{name}-H"))["test"]["example"]
    uar_hist = lambda exs, g: np.array([np.mean(h) if h else g for h in
                                        ([int(x) for x in re.findall(r"rated it as: (\d) out of 5", e)] for e in exs)])
    # test: statistics of the whole training part; validation: of the part before the validation period
    for out_, idx, end, exs in ((preds, te, cut, ex), (vpreds, va, vcut, va_ex)):
        train = pd.DataFrame({"u": U[:end], "i": I[:end], "r": R[:end]})
        gar = train.r.mean()
        out_["GAR"] = np.full(len(idx), gar)
        out_["IAR"] = pd.Series(I[idx]).map(train.groupby("i").r.mean()).fillna(gar).to_numpy()
        out_["UAR"] = pd.Series(U[idx]).map(train.groupby("u").r.mean()).fillna(gar).to_numpy()
        out_["UAR-hist"] = uar_hist(exs, gar)
    info = {}
    for budget, idx in (("full", np.arange(cut)), ("50k", np.array(tr_sample))):
        fit_idx, val_idx = idx[idx < vcut], idx[idx >= vcut]
        if budget == "50k":  # the 50k sample is spread over the whole training period: hold out its last 5%
            fit_idx, val_idx = idx[: int(len(idx) * 0.95)], idx[int(len(idx) * 0.95):]
        kinds = ("MF", "MLP") + (("NFM", "DeepFM", "AutoInt") if budget == "full" else ())
        for kind in kinds:
            label = {"MF": "BiasedMF"}.get(kind, kind) + ("" if budget == "full" else "-50k")
            print(f"{label}: {len(fit_idx)} fit / {len(val_idx)} validation interactions", flush=True)
            model, v, ep = fit_torch(kind, U[fit_idx], I[fit_idx], R[fit_idx], (U[val_idx], I[val_idx], R[val_idx]),
                                     len(uid), len(iid))
            vpreds[label] = model(U[va], I[va])  # validation: model that has not seen the validation period
            # test: refitted on the whole training part (incl. the validation period) for the chosen number of epochs
            refit, _, _ = fit_torch(kind, U[idx], I[idx], R[idx], None, len(uid), len(iid), max_epochs=ep)
            preds[label] = refit(U[te], I[te])
            info[label] = {"n_fit": int(len(fit_idx)), "val_rmse": v, "epochs": ep, "n_refit": int(len(idx))}
    out = {"truth": t.tolist(), "models": {k: list(map(float, v)) for k, v in preds.items()}, "info": info,
           "val_truth": R[va].tolist(), "val_models": {k: list(map(float, v)) for k, v in vpreds.items()},
           "n_train_full": cut, "test_users_unseen": int((~pd.Series(U[te]).isin(set(U[:cut]))).sum()),
           "test_items_unseen": int((~pd.Series(I[te]).isin(set(I[:cut]))).sum())}
    json.dump(out, open(os.path.join(HERE, "results", f"large_baselines_{name}.json"), "w"))
    for k, p in preds.items():
        print(f"{k:12s} RMSE {np.sqrt(np.mean((p - t) ** 2)):.4f}  MAE {np.mean(np.abs(p - t)):.4f}")


def metrics(p, t):
    from sklearn.metrics import roc_auc_score
    return np.sqrt(np.mean((p - t) ** 2)), np.mean(np.abs(p - t)), roc_auc_score(t >= 3, p)


def compare():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    report, lines = {}, []
    for name in ("MovieLens", "AmazonBooks"):  # Table 6; Yelp is in Table 5 (hybrid only)
        bp, lp = os.path.join(HERE, f"results/large_baselines_{name}.json"), llm_path(name, "H")
        if not (os.path.exists(bp) and os.path.exists(lp)):
            print(f"{name}: missing {bp if not os.path.exists(bp) else lp}")
            continue
        b, l = json.load(open(bp)), json.load(open(lp))
        t = np.array(b["truth"])
        assert np.array_equal(np.array(l["truth"], float), t), name
        llm = np.array(l["expected"])
        idx = rng.integers(0, len(t), (10000, len(t)))
        pos = t >= 3

        def boot(p):
            rm = np.sqrt(((p[idx] - t[idx]) ** 2).mean(1))
            ma = np.abs(p[idx] - t[idx]).mean(1)
            au = np.array([roc_auc_score(pos[j], p[j]) for j in idx[:2000]])  # AUC: 2,000 replicates (slow)
            return rm, ma, au
        L = boot(llm)
        res = {"CPR-LLM": dict(zip(("rmse", "mae", "auc"), map(float, metrics(llm, t))))}
        for m, p in b["models"].items():
            p = np.array(p)
            res[m] = dict(zip(("rmse", "mae", "auc"), map(float, metrics(p, t))))
            if m == "GAR":
                continue
            Bm = boot(p)
            for k, key in enumerate(("rmse", "mae", "auc")):
                d = L[k] - Bm[k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                res[m][f"llm_minus_{key}"] = [float(res["CPR-LLM"][key] - res[m][key]), float(lo), float(hi),
                                              bool(lo > 0 or hi < 0)]
        report[name] = res
        lines += [f"## {name} ({len(t)} test interactions)\n", "| method | RMSE | MAE | AUC | LLM − method: RMSE | MAE | AUC |",
                  "|---|---|---|---|---|---|---|"]
        for m, r in res.items():
            d = " | ".join(f"{r[f'llm_minus_{k}'][0]:+.4f} [{r[f'llm_minus_{k}'][1]:+.3f}, {r[f'llm_minus_{k}'][2]:+.3f}]"
                           f"{' *' if r[f'llm_minus_{k}'][3] else ''}" if f"llm_minus_{k}" in r else ""
                           for k in ("rmse", "mae", "auc"))
            lines.append(f"| {m} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['auc']:.4f} | {d} |")
        lines.append("")
    json.dump(report, open(os.path.join(HERE, "results", "table6_recomputed.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "table6_recomputed.md"), "w").write(
        "# MovieLens / Amazon-Books on the CPR-LLM split (* = 95% paired-bootstrap CI excludes 0)\n\n" + "\n".join(lines))
    print("\n".join(lines))


def hybrid():
    """CPR-LLM + classical model: w * LLM + (1 - w) * baseline. For each baseline, w in {0, 0.05, ..., 1} is
    chosen on the validation set (data/<Name>-H_val, probs/mistral_full_<Name>-H_val.json); the final pair
    (baseline, w) is the one with the lowest validation RMSE. Test numbers only reported, never used to choose."""
    from sklearn.metrics import roc_auc_score
    W = np.round(np.arange(0, 1.0001, 0.05), 2)
    rng = np.random.default_rng(0)
    report, lines = {}, []
    for name, v, lm in ((n, v, lm) for n in NAMES.values() for v in VARIANTS[n] for lm in LLMS.get(n, ["mistral"])):
        key = f"{name}-{v}" + ("" if lm == "mistral" else f"-{lm}")  # Mistral keys unchanged (table6_rows.py)
        paths = [os.path.join(HERE, f"results/large_baselines_{name}.json"), llm_path(name, v, m=lm), llm_path(name, v, True, lm)]
        if not all(os.path.exists(p) for p in paths):
            print(f"{key}: missing {[p for p in paths if not os.path.exists(p)]}")
            continue
        b = json.load(open(paths[0]))
        if "val_truth" not in b:
            print(f"{name}-{v}: baselines have no validation predictions (re-run fit)")
            continue
        lt, lv = (json.load(open(p)) for p in paths[1:])
        t, tv = np.array(b["truth"]), np.array(b["val_truth"])
        assert np.array_equal(np.array(lt["truth"], float), t) and np.array_equal(np.array(lv["truth"], float), tv)
        llm, llm_v = np.array(lt["expected"]), np.array(lv["expected"])
        rm = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
        cands = {}
        for m in b["models"]:
            bv, bt = np.array(b["val_models"][m]), np.array(b["models"][m])
            w = min(W, key=lambda w: rm(w * llm_v + (1 - w) * bv, tv))
            cands[m] = dict(w=float(w), val_rmse=rm(w * llm_v + (1 - w) * bv, tv), test=w * llm + (1 - w) * bt)
        chosen = min(cands, key=lambda m: cands[m]["val_rmse"])
        best_b = min((m for m in b["models"] if m != "GAR"), key=lambda m: rm(np.array(b["val_models"][m]), tv))
        idx = rng.integers(0, len(t), (10000, len(t)))

        def diff(a, c):
            out = {}
            for key, f in (("rmse", lambda p, y: np.sqrt(((p - y) ** 2).mean(-1))), ("mae", lambda p, y: np.abs(p - y).mean(-1))):
                d = f(a[idx], t[idx]) - f(c[idx], t[idx])
                lo, hi = np.percentile(d, [2.5, 97.5])
                out[key] = [float(f(a, t) - f(c, t)), float(lo), float(hi), bool(lo > 0 or hi < 0)]
            d = np.array([roc_auc_score(t[j] >= 3, a[j]) - roc_auc_score(t[j] >= 3, c[j]) for j in idx[:2000]])
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["auc"] = [float(roc_auc_score(t >= 3, a) - roc_auc_score(t >= 3, c)), float(lo), float(hi), bool(lo > 0 or hi < 0)]
            return out
        h = cands[chosen]["test"]
        os.makedirs(os.path.join(HERE, "probs", "hybrid"), exist_ok=True)  # per-prompt fused test predictions (Table 5)
        json.dump({"truth": t.tolist(), "expected": h.tolist(), "baseline": chosen, "w": cands[chosen]["w"]},
                  open(os.path.join(HERE, "probs", "hybrid", f"{lm}_hycpr_{name}-{v}.json"), "w"))
        res = {"chosen": chosen, "w": cands[chosen]["w"], "best_val_baseline": best_b,
               "test": dict(zip(("rmse", "mae", "auc"), map(float, metrics(h, t)))),
               "llm": dict(zip(("rmse", "mae", "auc"), map(float, metrics(llm, t)))),
               "vs_llm": diff(h, llm), "vs_best_baseline": diff(h, np.array(b["models"][best_b])),
               "candidates": {m: {"w": c["w"], "val_rmse": c["val_rmse"],
                                  **dict(zip(("test_rmse", "test_mae", "test_auc"), map(float, metrics(c["test"], t))))}
                              for m, c in cands.items()}}
        report[key] = res
        g = lambda p: ", ".join(f"{k.upper()} {v[0]:+.4f} [{v[1]:+.3f}, {v[2]:+.3f}]{' *' if v[3] else ''}" for k, v in p.items())
        lines += [f"## {key}\n", "| baseline | w (val) | val RMSE | test RMSE | test MAE | test AUC |", "|---|---|---|---|---|---|"]
        lines += [f"| {m}{' **chosen**' if m == chosen else ''} | {c['w']} | {c['val_rmse']:.4f} | {c['test_rmse']:.4f} | "
                  f"{c['test_mae']:.4f} | {c['test_auc']:.4f} |" for m, c in res["candidates"].items()]
        lines += ["", f"CPR-LLM alone: RMSE {res['llm']['rmse']:.4f}, MAE {res['llm']['mae']:.4f}, AUC {res['llm']['auc']:.4f}", "",
                  f"chosen hybrid − CPR-LLM alone: {g(res['vs_llm'])}", "",
                  f"chosen hybrid − best baseline on validation ({best_b}): {g(res['vs_best_baseline'])}", ""]
    json.dump(report, open(os.path.join(HERE, "results", "table6_hybrid.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "table6_hybrid.md"), "w").write(
        "# CPR-LLM + classical model on MovieLens / Amazon-Books / Yelp (w and baseline chosen on validation; "
        "* = 95% paired-bootstrap CI excludes 0)\n\n" + "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    {"fit": lambda: fit(sys.argv[2]), "compare": compare, "hybrid": hybrid}[sys.argv[1]]()
