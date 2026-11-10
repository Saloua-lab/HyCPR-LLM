"""Classical baselines on exactly the train/test interactions of the CPR-LLM prompts (reviewer remark on
Table 4: the original baseline rows used their own splits).

Small CARS datasets (LDOS, DePaul, Tijuana): the interaction tables are rebuilt from the prompt DatasetDicts
(prompt_table.py), so the test set is the LLM test set (452 / 684 / 250 interactions; the -WH and -H settings
share the same test interactions) and the context is the one shown in the prompt. Test prompts carry the user id;
training prompts do not, so it is recovered from the structured source files (LDOS, Tijuana: unique match on item,
context, rating and the user attributes in the prompt; DePaul: the xlsx files are row-aligned with the prompts).
Training prompts whose user cannot be identified are dropped (reported).

Yelp: the full chronological split of prepare_large.py is rebuilt from the raw review file (seed 0), checked
against the 2,000 test prompts, and the baselines are fitted on the whole 90% training part, as in Kang et al.
(the LLM itself saw 50,000 sampled training prompts; --yelp-train sample fits them on that sample instead).

Usage (from recomputed/):
  python baselines_same_split.py small > results/baselines_small.json
  python baselines_same_split.py yelp [--yelp-train full|sample] > results/baselines_yelp.json
  python baselines_same_split.py neural-small > results/baselines_neural_small.json
  python baselines_same_split.py neural-yelp > results/baselines_neural_yelp.json
Every output holds per-interaction predictions so that bootstrap intervals can be computed (bootstrap.py).
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_from_disk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prompt_table import parse  # noqa: E402

FE = os.path.join(HERE, "..", "small_data")
XLSX = os.path.join(FE, "Datasets")

SMALL = {  # name: (prompt DatasetDict used for the -WH setting, structured source)
    "LDOS": (os.path.join(FE, "LocationLDOSWithoutHis"), "LDOS"),
    "Depaul": (os.path.join(FE, "Datasets", "DepaulNoHistory"), "Depaul"),
    "Tijuana": (os.path.join(FE, "LocationTRestaurantWithoutHis"), "Tijuana"),
}


def user_info(example):
    m = re.search(r"### User Information:\n(.*?)\n\n", example, re.S)
    return m.group(1).strip() if m else ""


def item_name(details):
    return details[len("Name: "):] if details.startswith("Name: ") else details


def recover_users(path, source, keep_unknown=False):
    """Train and test interaction tables of a prompt dataset, with user ids on the training rows. Training rows
    whose user cannot be identified are dropped, or kept with u=None when keep_unknown (rows then align with the
    dataset)."""
    d = load_from_disk(path)
    tr_ex, te_ex = d["train"]["example"], d["test"]["example"]
    train, test = [parse(e) for e in tr_ex], [parse(e) for e in te_ex]
    if source == "Depaul":
        x = pd.read_excel(os.path.join(XLSX, "DepaulMoviexlsx/Depaul_Movies_Train.xlsx"))
        assert len(x) == len(train)
        for p, (_, r) in zip(train, x.iterrows()):
            assert item_name(p["i"]).startswith(f"{r.movie_name},") and int(p["r"]) == r.rating
            p["u"] = str(r.userid)
        return pd.DataFrame(train), pd.DataFrame(test), 0
    if source == "LDOS":
        files, U, I, R = ("LDOSInitial/train_data.xlsx", "LDOSInitial/test_data.xlsx"), "userID", "Movie_Name", "rating"
        cmap = {"c_time": "time", "c_daytype": "daytype", "c_season": "season", "c_location": "location",
                "c_weather": "weather"}
        uinfo = lambda r: f"age: {r['age']}, sex: {r['sex']}"
    else:
        files, U, I, R = ("TijuanaWithAlpaca/train_data.xlsx", "TijuanaWithAlpaca/test_data.xlsx"), "UserID", "RESTAURANT", "RATING"
        cmap = {"c_Context": "Context"}
        uinfo = None  # occupation/preferences text: matched through the users of the test prompts
    full = pd.concat([pd.read_excel(os.path.join(XLSX, f)) for f in files], ignore_index=True)
    full[I] = full[I].astype(str).str.strip()
    idx = defaultdict(list)
    for _, r in full.iterrows():
        idx[tuple(str(r[c]) for c in cmap.values()) + (str(int(r[R])),)].append(r)
    info_users = defaultdict(set)
    for p, e in zip(test, te_ex):
        info_users[user_info(e)].add(p["u"])
    dropped = 0
    for p, e in zip(train, tr_ex):
        sig = tuple(str(p.get(c, "nan")) for c in cmap) + (str(int(p["r"])),)
        name = item_name(p["i"])
        cands = [r for r in idx.get(sig, []) if name == r[I] or name.startswith(r[I] + ",")]
        info = user_info(e)
        if uinfo is not None:
            users = {str(r[U]) for r in cands if uinfo(r) == info}
        else:
            users = {str(r[U]) for r in cands} & info_users.get(info, set()) or {str(r[U]) for r in cands}
        if len(users) == 1:
            p["u"] = users.pop()
        else:
            p["u"] = None
            dropped += 1
    train = pd.DataFrame([p for p in train if keep_unknown or p["u"] is not None])
    return train, pd.DataFrame(test), dropped


def heuristics(train, test):
    gar = train["r"].mean()
    iar = test["i"].map(train.groupby("i")["r"].mean()).fillna(gar).to_numpy()
    uar = test["u"].map(train.groupby("u")["r"].mean()).fillna(gar).to_numpy()
    return {"GAR": np.full(len(test), gar), "IAR": iar, "UAR": uar}


def scores(truth, pred):
    e = np.asarray(pred, float) - np.asarray(truth, float)
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def run_small():
    from baselines import MODELS, factor_model, knn
    out = {}
    for name, (path, source) in SMALL.items():
        train, test, dropped = recover_users(path, source)
        ctx = sorted(c for c in set(train.columns) | set(test.columns) if c.startswith("c_"))
        for d in (train, test):
            for c in ctx:
                d[c] = d[c].fillna("NA").astype(str) if c in d else "NA"
            d["u"], d["i"] = d["u"].astype(str), d["i"].astype(str)
        preds = heuristics(train, test)
        for m in MODELS:
            preds[m] = (knn(train, test, ctx, by="item" if m == "ItemKNN" else "user") if m.endswith("KNN")
                        else factor_model(train, test, ctx, m))
        truth = test["r"].tolist()
        out[name] = {"n_train": len(train), "n_train_dropped": dropped, "n_test": len(test), "context": ctx,
                     "truth": truth,
                     "models": {m: {"pred": np.asarray(p, float).tolist(), "mae": scores(truth, p)[0],
                                    "rmse": scores(truth, p)[1]} for m, p in preds.items()}}
        print(name, len(train), "train (dropped", dropped, ")", len(test), "test", file=sys.stderr)
        for m, v in out[name]["models"].items():
            print(f"  {m:10s} {v['mae']:.3f} {v['rmse']:.3f}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------------------------- Yelp

def yelp_split():
    """(df sorted by time, train_idx sample, test_idx sample, cut) exactly as prepare_large.py builds YELP."""
    import prepare_large as pl
    cache = os.path.join(HERE, "data", "YELP_interactions.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        rows = []
        with open(os.path.join(HERE, "..", "raw_data", "yelp", "yelp_academic_dataset_review.json")) as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["user_id"], r["business_id"], int(r["stars"]), r["date"]))
        df = pd.DataFrame(rows, columns=["user", "item", "label", "ts"])
        df["ts"] = pd.to_datetime(df["ts"])
        df = pl.k_core(df, 20)
        df.to_parquet(cache)
    df, tr, te = pl.split_and_sample(df, 50000, 2000)
    return df, tr, te, int(len(df) * 0.9)


def biased_mf(train, test_u, test_i, f=32, epochs=20, reg=0.05, lr=0.01, seed=0, val=None):
    """Biased MF r = mu + b_u + b_i + p_u.q_i, mini-batch Adam with L2 on the batch's parameters (torch, CPU).
    With `val` = (u, i, r) returns the validation RMSE per epoch instead of test predictions."""
    import torch
    torch.manual_seed(seed)
    u = torch.tensor(train[0]); i = torch.tensor(train[1]); r = torch.tensor(train[2], dtype=torch.float32)
    nu, ni = int(max(u.max(), test_u.max()) + 1), int(max(i.max(), test_i.max()) + 1)
    mu = r.mean()
    P = torch.nn.Embedding(nu, f); Q = torch.nn.Embedding(ni, f)
    bu = torch.nn.Embedding(nu, 1); bi = torch.nn.Embedding(ni, 1)
    for e in (P, Q):
        torch.nn.init.normal_(e.weight, 0, 0.01)
    for e in (bu, bi):
        torch.nn.init.zeros_(e.weight)
    params = [P.weight, Q.weight, bu.weight, bi.weight]
    opt = torch.optim.Adam(params, lr=lr)

    def pred(uu, ii):
        return mu + bu(uu).squeeze(-1) + bi(ii).squeeze(-1) + (P(uu) * Q(ii)).sum(-1)

    hist = []
    for _ in range(epochs):
        perm = torch.randperm(len(r))
        for k in range(0, len(r), 8192):
            b = perm[k:k + 8192]
            pu, qi = P(u[b]), Q(i[b])
            loss = ((pred(u[b], i[b]) - r[b]) ** 2).mean() + reg * ((pu ** 2).sum(-1) + (qi ** 2).sum(-1)
                                                                     + bu(u[b]).squeeze(-1) ** 2
                                                                     + bi(i[b]).squeeze(-1) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if val is not None:
            with torch.no_grad():
                pv = pred(torch.tensor(val[0]), torch.tensor(val[1])).clamp(1, 5).numpy()
            hist.append(float(np.sqrt(((pv - val[2]) ** 2).mean())))
    if val is not None:
        return hist
    with torch.no_grad():
        return pred(torch.tensor(test_u), torch.tensor(test_i)).clamp(1, 5).numpy()


def run_yelp(train_mode):
    df, tr_idx, te_idx, cut = yelp_split()
    # check against the saved prompts: user id prefix and label of every test prompt
    ex = load_from_disk(os.path.join(HERE, "data", "YELP-WH"))["test"]["example"]
    pr = [parse(e) for e in ex]
    assert [p["u"] for p in pr] == df.loc[te_idx, "user"].tolist()
    assert [int(p["r"]) for p in pr] == df.loc[te_idx, "label"].tolist()
    train_rows = np.arange(cut) if train_mode == "full" else np.array(tr_idx)
    tr = df.iloc[train_rows]
    te = df.loc[te_idx]
    uid = {u: k for k, u in enumerate(pd.unique(df["user"]))}
    iid = {i: k for k, i in enumerate(pd.unique(df["item"]))}
    T = (tr["user"].map(uid).to_numpy(), tr["item"].map(iid).to_numpy(), tr["label"].to_numpy(float))
    tu, ti = te["user"].map(uid).to_numpy(), te["item"].map(iid).to_numpy()
    preds = heuristics(tr.rename(columns={"user": "u", "item": "i", "label": "r"}),
                       te.rename(columns={"user": "u", "item": "i", "label": "r"}))
    # hyperparameters chosen on a chronological validation part (last 5% of the training rows), not on test
    v0 = int(len(train_rows) * 0.95)
    fit = tuple(a[:v0] for a in T); val = tuple(a[v0:] for a in T)
    best = None
    for reg in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
        h = biased_mf(fit, tu, ti, reg=reg, val=val, epochs=20)
        ep = int(np.argmin(h)) + 1
        print(f"BiasedMF reg {reg}: best val RMSE {min(h):.4f} at epoch {ep}", file=sys.stderr)
        if best is None or min(h) < best[0]:
            best = (min(h), reg, ep)
    _, reg, ep = best
    preds["BiasedMF"] = biased_mf(T, tu, ti, reg=reg, epochs=ep)
    truth = te["label"].tolist()
    out = {"YELP": {"train_mode": train_mode, "n_train": len(train_rows), "n_test": len(te),
                    "mf_hyper": {"reg": reg, "epochs": ep, "f": 32}, "truth": truth,
                    "models": {m: {"pred": np.asarray(p, float).tolist(), "mae": scores(truth, p)[0],
                                   "rmse": scores(truth, p)[1]} for m, p in preds.items()}}}
    for m, v in out["YELP"]["models"].items():
        print(f"  {m:10s} {v['mae']:.3f} {v['rmse']:.3f}", file=sys.stderr)
    return out


# ------------------------------------------------------------------------------- neural CARS baselines

def _pack(truth, preds, extra):
    return {**extra, "truth": truth,
            "models": {m: {"pred": np.asarray(p, float).tolist(), "mae": scores(truth, p)[0],
                           "rmse": scores(truth, p)[1], "hyper": h} for m, (p, h) in preds.items()}}


def run_neural_small():
    """NFM, DeepFM, AutoInt (neural_cars.py) on the same small-dataset splits and context as run_small()."""
    import neural_cars as nc
    out = {}
    for name, (path, source) in SMALL.items():
        train, test, dropped = recover_users(path, source)
        ctx = sorted(c for c in set(train.columns) | set(test.columns) if c.startswith("c_"))
        for d in (train, test):
            for c in ctx:
                d[c] = d[c].fillna("NA").astype(str) if c in d else "NA"
            d["u"], d["i"] = d["u"].astype(str), d["i"].astype(str)
        print(name, file=sys.stderr)
        preds = {m: nc.fit_predict(train, test, ["u", "i"] + ctx, m) for m in nc.MODELS}
        truth = test["r"].tolist()
        out[name] = _pack(truth, preds, {"n_train": len(train), "n_test": len(test), "context": ctx})
        for m, v in out[name]["models"].items():
            print(f"  {m:10s} {v['mae']:.3f} {v['rmse']:.3f}", file=sys.stderr)
    return out


def run_neural_yelp():
    """The same models on Yelp, fitted on the whole 90% training part; the Yelp prompts carry no context, so the
    fields are the user and the item only."""
    import neural_cars as nc
    df, tr_idx, te_idx, cut = yelp_split()
    tr = df.iloc[np.arange(cut)].rename(columns={"user": "u", "item": "i", "label": "r"})
    te = df.loc[te_idx].rename(columns={"user": "u", "item": "i", "label": "r"})
    grid = {"dim": [16, 32, 64], "wd": [0.0, 1e-6, 1e-5], "dropout": [0.0, 0.3], "lr": [1e-3]}
    preds = {m: nc.fit_predict(tr, te, ["u", "i"], m, grid=grid, max_epochs=30, batch=8192, seeds=(0, 1, 2),
                               val_frac=0.05) for m in nc.MODELS}
    truth = te["r"].tolist()
    out = {"YELP": _pack(truth, preds, {"n_train": len(tr), "n_test": len(te)})}
    for m, v in out["YELP"]["models"].items():
        print(f"  {m:10s} {v['mae']:.3f} {v['rmse']:.3f}", file=sys.stderr)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["small", "yelp", "neural-small", "neural-yelp"])
    ap.add_argument("--yelp-train", choices=["full", "sample"], default="full")
    a = ap.parse_args()
    run = {"small": run_small, "yelp": lambda: run_yelp(a.yelp_train), "neural-small": run_neural_small,
           "neural-yelp": run_neural_yelp}[a.what]
    json.dump(run(), sys.stdout)
