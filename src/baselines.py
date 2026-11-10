"""Traditional and context-aware rating-prediction baselines of Table 5, in numpy.

The paper's baseline rows were produced outside this workspace (most likely with CARSKit) and no code or
outputs survive, so these are re-implementations of the standard definitions:

  ItemKNN, UserKNN  neighbourhood CF on mean-centred ratings, cosine similarity, k nearest raters
  BiasedMF          r = mu + b_u + b_i + p_u . q_i
  SVD++             r = mu + b_u + b_i + q_i . (p_u + |N(u)|^-1/2 sum_{j in N(u)} y_j)
  FM                2-way factorization machine over one-hot user, item and context conditions
  CAMF_C            r = mu + b_u + b_i + sum_c B_c + p_u . q_i                 (Baltrunas et al. 2011)
  CAMF_CI           r = mu + b_u + sum_c B_{i,c} + p_u . q_i
  CAMF_CU           r = mu + b_i + sum_c B_{u,c} + p_u . q_i
  CAMF_CUCI         r = mu + sum_c B_{u,c} + sum_c B_{i,c} + p_u . q_i

All are trained with SGD on the training split with fixed hyperparameters (no tuning on the test split)
and predictions are clipped to 1..5.
"""
import os
import numpy as np
import pandas as pd

FIRSTEVAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "small_data")

# (train file, test file, user col, item col, rating col, context columns) relative to FirstEval/Datasets
DATASETS = {
    "LDOS-WH": ("LDOSInitial/train_data.xlsx", "LDOSInitial/test_data.xlsx", "userID", "ItemID", "rating",
                ["time", "daytype", "season", "location", "weather", "social", "endEmo", "dominantEmo",
                 "mood", "physical", "decision", "interaction"]),
    "Depaul-WH": ("DepaulMoviexlsx/Depaul_Movies_Train.xlsx", "DepaulMoviexlsx/Depaul_Movies_Test.xlsx",
                  "userid", "itemid", "rating", ["Time", "Location", "Companion"]),
    "Tijuana-WH": ("TijuanaWithAlpaca/train_data.xlsx", "TijuanaWithAlpaca/test_data.xlsx", "UserID",
                   "RESTAURANT", "RATING", ["Context"]),
}


def load(setting):
    """Train and test frames with integer columns u, i, r and one column per context dimension."""
    tr_f, te_f, u, i, r, ctx = DATASETS[setting]
    tr = pd.read_excel(f"{FIRSTEVAL}/Datasets/{tr_f}")
    te = pd.read_excel(f"{FIRSTEVAL}/Datasets/{te_f}")
    if setting == "Tijuana-WH":  # "Weekday at Work" -> day type and place
        for d in (tr, te):
            parts = d["Context"].astype(str).str.split(" at ", n=1, expand=True)
            d["daytype"], d["place"] = parts[0], parts[1]
        ctx = ["daytype", "place"]
    out = []
    for d in (tr, te):
        d = d.dropna(subset=[u, i, r]).copy()
        d = pd.DataFrame({"u": d[u].astype(str), "i": d[i].astype(str), "r": d[r].astype(float),
                          **{c: d[c].astype(str).fillna("NA") for c in ctx}})
        out.append(d)
    return out[0], out[1], ctx


class Index:
    """Maps users, items and context conditions to integer ids; unseen test values get id -1."""

    def __init__(self, train, ctx):
        self.users = {v: k for k, v in enumerate(train["u"].unique())}
        self.items = {v: k for k, v in enumerate(train["i"].unique())}
        conds = sorted({(c, v) for c in ctx for v in train[c].unique()})
        self.conds = {cv: k for k, cv in enumerate(conds)}
        self.ctx = ctx

    def encode(self, d):
        u = d["u"].map(self.users).fillna(-1).astype(int).to_numpy()
        i = d["i"].map(self.items).fillna(-1).astype(int).to_numpy()
        c = np.stack([d[c].map(lambda v, c=c: self.conds.get((c, v), -1)).to_numpy() for c in self.ctx], 1)
        return u, i, c.astype(int), d["r"].to_numpy()


def knn(train, test, ctx, by="item", k=20):
    """ItemKNN (by='item') or UserKNN (by='user'): mu + b + weighted mean-centred neighbour ratings."""
    idx = Index(train, ctx)
    u, i, _, r = idx.encode(train)
    nu, ni = len(idx.users), len(idx.items)
    R = np.full((nu, ni), np.nan)
    R[u, i] = r  # duplicate (user, item) pairs under different contexts: last one wins
    if by == "user":
        R = R.T  # rows: items, columns: users; neighbours are users
    mu = np.nanmean(R)
    col_mean = np.nanmean(R, 0)  # every column has at least one training rating
    C = np.nan_to_num(R - col_mean)  # centre each neighbour column
    norms = np.linalg.norm(C, axis=0) + 1e-9
    S = (C.T @ C) / np.outer(norms, norms)
    tu, ti, _, _ = idx.encode(test)
    rows, cols = (tu, ti) if by == "item" else (ti, tu)
    preds = []
    for a, b in zip(rows, cols):  # a: the row entity (user for ItemKNN), b: the target column
        if a < 0 or b < 0:
            preds.append(mu if b < 0 else col_mean[b])
            continue
        rated = np.where(~np.isnan(R[a]))[0]
        rated = rated[rated != b]
        if len(rated) == 0:
            preds.append(col_mean[b])
            continue
        sims = S[b, rated]
        top = rated[np.argsort(-sims)[:k]]
        w = S[b, top]
        dev = R[a, top] - col_mean[top]
        preds.append(col_mean[b] + (w @ dev) / (np.abs(w).sum() + 1e-9))
    return np.clip(np.array(preds), 1, 5)


def factor_model(train, test, ctx, kind, f=10, lr=0.01, reg=0.02, epochs=100, seed=0):
    """SGD-trained BiasedMF, SVD++, FM or CAMF_{C,CI,CU,CUCI}; see the module docstring."""
    rng = np.random.default_rng(seed)
    idx = Index(train, ctx)
    u, i, c, r = idx.encode(train)
    nu, ni, nc = len(idx.users), len(idx.items), len(idx.conds)
    mu = r.mean()
    bu, bi, bc = np.zeros(nu), np.zeros(ni), np.zeros(nc)
    Bic, Buc = np.zeros((ni, nc)), np.zeros((nu, nc))
    P, Q = rng.normal(0, 0.1, (nu, f)), rng.normal(0, 0.1, (ni, f))
    Y = rng.normal(0, 0.1, (ni, f))  # SVD++ implicit factors
    V = rng.normal(0, 0.1, (nu + ni + nc, f))  # FM factors over [users | items | conditions]
    w = np.zeros(nu + ni + nc)
    N = {uu: i[u == uu] for uu in range(nu)}  # items rated by each user (SVD++)

    def feats(uu, ii, cc):
        return [x for x in ([uu] if uu >= 0 else []) + ([nu + ii] if ii >= 0 else [])
                + [nu + ni + x for x in cc if x >= 0]]

    def predict(uu, ii, cc):
        cc = [x for x in cc if x >= 0]
        if kind == "FM":
            fs = feats(uu, ii, cc)
            s = V[fs].sum(0)
            return mu + w[fs].sum() + 0.5 * (s @ s - (V[fs] ** 2).sum())
        pu = P[uu] if uu >= 0 else np.zeros(f)
        if kind == "SVD++" and uu >= 0 and len(N[uu]):
            pu = pu + Y[N[uu]].sum(0) / np.sqrt(len(N[uu]))
        dot = pu @ Q[ii] if ii >= 0 else 0.0
        b_u = bu[uu] if uu >= 0 else 0.0
        b_i = bi[ii] if ii >= 0 else 0.0
        if kind in ("BiasedMF", "SVD++"):
            return mu + b_u + b_i + dot
        if kind == "CAMF_C":
            return mu + b_u + b_i + bc[cc].sum() + dot
        if kind == "CAMF_CI":
            return mu + b_u + (Bic[ii, cc].sum() if ii >= 0 else 0.0) + dot
        if kind == "CAMF_CU":
            return mu + b_i + (Buc[uu, cc].sum() if uu >= 0 else 0.0) + dot
        if kind == "CAMF_CUCI":
            return mu + (Buc[uu, cc].sum() if uu >= 0 else 0.0) + (Bic[ii, cc].sum() if ii >= 0 else 0.0) + dot
        raise ValueError(kind)

    order = np.arange(len(r))
    for _ in range(epochs):
        rng.shuffle(order)
        for n in order:
            uu, ii, cc = u[n], i[n], [x for x in c[n] if x >= 0]
            e = r[n] - predict(uu, ii, cc)
            if kind == "FM":
                fs = feats(uu, ii, cc)
                s = V[fs].sum(0)
                w[fs] += lr * (e - reg * w[fs])
                V[fs] += lr * (e * (s - V[fs]) - reg * V[fs])
                continue
            pu = P[uu].copy()
            if kind == "SVD++" and len(N[uu]):
                imp = Y[N[uu]].sum(0) / np.sqrt(len(N[uu]))
                P[uu] += lr * (e * Q[ii] - reg * P[uu])
                Y[N[uu]] += lr * (e * Q[ii] / np.sqrt(len(N[uu])) - reg * Y[N[uu]])
                Q[ii] += lr * (e * (pu + imp) - reg * Q[ii])
            else:
                P[uu] += lr * (e * Q[ii] - reg * P[uu])
                Q[ii] += lr * (e * pu - reg * Q[ii])
            if kind in ("BiasedMF", "SVD++", "CAMF_C", "CAMF_CI"):
                bu[uu] += lr * (e - reg * bu[uu])
            if kind in ("BiasedMF", "SVD++", "CAMF_C", "CAMF_CU"):
                bi[ii] += lr * (e - reg * bi[ii])
            if kind == "CAMF_C":
                bc[cc] += lr * (e - reg * bc[cc])
            if kind in ("CAMF_CI", "CAMF_CUCI"):
                Bic[ii, cc] += lr * (e - reg * Bic[ii, cc])
            if kind in ("CAMF_CU", "CAMF_CUCI"):
                Buc[uu, cc] += lr * (e - reg * Buc[uu, cc])
    tu, ti, tc, _ = idx.encode(test)
    return np.clip(np.array([predict(a, b, list(cc)) for a, b, cc in zip(tu, ti, tc)]), 1, 5)


MODELS = ["ItemKNN", "UserKNN", "SVD++", "BiasedMF", "FM", "CAMF_CI", "CAMF_CU", "CAMF_C", "CAMF_CUCI"]


def run(model, setting):
    """(MAE, RMSE, n_test) of `model` on the structured train/test split of `setting`."""
    train, test, ctx = load(setting)
    if model in ("ItemKNN", "UserKNN"):
        pred = knn(train, test, ctx, by="item" if model == "ItemKNN" else "user")
    else:
        pred = factor_model(train, test, ctx, model)
    err = test["r"].to_numpy() - pred
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean())), len(test)
