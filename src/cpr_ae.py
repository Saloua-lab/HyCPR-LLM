"""Re-implementation of the conformal context-aware recommender with deep autoencoders (Zammali, Dutta & Ben Yahia,
WI-IAT 2024, "Enhancing the conformal predictability of context-aware recommendation systems by using Deep
Autoencoders"), for a head-to-head comparison (review 6: "conformal rating prediction in context-aware settings (CPR
with autoencoders) not clearly compared in depth").

Model (neural contextual matrix factorization + autoencoder, our implementation of the published description):
  * an autoencoder encodes the user's training rating vector over the items (centred by the global mean, 0 if
    unrated; the target item of a training example is masked out of its own input, so that training matches test)
    into a code z_u (sigmoid layer), and a linear decoder reconstructs the observed ratings (masked MSE);
  * the rating is mu + MLP([z_u, e_u, e_i, sum_c e_c]) with embeddings of the user, the item and every context
    variable of the prompt (contextual neural MF);
  * loss = MSE(rating) + lambda * reconstruction; Adam.
Model selection exactly as for NFM / DeepFM / AutoInt (neural_cars.py): grid over the code / embedding size
{16, 32, 64}, weight decay {1e-4, 1e-3, 1e-2}, dropout {0, 0.3}, lambda {0.1, 1}, early stopping on a random 10% of
the training interactions, retraining on all of them for the selected number of epochs, test predictions averaged over
5 seeds. Training / test interactions: those of the CF baselines (baselines_same_split.recover_users).
CPR: split conformal prediction with the absolute residual (30% calibration, 1,000 splits, eps = 0.1), as in the paper.
Compared with HyCPR-LLM (Mistral, -H) and the best baseline: MAE / RMSE with paired bootstrap, interval width,
admissible ratings, coverage.

Usage: python cpr_ae.py → results/cpr_ae.{json,md}, probs/cprae_<setting>.json
"""
import itertools
import json
import sys

import numpy as np
import torch
from torch import nn

from cp_adaptive import K, q
from holm import test

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = {"dim": [16, 32, 64], "wd": [1e-4, 1e-3, 1e-2], "dropout": [0.0, 0.3], "lam": [0.1, 1.0], "lr": [1e-3]}


class AENCMF(nn.Module):
    def __init__(self, n_items, sizes, dim, dropout):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_items, dim), nn.Sigmoid())
        self.dec = nn.Linear(dim, n_items)
        self.offsets = torch.tensor(np.concatenate([[0], np.cumsum(sizes)[:-1]]), dtype=torch.long)
        self.emb = nn.Embedding(int(sum(sizes)), dim)
        nn.init.normal_(self.emb.weight, 0, 0.01)
        self.mlp = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, R_in, x):
        z = self.enc(R_in)
        e = self.emb(x + self.offsets.to(x.device))           # fields: user, item, contexts...
        h = torch.cat([z, e[:, 0], e[:, 1], e[:, 2:].sum(1)], 1)
        return self.mlp(h).squeeze(-1), self.dec(z)


def encode(train, frames, ctx):
    users = {u: k + 1 for k, u in enumerate(sorted(set(train.u)))}
    items = {i: k for k, i in enumerate(sorted(set(train.i)))}
    cmaps = [{v: k + 1 for k, v in enumerate(sorted(set(map(str, train[c]))))} for c in ctx]
    sizes = [len(users) + 1, len(items) + 1] + [len(m) + 1 for m in cmaps]
    mu = float(train.r.mean())
    R = np.zeros((len(users) + 1, len(items)), np.float32)
    M = np.zeros_like(R)
    g = train.groupby(["u", "i"]).r.mean()
    for (u, i), r in g.items():
        R[users[u], items[i]] = r - mu
        M[users[u], items[i]] = 1
    out = []
    for f in frames:
        X = np.stack([f.u.map(users).fillna(0).astype(int).to_numpy(), f.i.map(items).fillna(-1).astype(int).to_numpy() + 1]
                     + [f[c].map(str).map(m).fillna(0).astype(int).to_numpy() for c, m in zip(ctx, cmaps)], 1)
        out.append(X)
    return R, M, sizes, mu, out


def train_net(R, M, sizes, mu, X, y, cfg, epochs, seed, Xv=None, yv=None, patience=20, batch=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = AENCMF(R.shape[1], sizes, cfg["dim"], cfg["dropout"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Rt, Mt = torch.tensor(R, device=DEV), torch.tensor(M, device=DEV)
    Xt, yt = torch.tensor(X, device=DEV), torch.tensor(y - mu, dtype=torch.float32, device=DEV)
    best, best_ep, bad = np.inf, 0, 0
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(yt), device=DEV)
        for k in range(0, len(yt), batch):
            b = perm[k:k + batch]
            u, it = Xt[b, 0], Xt[b, 1] - 1
            Rin, Mb = Rt[u].clone(), Mt[u].clone()
            ok = it >= 0
            Rin[ok.nonzero().squeeze(-1), it[ok]] = 0          # mask the target rating out of its own input
            Mb[ok.nonzero().squeeze(-1), it[ok]] = 0
            pred, rec = net(Rin, Xt[b])
            loss = ((pred - yt[b]) ** 2).mean() + cfg["lam"] * (((rec - Rin) ** 2) * Mb).sum() / Mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xv is None:
            continue
        pv = predict(net, Rt, Xv, mu)
        rmse = float(np.sqrt(((pv - yv) ** 2).mean()))
        if rmse < best - 1e-5:
            best, best_ep, bad = rmse, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return (best, best_ep) if Xv is not None else net


def predict(net, Rt, X, mu):
    net.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, device=DEV)
        p, _ = net(Rt[Xt[:, 0]], Xt)
    return (p.cpu().numpy() + mu).clip(1, 5)


def fit_predict(train, test_df, ctx, max_epochs=1000, seeds=(0, 1, 2, 3, 4), log=sys.stderr):
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train))
    nv = max(1, int(len(train) * 0.1))
    fit, val = train.iloc[perm[nv:]], train.iloc[perm[:nv]]
    R, M, sizes, mu, (Xf, Xv) = encode(fit, [fit, val], ctx)
    best = None
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID.keys(), vals))
        r, ep = train_net(R, M, sizes, mu, Xf, fit.r.to_numpy(float), cfg, max_epochs, 0, Xv, val.r.to_numpy(float))
        if best is None or r < best[0]:
            best = (r, ep, cfg)
    r, ep, cfg = best
    print(f"  CPR-AE: val RMSE {r:.4f}, {ep} epochs, {cfg}", file=log, flush=True)
    R, M, sizes, mu, (X, Xte) = encode(train, [train, test_df], ctx)
    Rt = torch.tensor(R, device=DEV)
    preds = [predict(train_net(R, M, sizes, mu, X, train.r.to_numpy(float), cfg, ep, s), Rt, Xte, mu) for s in seeds]
    return np.mean(preds, 0), {"val_rmse": r, "epochs": ep, **cfg}


def split_cp(t, p, eps=0.1, splits=1000):
    rng = np.random.default_rng(0)
    n, nc = len(t), int(0.3 * len(t))
    cov, width, size = [], [], []
    for _ in range(splits):
        perm = rng.permutation(n)
        cal, te = perm[:nc], perm[nc:]
        tau = q(np.abs(t - p)[cal], eps)
        cov.append((np.abs(t[te] - p[te]) <= tau).mean())
        width.append(2 * tau)
        size.append(((np.abs(K[None] - p[te][:, None]) <= tau + 1e-9).sum(1)).mean())
    return {"coverage": 100 * float(np.mean(cov)), "width": float(np.mean(width)), "set_size": float(np.mean(size))}


def main():
    from baselines_same_split import SMALL, recover_users
    from bootstrap import collect
    from segments import BEST
    cols = collect()
    out, md = {}, ["| Setting | CPR-AE MAE / RMSE | HyCPR-LLM (Mistral) | best baseline | CPR-AE - HyCPR (MAE, RMSE) | CP width / admissible / coverage: CPR-AE | HyCPR-LLM |",
                   "|---|---|---|---|---|---|---|"]
    for name, settings in (("LDOS", ["LDOS-H"]), ("Depaul", ["Depaul-H"]), ("Tijuana", ["Restaurant-H"])):
        tr, te, _ = recover_users(*SMALL[name])
        tr, te = tr.assign(u=tr.u.astype(str)), te.assign(u=te.u.astype(str))
        ctx = [c for c in tr.columns if c.startswith("c_")]
        p, hyper = fit_predict(tr, te, ctx)
        t = te.r.to_numpy(float)
        for s in settings:
            h = json.load(open(f"probs/hybrid/mistral_hycpr_{s}.json"))
            assert np.array_equal(np.array(h["truth"], float), t), s
            hy = np.array(h["expected"], float)
            best = cols[s]["models"][BEST[s]]
            f = lambda x: [float(np.mean(np.abs(x - t))), float(np.sqrt(np.mean((x - t) ** 2)))]
            rec = {"hyper": hyper, "cprae": f(p), "hycpr": f(hy), "best": [BEST[s]] + f(best),
                   "vs_hycpr": {m: test(p, hy, t, m) for m in ("mae", "rmse")},
                   "vs_best": {m: test(p, best, t, m) for m in ("mae", "rmse")},
                   "cp_cprae": split_cp(t, p), "cp_hycpr": split_cp(t, hy)}
            out[s] = rec
            json.dump({"truth": t.tolist(), "expected": p.tolist()}, open(f"probs/cprae_{s}.json", "w"))
            sg = lambda d: f"{d['diff']:+.3f}{'*' if d['p'] < 0.05 else ''}"
            c = lambda r: f"{r['width']:.2f} / {r['set_size']:.2f} / {r['coverage']:.1f}"
            md.append(f"| {s} | {rec['cprae'][0]:.3f} / {rec['cprae'][1]:.3f} | {rec['hycpr'][0]:.3f} / {rec['hycpr'][1]:.3f} | "
                      f"{BEST[s]} {rec['best'][1]:.3f} / {rec['best'][2]:.3f} | {sg(rec['vs_hycpr']['mae'])}, {sg(rec['vs_hycpr']['rmse'])} | "
                      f"{c(rec['cp_cprae'])} | {c(rec['cp_hycpr'])} |")
            print(md[-1], flush=True)
    json.dump(out, open("results/cpr_ae.json", "w"), indent=1)
    open("results/cpr_ae.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
