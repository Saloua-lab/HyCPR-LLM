"""Neural context-aware rating-prediction baselines (review: "strong non-LLM neural CARS models").

Every model reads one categorical field per input: user, item and each context variable shown in the prompt.
Index 0 of every field is reserved for values not seen in training.

  NFM      He & Chua 2017: global bias + linear terms + MLP over the bi-interaction pooling of the field embeddings
  DeepFM   Guo et al. 2017: FM (linear + pairwise interactions) + MLP over the concatenated field embeddings,
           sharing the embeddings
  AutoInt  Song et al. 2019: linear terms + stacked multi-head self-attention over the field embeddings (with
           residual connections), flattened into a linear output

Training: MSE on the rating, Adam, L2 weight decay, prediction = global mean + model output, clipped to 1..5.
Model selection uses only the training data. A random 10% of the training interactions is held out, every
configuration of the grid is trained with early stopping on that part, and the configuration and number of
epochs with the lowest validation RMSE are kept. The model is then retrained on the whole training set for that
number of epochs, and the test predictions are averaged over 5 seeds, which reduces the variance of training
on a few hundred interactions.
"""
import itertools
import sys

import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["NFM", "DeepFM", "AutoInt"]


class Fields:
    """Integer encoding of the categorical fields; unseen values map to 0."""

    def __init__(self, train, cols):
        self.cols = cols
        self.maps = [{v: k + 1 for k, v in enumerate(sorted(set(map(str, train[c]))))} for c in cols]
        self.sizes = [len(m) + 1 for m in self.maps]

    def encode(self, d):
        return np.stack([d[c].map(str).map(m).fillna(0).astype(int).to_numpy()
                         for c, m in zip(self.cols, self.maps)], 1)


class Net(nn.Module):
    def __init__(self, kind, sizes, dim, dropout, hidden=64, heads=2, layers=2):
        super().__init__()
        self.kind, self.nf = kind, len(sizes)
        self.offsets = torch.tensor(np.concatenate([[0], np.cumsum(sizes)[:-1]]), dtype=torch.long)
        total = int(sum(sizes))
        self.lin = nn.Embedding(total, 1)
        self.emb = nn.Embedding(total, dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.normal_(self.emb.weight, 0, 0.01)
        if kind == "NFM":
            self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, 1))
        elif kind == "DeepFM":
            self.mlp = nn.Sequential(nn.Linear(self.nf * dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        elif kind == "AutoInt":
            self.att = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                                      for _ in range(layers)])
            self.res = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layers)])
            self.out = nn.Linear(self.nf * dim, 1)
        else:
            raise ValueError(kind)

    def forward(self, x):
        x = x + self.offsets.to(x.device)
        first = self.lin(x).sum((1, 2))
        e = self.emb(x)  # (batch, fields, dim)
        if self.kind == "NFM":
            bi = 0.5 * (e.sum(1) ** 2 - (e ** 2).sum(1))
            return first + self.mlp(bi).squeeze(-1)
        if self.kind == "DeepFM":
            fm = 0.5 * (e.sum(1) ** 2 - (e ** 2).sum(1)).sum(-1)
            return first + fm + self.mlp(e.flatten(1)).squeeze(-1)
        h = e
        for att, res in zip(self.att, self.res):
            a, _ = att(h, h, h, need_weights=False)
            h = torch.relu(a + res(h))
        return first + self.out(h.flatten(1)).squeeze(-1)


def train_net(kind, Xtr, ytr, sizes, cfg, epochs, seed, Xval=None, yval=None, patience=20, batch=256):
    """Trains one network. With validation data: returns (best val RMSE, best epoch). Otherwise: the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(kind, sizes, cfg["dim"], cfg["dropout"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    mu = float(ytr.mean())
    X, y = torch.tensor(Xtr, device=DEV), torch.tensor(ytr - mu, dtype=torch.float32, device=DEV)
    if Xval is not None:
        Xv = torch.tensor(Xval, device=DEV)
    best, best_ep, bad = np.inf, 0, 0
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(y), device=DEV)
        for k in range(0, len(y), batch):
            b = perm[k:k + batch]
            loss = ((net(X[b]) - y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xval is None:
            continue
        net.eval()
        with torch.no_grad():
            pv = (net(Xv).cpu().numpy() + mu).clip(1, 5)
        rmse = float(np.sqrt(((pv - yval) ** 2).mean()))
        if rmse < best - 1e-5:
            best, best_ep, bad = rmse, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if Xval is not None:
        return best, best_ep
    net.eval()
    net.mu = mu
    return net


def predict(net, X, batch=65536):
    out = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            out.append(net(torch.tensor(X[k:k + batch], device=DEV)).cpu().numpy())
    return (np.concatenate(out) + net.mu).clip(1, 5)


GRID = {"dim": [8, 16, 32, 64], "wd": [1e-5, 1e-4, 1e-3, 1e-2], "dropout": [0.0, 0.3], "lr": [1e-3]}


def fit_predict(train, test, cols, kind, grid=GRID, max_epochs=1000, seeds=(0, 1, 2, 3, 4), batch=256,
                val_frac=0.1, log=sys.stderr):
    """Test predictions of `kind` trained on `train` (pandas frames with column r and the field columns)."""
    f = Fields(train, cols)
    X, y = f.encode(train), train["r"].to_numpy(float)
    Xte = f.encode(test)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    nv = max(1, int(len(y) * val_frac))
    vi, ti = perm[:nv], perm[nv:]
    # encode the fitting part again so that values seen only in the validation part count as unseen
    ftr = Fields(train.iloc[ti], cols)
    Xf, Xv = ftr.encode(train.iloc[ti]), ftr.encode(train.iloc[vi])
    best = None
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(grid.keys(), vals))
        rmse, ep = train_net(kind, Xf, y[ti], ftr.sizes, cfg, max_epochs, 0, Xv, y[vi], batch=batch)
        if best is None or rmse < best[0]:
            best = (rmse, ep, cfg)
    rmse, ep, cfg = best
    print(f"  {kind}: val RMSE {rmse:.4f}, {ep} epochs, {cfg}", file=log)
    preds = [predict(train_net(kind, X, y, f.sizes, cfg, ep, s, batch=batch), Xte) for s in seeds]
    return np.mean(preds, 0), {"val_rmse": rmse, "epochs": ep, **cfg, "seeds": list(seeds)}
