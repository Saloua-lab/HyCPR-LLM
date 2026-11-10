"""Calibrated and ordinal read-outs of the LLM's rating distribution (review 6: "does not exploit richer ordinal
modeling (e.g., explicit ordinal regression heads or calibrated class probabilities)").

For each LLM and -H setting, the distribution p(.|x) over the five rating tokens of the LLM inside the fusion
(small datasets: v0 adapter, validation = 10% of the training interactions; large: validation period) is
post-processed with a model fitted on the validation predictions only:
  raw          p itself (the paper's read-out)
  temperature  p_k^(1/T) renormalized, T minimizing the validation NLL (Guo et al. 2017)
  ordinal      proportional-odds (cumulative-logit) head on the LLM's log-probabilities:
               P(r <= k | x) = sigmoid(theta_k - beta^T z(x)), k = 1..4, z = (log p_k - mean_j log p_j)_{k=1..5},
               theta increasing, fitted by L2-penalized (lambda = 1e-2) maximum likelihood on the validation set
               (McCullagh 1980); this is an ordinal regression head on top of the frozen LLM's output layer
Test metrics: NLL, ranked probability score (RPS, the proper score for ordinal outcomes), expected calibration error of
the expected rating (10 equal-mass bins), MAE / RMSE of the expected rating; paired bootstrap (10,000) of the MAE and
RMSE differences with the raw read-out.

Usage: python ordinal_calib.py → results/ordinal_calib.{json,md}
"""
import json

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from bootstrap import mae_rmse

LLMS = ["mistral", "eurollm", "llama", "deepseek"]
SETTINGS = ["LDOS-H", "Depaul-H", "Restaurant-H", "MovieLens-H", "AmazonBooks-H"]
K = np.arange(1, 6)


def paths(m, s):
    if s in ("MovieLens-H", "AmazonBooks-H"):
        return f"probs/{m}_full_{s}_val.json", f"probs/{m}_full_{s}.json"
    return f"probs/variants/{m}_v0_{s}_validation.json", f"probs/variants/{m}_v0_{s}_test.json"


def temp(P, T):
    L = np.log(np.clip(P, 1e-12, 1)) / T
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def feats(P):
    L = np.log(np.clip(P, 1e-12, 1))
    return L - L.mean(1, keepdims=True)


def po_probs(params, Z):
    th = np.cumsum(np.r_[params[0], np.exp(params[1:4])])          # increasing thresholds
    beta = params[4:]
    c = expit(th[None] - (Z @ beta)[:, None])                       # (n, 4) P(r <= k)
    c = np.c_[np.zeros(len(Z)), c, np.ones(len(Z))]
    return np.clip(np.diff(c, axis=1), 1e-12, 1)


def fit_po(Z, y, lam=1e-2):
    def nll(p):
        return -np.log(po_probs(p, Z)[np.arange(len(y)), y - 1]).mean() + lam * (p[4:] ** 2).sum()
    # start: beta along the expected-rating direction of the log-probabilities
    x0 = np.r_[-1.5, np.log([1.0, 1.0, 1.0]), 0.5 * (K - 3) / 2]
    return minimize(nll, x0, method="L-BFGS-B").x


def metrics(P, y):
    e = P @ K
    nll = -np.log(np.clip(P[np.arange(len(y)), y - 1], 1e-12, 1)).mean()
    cdf, oh = np.cumsum(P, 1)[:, :4], np.cumsum(np.eye(5)[y - 1], 1)[:, :4]
    rps = ((cdf - oh) ** 2).sum(1).mean() / 4
    bins = np.array_split(np.argsort(e), 10)
    ece = sum(len(b) * abs(e[b].mean() - y[b].mean()) for b in bins) / len(y)
    mae, rmse = mae_rmse(e - y)
    return {"nll": float(nll), "rps": float(rps), "ece": float(ece), "mae": float(mae), "rmse": float(rmse)}


def boot(a, b, y, B=10000):
    idx = np.random.default_rng(0).integers(0, len(y), (B, len(y)))
    out = {}
    for name, f in (("mae", lambda x, yy: np.abs(x - yy).mean(-1)), ("rmse", lambda x, yy: np.sqrt(((x - yy) ** 2).mean(-1)))):
        d = f(a[idx], y[idx]) - f(b[idx], y[idx])
        out[name] = [float(f(a, y) - f(b, y)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def main():
    out, md = {}, ["| LLM | setting | n val | read-out | T | NLL | RPS | ECE | MAE | RMSE | vs raw MAE | vs raw RMSE |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        for m in LLMS:
            pv, pt = paths(m, s)
            try:
                V, Tt = json.load(open(pv)), json.load(open(pt))
            except FileNotFoundError:
                continue
            Pv, yv = np.array(V["probs"], float), np.array(V["truth"], int)
            Pt, yt = np.array(Tt["probs"], float), np.array(Tt["truth"], int)
            T = minimize_scalar(lambda T: -np.log(temp(Pv, T)[np.arange(len(yv)), yv - 1]).mean(),
                                bounds=(0.2, 10), method="bounded").x
            po = fit_po(feats(Pv), yv)
            reads = {"raw": Pt, "temperature": temp(Pt, T), "ordinal": po_probs(po, feats(Pt))}
            rec = {"n_val": len(yv), "T": float(T)}
            for k, P in reads.items():
                r = metrics(P, yt)
                if k != "raw":
                    r["vs_raw"] = boot(P @ K, Pt @ K, yt.astype(float))
                rec[k] = r
                sig = lambda d: f"{d[0]:+.3f}{'*' if d[1] > 0 or d[2] < 0 else ''}"
                md.append(f"| {m} | {s} | {len(yv)} | {k} | {T:.2f} | {r['nll']:.3f} | {r['rps']:.4f} | {r['ece']:.3f} | "
                          f"{r['mae']:.3f} | {r['rmse']:.3f} | " + (" | ".join(sig(r['vs_raw'][x]) for x in ("mae", "rmse")) if k != "raw" else " | ") + " |")
            out[f"{m} {s}"] = rec
            print("\n".join(md[-3:]), flush=True)
    json.dump(out, open("results/ordinal_calib.json", "w"), indent=1)
    open("results/ordinal_calib.md", "w").write("# Calibrated / ordinal read-outs (test; * = paired bootstrap 95% CI excludes 0)\n\n" + "\n".join(md) + "\n")


if __name__ == "__main__":
    main()
