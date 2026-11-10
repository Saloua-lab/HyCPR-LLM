"""Conformal prediction without logits (review 6: "it is unclear how the uncertainty component (intervals) is adapted
and validated without logits").

API-only variant of HyCPR-LLM: the LLM's distribution over the five ratings is replaced by the empirical frequencies
of n sampled answers (temperature 1, sample_eval.py, Mistral -H models; n = 5, 10, 20), the LLM rating by their mean,
and the fused rating by w * mean + (1 - w) * b with the CF model b and weight w of HyCPR-LLM (b recovered as in
cp_cqr.py). The four scores of cp_adaptive.py (absolute, normalized, CQR-style, ordinal APS) are then computed from
the empirical distribution exactly as from the token probabilities, with split CP (30% calibration, 1,000 splits,
eps = 0.1). For the scores that use the distribution, the frequencies are smoothed (add 0.5 to each of the five
counts), since a rating that was never sampled would otherwise get probability 0 and make the ordinal APS score tie
at 1 for every such interaction (full sets). The logit-based values are recomputed on the same splits for comparison.
On the small datasets the sampled answers come from the leak-free adapters (probs/mistral_clean_*), whose fusion
weight is taken from HyCPR-LLM's v0 fusion: the interval comparison is therefore between the two read-outs of the
LLM, not a re-validation of w.

Usage: python cp_sampled.py → results/cp_sampled.{json,md}
"""
import json

import numpy as np

from cp_adaptive import K, aps_sets, evaluate, load, quantile_of

SETTINGS = {"LDOS-H": "probs/mistral_clean_LDOS-H.json", "Depaul-H": "probs/mistral_clean_Depaul-H.json",
            "Restaurant-H": "probs/mistral_clean_Restaurant-H.json", "MovieLens-H": "probs/mistral_full_MovieLens-H.json",
            "AmazonBooks-H": "probs/mistral_full_AmazonBooks-H.json"}
EPS = 0.10


def scores(t, hy, P, mu):
    sig = np.sqrt((P * (K[None] - mu[:, None]) ** 2).sum(1)) + 0.1
    within = lambda lo, hi, idx: (K[None] >= lo[idx][:, None] - 1e-9) & (K[None] <= hi[idx][:, None] + 1e-9)
    rng = lambda: np.random.default_rng(0)
    res = {"abs": evaluate(t, np.abs(t - hy), lambda tau, i: within(hy - tau, hy + tau, i), EPS, rng()),
           "normalized": evaluate(t, np.abs(t - hy) / sig, lambda tau, i: within(hy - tau * sig, hy + tau * sig, i), EPS, rng())}
    qlo, qhi = quantile_of(P, EPS / 2) + (hy - mu), quantile_of(P, 1 - EPS / 2) + (hy - mu)
    res["cqr"] = evaluate(t, np.maximum(qlo - t, t - qhi), lambda tau, i: within(qlo - tau, qhi + tau, i), EPS, rng())
    order = aps_sets(P)
    rank = np.argsort(order, 1)
    Ps = np.take_along_axis(P, order, 1).cumsum(1)
    sc = Ps[np.arange(len(t)), rank[np.arange(len(t)), (t - 1).astype(int)]]
    res["ordinal_aps"] = evaluate(t, sc, lambda tau, i: np.take_along_axis(Ps[i] <= tau + 1e-12, rank[i], 1) | (rank[i] == 0),
                                  EPS, rng())
    return res


def main():
    out, md = {}, ["| setting | read-out | abs | normalized | CQR-style | ordinal APS (LLM alone) |", "|---|---|---|---|---|---|"]
    for s, lp in SETTINGS.items():
        t, hy, P_v0, mu_v0 = load("mistral", s)
        w = json.load(open(f"probs/hybrid/mistral_hycpr_{s}.json"))["w"]
        b = (hy - w * mu_v0) / (1 - w)
        L = json.load(open(lp))
        S = json.load(open(f"probs/sampled/{s}.json"))
        assert np.array_equal(np.array(L["truth"], float), t) and np.array_equal(np.array(S["truth"], float), t), s
        P_log, mu_log = np.array(L["probs"], float), np.array(L["expected"], float)
        rec = {"logits": scores(t, w * mu_log + (1 - w) * b, P_log, mu_log)}
        for n in (5, 10, 20):
            C = np.zeros((len(t), 5))
            for j, row in enumerate(S["samples"]):
                v = [x for x in row[:n] if x is not None and 1 <= x <= 5]
                if v:
                    np.add.at(C[j], np.array(v) - 1, 1)
                else:
                    C[j] = P_log[j]                     # no parsable sample (< 0.3% of the prompts): logits
            mun = (C / C.sum(1, keepdims=True)) @ K           # mean of the sampled ratings
            Ps = (C + 0.5) / (C.sum(1, keepdims=True) + 2.5)  # Jeffreys smoothing: a rating never sampled keeps a
            rec[f"sampled_{n}"] = scores(t, w * mun + (1 - w) * b, Ps, Ps @ K)  # small mass (else APS ties at 1)
        out[s] = rec
        for k, r in rec.items():
            md.append(f"| {s} | {k} | " + " | ".join(f"{r[x]['coverage']:.1f} / {r[x]['set_size']:.2f} / {r[x]['low_coverage']:.0f}"
                                                   for x in ("abs", "normalized", "cqr", "ordinal_aps")) + " |")
        print("\n".join(md[-4:]), flush=True)
    json.dump(out, open("results/cp_sampled.json", "w"), indent=1)
    open("results/cp_sampled.md", "w").write("# coverage % / admissible ratings / coverage of ratings 1-2 %, eps = 0.1\n\n" + "\n".join(md) + "\n")


if __name__ == "__main__":
    main()
