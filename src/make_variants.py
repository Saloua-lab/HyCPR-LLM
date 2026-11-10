"""Datasets for the improvement experiments (validation split + collaborative hints in the prompt).

For one prompt DatasetDict (train/test) it writes two DatasetDicts with splits train / validation / test:
  data/variants/<name>_val : the original prompts; `validation` = 10% of the training interactions
  data/variants/<name>_cf  : the same prompts with a "### Rating Statistics:" section before the answer marker
The validation interactions are chosen by a hash of (user, item, context, rating), so the split is the same for
every model and for the -WH and -H versions of a dataset. The test split is unchanged.

Hints (no test or validation information leaks into them):
  - user's average rating and count, item's average rating and count, a BiasedMF estimate;
  - for training prompts: user/item means leave the prompt's own rating out, and the MF estimate is out-of-fold
    (5 folds within the training part);
  - for validation and test prompts: computed from the training part only (the LLM is trained on it only).
Users that could not be identified in the source files (11 LDOS and 7 Tijuana training rows) get "unknown".

Usage: python make_variants.py <name> <prompt DatasetDict> <LDOS|Depaul|Tijuana>
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines_same_split import recover_users  # noqa: E402
from evaluate_probs import MARKER_RE  # noqa: E402

VAL_MOD = 10   # 1 in 10 training interactions goes to validation
FOLDS = 5
ITEM_WORD = {"LDOS": "movie", "Depaul": "movie", "Tijuana": "restaurant"}


def interaction_key(row, ctx):
    return "|".join([str(row["u"]), str(row["i"])] + [str(row.get(c, "NA")) for c in ctx] + [str(row["r"])])


def is_val(row, ctx):
    return int(hashlib.md5(interaction_key(row, ctx).encode()).hexdigest(), 16) % VAL_MOD == 0


def mf_predict(fit, pred, ctx):
    from baselines import factor_model
    known = fit[fit["u"] != "None"]
    return np.asarray(factor_model(known, pred, ctx, "BiasedMF"), float)


def means(fit, pred, col, loo):
    """Mean rating and count of pred[col] in fit; loo: pred rows are fit rows, leave their own rating out."""
    g = fit.groupby(col)["r"].agg(["sum", "count"])
    s = pred[col].map(g["sum"]).fillna(0).to_numpy(float)
    n = pred[col].map(g["count"]).fillna(0).to_numpy(float)
    if loo:
        s, n = s - pred["r"].to_numpy(float), n - 1
    valid = (n > 0) & (pred[col] != "None").to_numpy()
    return np.where(valid, s / np.maximum(n, 1), np.nan), np.where(valid, n, 0).astype(int)


def hint(word, um, un, im, in_, mf):
    u = f"{um:.1f} (from {un} other ratings)" if not np.isnan(um) else "unknown"
    i = f"{im:.1f} (from {in_} other ratings)" if not np.isnan(im) else "unknown"
    return (f"### Rating Statistics:\nUser's average rating: {u}\n{word.capitalize()}'s average rating: {i}\n"
            f"Collaborative-filtering estimate: {mf:.1f}\n\n")


def add_hint(example, text):
    m = None
    for m in MARKER_RE.finditer(example):
        pass
    return example[:m.start()] + text + example[m.start():]


def main(name, path, source):
    d = load_from_disk(path)
    tr, te, _ = recover_users(path, source, keep_unknown=True)
    assert len(tr) == len(d["train"]) and len(te) == len(d["test"])
    ctx = sorted(c for c in set(tr.columns) | set(te.columns) if c.startswith("c_"))
    for t in (tr, te):
        for c in ctx:
            t[c] = t[c].fillna("NA").astype(str) if c in t else "NA"
        t["u"], t["i"] = t["u"].astype(str), t["i"].astype(str)
    val_mask = np.array([is_val(r, ctx) for _, r in tr.iterrows()])
    T, V = tr[~val_mask].reset_index(drop=True), tr[val_mask].reset_index(drop=True)

    # hints for the training part: leave-one-out means, out-of-fold MF
    fold = np.arange(len(T)) % FOLDS
    np.random.default_rng(0).shuffle(fold)
    mf_T = np.zeros(len(T))
    for f in range(FOLDS):
        mf_T[fold == f] = mf_predict(T[fold != f], T[fold == f], ctx)
    hT = (*means(T, T, "u", True), *means(T, T, "i", True), mf_T)
    hV = (*means(T, V, "u", False), *means(T, V, "i", False), mf_predict(T, V, ctx))
    hS = (*means(T, te, "u", False), *means(T, te, "i", False), mf_predict(T, te, ctx))

    w = ITEM_WORD[source]
    ex_tr = np.array(d["train"]["example"], dtype=object)
    ex_T, ex_V, ex_S = list(ex_tr[~val_mask]), list(ex_tr[val_mask]), list(d["test"]["example"])
    with_h = lambda exs, h: [add_hint(e, hint(w, *[x[k] for x in h])) for k, e in enumerate(exs)]
    out = os.path.join("data", "variants")
    os.makedirs(out, exist_ok=True)
    DatasetDict({"train": Dataset.from_dict({"example": ex_T}), "validation": Dataset.from_dict({"example": ex_V}),
                 "test": Dataset.from_dict({"example": ex_S})}).save_to_disk(os.path.join(out, f"{name}_val"))
    DatasetDict({"train": Dataset.from_dict({"example": with_h(ex_T, hT)}),
                 "validation": Dataset.from_dict({"example": with_h(ex_V, hV)}),
                 "test": Dataset.from_dict({"example": with_h(ex_S, hS)})}).save_to_disk(os.path.join(out, f"{name}_cf"))
    # the interaction tables of the split, for the baselines and the hybrid
    pd.concat([T.assign(split="train"), V.assign(split="validation"), te.assign(split="test")]).to_parquet(
        os.path.join(out, f"{name}_interactions.parquet"))
    print(f"{name}: train {len(T)}, validation {len(V)}, test {len(te)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
