"""Semantic text baseline without LLM fine-tuning (review 5: "additional rating-prediction baselines that exploit
semantic text without LLM fine-tuning (e.g., Text-KNN) could help disentangle the value of LLMs vs. simpler semantic
models").

Text-KNN: item-based kNN in which the item-item similarity is the cosine similarity of frozen sentence embeddings
(SBERT all-MiniLM-L6-v2, mean pooling) of the item description shown in the prompts (title, genres / categories /
brand, and the other attributes of the "Details" line). For a test interaction (u, i):
    r_hat = mu_u + sum_{j in N_k(i; u)} s_ij (r_uj - mu_u) / sum_{j in N_k(i; u)} s_ij ,
where N_k(i; u) are the k = 20 training items of u most similar to i with s_ij > 0, and mu_u the user's mean training
rating (global mean for users without training ratings; mu_u alone when no similar item); clipped to [1, 5].
k = 20 as for ItemKNN, fixed in advance. Training data: the same as the CF baselines (small datasets: the training
interactions of the prompts, users recovered as in baselines_same_split.py; Yelp, MovieLens-1M, Amazon-Books: the
whole training part of the chronological split). Test interactions: the same as every other model.

Paired bootstrap (10,000 replicates) against CPR-LLM and HyCPR-LLM (Mistral) and the best baseline of the column.
Usage: python text_knn.py  → results/text_knn.{json,md}, probs/textknn_<setting>.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
K = 20


def embed(texts, dev="cuda", bs=512):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), bs):
            t = tok(texts[b:b + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            h = enc(**t).last_hidden_state
            a = t["attention_mask"][..., None].float()
            out.append(torch.nn.functional.normalize((h * a).sum(1) / a.sum(1), dim=-1).cpu().numpy())
    return np.concatenate(out)


def predict(tr_u, tr_i, tr_r, te_u, te_i, text_of):
    """tr_*: training users / item keys / ratings; te_*: test users / item keys; text_of: item key -> text."""
    items = sorted(set(tr_i) | set(te_i), key=str)
    E = dict(zip(items, embed([text_of(i) for i in items])))
    g = float(np.mean(tr_r))
    by_user = {}
    for u, i, r in zip(tr_u, tr_i, tr_r):
        by_user.setdefault(u, []).append((i, r))
    pred, n_sim = [], []
    for u, i in zip(te_u, te_i):
        hist = by_user.get(u, [])
        if not hist:
            pred.append(g)
            n_sim.append(0)
            continue
        r = np.array([x[1] for x in hist], float)
        mu = r.mean()
        s = np.stack([E[x[0]] for x in hist]) @ E[i]
        top = np.argsort(-s)[:K]
        top = top[s[top] > 0]
        n_sim.append(len(top))
        p = mu if len(top) == 0 else mu + (s[top] * (r[top] - mu)).sum() / s[top].sum()
        pred.append(float(np.clip(p, 1, 5)))
    return np.array(pred), np.array(n_sim)


def small(name):
    from baselines_same_split import SMALL, recover_users
    tr, te, _ = recover_users(*SMALL[name])
    return predict(tr.u.astype(str).tolist(), tr.i.tolist(), tr.r.astype(float).tolist(),
                   te.u.astype(str).tolist(), te.i.tolist(), lambda i: i), te.r.astype(float).to_numpy()


def large(dataset):
    if dataset == "yelp":
        from baselines_same_split import yelp_split
        df, _, te, cut = yelp_split()
        items = set(df["item"])
        biz = {}
        with open(os.path.join(HERE, "..", "raw_data", "yelp", "yelp_academic_dataset_business.json")) as f:
            for line in f:
                b = json.loads(line)
                if b["business_id"] in items:
                    biz[b["business_id"]] = f"Name: {b['name']}, Categories: {b.get('categories') or 'unknown'}"
        text_of = biz.get
    else:
        import large_baselines as lb
        cap = lb.rebuild(dataset)
        df, te = cap["df"], list(cap["te"])
        cut = int(len(df) * 0.9)
        first = df.drop_duplicates("item").set_index("item")

        def text_of(i, first=first, render=cap["render"]):
            r = first.loc[i].copy()
            r["item"] = i
            return render(r).split(". The user rated it as:")[0]
    trd = df.iloc[:cut]
    users = set(df.loc[te, "user"])
    trd = trd[trd["user"].isin(users)]
    pred = predict(trd["user"].tolist(), trd["item"].tolist(), trd["label"].astype(float).tolist(),
                   df.loc[te, "user"].tolist(), df.loc[te, "item"].tolist(), text_of)
    return pred, df.loc[te, "label"].astype(float).to_numpy()


def main():
    from bootstrap import collect
    from holm import test
    cols = collect()
    out, md = {}, ["| Setting | Text-KNN MAE / RMSE | CPR-LLM Mistral | HyCPR-LLM Mistral | best baseline | Text-KNN - HyCPR (MAE, RMSE) |",
                   "|---|---|---|---|---|---|"]
    jobs = [("LDOS", ["LDOS-WH", "LDOS-H"]), ("Depaul", ["Depaul-WH", "Depaul-H"]),
            ("Tijuana", ["Tijuana-WH", "Restaurant-H"]), ("yelp", ["YELP-WH", "YELP-H"]),
            ("movielens", ["MovieLens-H"]), ("amazon", ["AmazonBooks-H"])]
    for src, settings in jobs:
        (p, nsim), t = small(src) if src in ("LDOS", "Depaul", "Tijuana") else large(src)
        print(src, "users without a similar training item:", float((nsim == 0).mean()), flush=True)
        for s in settings:
            h = json.load(open(f"probs/hybrid/mistral_hycpr_{s}.json"))
            th = np.array(h["truth"], float)
            assert np.array_equal(th, t), s
            hy = np.array(h["expected"], float)
            if s in cols:
                llm = cols[s]["models"]["mistral_clean" if s.endswith("-H") and not s.startswith("YELP") else "mistral"]
                base = {k: v for k, v in cols[s]["models"].items() if not any(x in k for x in ("mistral", "llama", "eurollm", "deepseek"))}
            else:
                name = s[:-2]
                llm = np.array(json.load(open(f"probs/mistral_full_{s}.json"))["expected"], float)
                base = {k: np.array(v, float) for k, v in json.load(open(f"results/large_baselines_{name}.json"))["models"].items()}
            mae = lambda x: float(np.mean(np.abs(x - t)))
            rmse = lambda x: float(np.sqrt(np.mean((x - t) ** 2)))
            bname = min(base, key=lambda k: mae(base[k]))
            rec = {"textknn": [mae(p), rmse(p)], "cpr": [mae(llm), rmse(llm)], "hycpr": [mae(hy), rmse(hy)],
                   "best_baseline": [bname, mae(base[bname]), rmse(base[bname])],
                   "frac_no_similar": float((nsim == 0).mean()),
                   "vs_hycpr": {m: test(p, hy, t, m) for m in ("mae", "rmse")},
                   "vs_cpr": {m: test(p, llm, t, m) for m in ("mae", "rmse")},
                   "vs_best": {m: test(p, base[bname], t, m) for m in ("mae", "rmse")}}
            out[s] = rec
            json.dump({"truth": t.tolist(), "expected": p.tolist()}, open(f"probs/textknn_{s}.json", "w"))
            f = lambda d: f"{d['diff']:+.3f}{'*' if d['p'] < 0.05 else ''}"
            md.append(f"| {s} | {rec['textknn'][0]:.3f} / {rec['textknn'][1]:.3f} | {rec['cpr'][0]:.3f} / {rec['cpr'][1]:.3f} | "
                      f"{rec['hycpr'][0]:.3f} / {rec['hycpr'][1]:.3f} | {bname} {rec['best_baseline'][1]:.3f} / {rec['best_baseline'][2]:.3f} | "
                      f"{f(rec['vs_hycpr']['mae'])}, {f(rec['vs_hycpr']['rmse'])} |")
            print(md[-1], flush=True)
    json.dump(out, open("results/text_knn.json", "w"), indent=1)
    open("results/text_knn.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
