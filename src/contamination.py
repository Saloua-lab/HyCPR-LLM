"""Pretraining-contamination probes (review: "no analysis of potential pretraining contamination, e.g. MovieLens").

Run on the PRETRAINED base models (no instruction tuning, no LoRA), with greedy decoding, in the format in which
the datasets are distributed, following the memorization probes of Di Palma et al. (2025) for MovieLens-1M:

  ml_items    movies.dat: the 5 lines that precede the movie in the file ("MovieID::Title (Year)::Genres"),
              then "<MovieID>::" -> the model completes the title. Exact match = memorized item. Probed for every movie of our test split.
  ml_ratings  ratings.dat: for each of the 2,000 test interactions, the 10 lines that precede it in the file,
              then "<UserID>::<MovieID>::" -> rating and timestamp. Exact rating and exact Unix timestamp
              (10 digits) compared with the controls: the mode of the user's ratings among the context lines,
              and the timestamp of the previous line.
  az_items    Amazon Books metadata: 5 example pairs "ASIN: <asin> / Title: <title>", then the ASIN of a test
              item -> title. Exact (normalized) and fuzzy (difflib ratio >= 0.8) match.

Output: results/contamination/<model>.json (per-probe metrics and per-item / per-interaction flags).
Usage: HF_HOME=../.hf_cache python contamination.py --model mistral [--limit N]
"""
import argparse
import difflib
import gzip
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RAW = os.path.join(HERE, "..", "raw_data")
BASES = {"mistral": "mistralai/Mistral-7B-v0.1", "llama": "meta-llama/Llama-2-7b-hf",
         "eurollm": "utter-project/EuroLLM-9B", "deepseek": "deepseek-ai/deepseek-llm-7b-base"}


def norm(s):
    s = s.lower().replace("&amp;", "&").replace("&apos;", "'").replace("&quot;", '"')
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def test_split(dataset):
    from large_baselines import rebuild
    cap = rebuild(dataset)
    df, te = cap["df"], cap["te"]
    return df, list(te)


def amazon_titles(asins):
    cache = os.path.join(HERE, "data", "amazon_test_titles.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    want, out = set(asins), {}
    with gzip.open(os.path.join(RAW, "amazon", "meta_Books.json.gz"), "rt") as f:
        for line in f:
            m = json.loads(line)
            if m.get("asin") in want and m.get("title"):
                out[m["asin"]] = m["title"]
    json.dump(out, open(cache, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(BASES), required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from tokenization import load_tokenizer

    tok = load_tokenizer(BASES[a.model])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASES[a.model], dtype=torch.bfloat16, device_map="cuda").eval()

    def generate(prompts, n):
        out = []
        for k in range(0, len(prompts), a.batch):
            enc = tok(prompts[k:k + a.batch], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    res = {"model": BASES[a.model]}
    # ------------------------------------------------------------------ MovieLens items
    ml = os.path.join(RAW, "movielens", "ml-1m")
    movies, order = {}, []
    mlines = [l.rstrip("\n") for l in open(os.path.join(ml, "movies.dat"), encoding="latin-1") if l.strip()]
    for line in mlines:
        mid, title, genres = line.split("::")
        movies[int(mid)] = (title, genres)
        order.append(int(mid))
    where = {m: k for k, m in enumerate(order)}
    df, te = test_split("movielens")
    test_items = sorted({int(df.at[i, "item"]) for i in te})[: a.limit]
    shots = "5 preceding lines of movies.dat"
    # context: the 5 lines that precede the movie in movies.dat (verbatim file continuation, as for ratings.dat)
    outs = generate(["\n".join(mlines[max(0, where[m] - 5):where[m]]) + ("\n" if where[m] else "") + f"{m}::"
                     for m in test_items], 32)
    flags = {}
    for m, o in zip(test_items, outs):
        guess = o.split("::")[0].split("\n")[0]
        flags[m] = {"exact": norm(guess) == norm(movies[m][0]),
                    "no_year": norm(re.sub(r"\(\d{4}\)", "", guess)) == norm(re.sub(r"\(\d{4}\)", "", movies[m][0])),
                    "guess": guess}
    res["ml_items"] = {"n": len(flags), "exact": float(np.mean([f["exact"] for f in flags.values()])),
                       "title_without_year": float(np.mean([f["no_year"] for f in flags.values()])),
                       "shots": shots, "flags": {str(k): v for k, v in flags.items()}}
    print(a.model, "ml_items", res["ml_items"]["n"], f"exact {res['ml_items']['exact']:.3f}",
          f"no-year {res['ml_items']['title_without_year']:.3f}", flush=True)
    # ------------------------------------------------------------------ MovieLens ratings
    lines = open(os.path.join(ml, "ratings.dat")).read().split("\n")
    lines = [l for l in lines if l]
    pos = {tuple(map(int, l.split("::")[:2])): k for k, l in enumerate(lines)}
    rows = te[: a.limit] if a.limit else te
    prompts, ctrl = [], []
    for i in rows:
        u, m = int(df.at[i, "user"]), int(df.at[i, "item"])
        k = pos[(u, m)]
        ctx = lines[max(0, k - 10):k]
        prompts.append("\n".join(ctx) + f"\n{u}::{m}::")
        same = [int(c.split("::")[2]) for c in ctx if int(c.split("::")[0]) == u]
        ctrl.append({"truth": lines[k].split("::"), "mode": Counter(same).most_common(1)[0][0] if same else 4,
                     "prev_ts": ctx[-1].split("::")[3] if ctx else ""})
    outs = generate(prompts, 16)
    per = []
    for o, c in zip(outs, ctrl):
        mt = re.match(r"\s*([1-5])::(\d+)", o)
        r_hat, ts_hat = (int(mt.group(1)), mt.group(2)) if mt else (None, None)
        _, _, r, ts = c["truth"]
        per.append({"rating_exact": r_hat == int(r), "mode_exact": c["mode"] == int(r), "ts_exact": ts_hat == ts,
                    "prev_ts_exact": c["prev_ts"] == ts,
                    "ts_within_60s": ts_hat is not None and abs(int(ts_hat) - int(ts)) <= 60, "parsed": mt is not None})
    res["ml_ratings"] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    res["ml_ratings"]["n"] = len(per)
    res["ml_ratings"]["per_interaction"] = per
    print(a.model, "ml_ratings", {k: round(v, 3) for k, v in res["ml_ratings"].items() if k != "per_interaction"}, flush=True)
    # ------------------------------------------------------------------ Amazon items
    dfa, tea = test_split("amazon")
    asins = sorted({str(dfa.at[i, "item"]) for i in tea})
    titles = amazon_titles(asins)
    asins = [x for x in asins if x in titles][: a.limit]
    pool = amazon_titles_any(5, titles)
    head = "".join(f"ASIN: {x}\nTitle: {t}\n\n" for x, t in pool)
    outs = generate([head + f"ASIN: {x}\nTitle:" for x in asins], 32)
    fl = {}
    for x, o in zip(asins, outs):
        g = o.strip().split("\n")[0]
        fl[x] = {"exact": norm(g) == norm(titles[x]),
                 "fuzzy": difflib.SequenceMatcher(None, norm(g), norm(titles[x])).ratio() >= 0.8, "guess": g}
    res["az_items"] = {"n": len(fl), "exact": float(np.mean([f["exact"] for f in fl.values()])),
                       "fuzzy": float(np.mean([f["fuzzy"] for f in fl.values()])), "flags": fl}
    print(a.model, "az_items", len(fl), f"exact {res['az_items']['exact']:.3f} fuzzy {res['az_items']['fuzzy']:.3f}")
    os.makedirs(os.path.join(HERE, "results", "contamination"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "contamination", f"{a.model}.json"), "w"))


def amazon_titles_any(k, known):
    """k (asin, title) few-shot pairs that are not test items: read from the metadata file."""
    out = []
    with gzip.open(os.path.join(RAW, "amazon", "meta_Books.json.gz"), "rt") as f:
        for line in f:
            m = json.loads(line)
            if m.get("title") and m.get("asin") not in known and len(m["title"]) < 80:
                out.append((m["asin"], m["title"]))
                if len(out) == k:
                    return out
    return out


if __name__ == "__main__":
    main()
