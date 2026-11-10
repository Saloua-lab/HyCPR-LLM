"""Build CPR-LLM prompt datasets for the large datasets of the paper: YELP, MIND, MovieLens-1M, Amazon-Books.

Usage: python prepare_large.py {yelp,mind,movielens,amazon} [--n-train 50000] [--n-test 2000]

Protocol (Kang et al. 2023, arXiv:2305.06474, used for the MovieLens / Amazon-Books baselines of the paper):
global chronological split, the first 90% of interactions are training data, the rest test data; 2,000 random
test interactions are evaluated. Here also --n-train random training interactions are kept (fine-tuning budget).
History = the user's most recent interactions strictly before the target (no leakage by construction): up to
5 items for YELP and MIND (as in the small datasets), the 10 most recent for MovieLens and Amazon-Books (Kang).
No feature computed over the whole dataset (item average stars, user average stars) is put in the prompt.
MIND has clicks, not ratings: per impression one clicked and one non-clicked article, answer 1 (click) or 0.

Writes data/<Name>-H and data/<Name>-WH (DatasetDict, column `example`, same layout as the FirstEval datasets:
train = prompt with answer, test = "<user><s>[INST] prompt[/INST]</s>") and data/<Name>_meta.json with counts.
"""
import argparse
import gzip
import html
import json
import os
import random
import re
import zipfile
from collections import defaultdict

import pandas as pd
from datasets import Dataset, DatasetDict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw_data")

HEAD = ("You are CALLREC, functioning as a Context-Aware Recommendation System.\n"
        "Given {given}, predict {what}.\n\n"
        "Only respond with the {answer} with no text before or after the {answer}.\n\n")


def build(sections, history, answer):
    """Prompt body in the FirstEval Alpaca layout; history=None for the -WH variant."""
    body = "".join(f"### {title}:\n{text}\n\n" for title, text in sections)
    if history is not None:
        body += "### History:\n" + ("\n\n".join(history) if history else "No history available") + "\n\n"
    return body + f"### Response:\n{answer}\n"


def time_context(ts, with_hour=True):
    seasons = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    parts = []
    if with_hour:
        h = ts.hour
        parts.append("time: " + ("Morning" if 5 <= h < 12 else "Afternoon" if 12 <= h < 17
                                 else "Evening" if 17 <= h < 22 else "Night"))
    parts += [f"day: {ts.day_name()}", "daytype: " + ("Weekend" if ts.dayofweek >= 5 else "Working day"),
              f"season: {seasons[ts.month]}"]
    return ", ".join(parts)


def split_and_sample(df, n_train, n_test, seed=0):
    """Global chronological 90/10 split, then random samples of the two parts (Kang et al.)."""
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    cut = int(len(df) * 0.9)
    rng = random.Random(seed)
    train_idx = sorted(rng.sample(range(cut), min(n_train, cut)))
    test_idx = sorted(rng.sample(range(cut, len(df)), min(n_test, len(df) - cut)))
    return df, train_idx, test_idx


def histories(df, idx, k, render):
    """For each row index in idx, the rendered last k interactions of the same user strictly before it
    (df sorted by time; ties at the same timestamp are excluded)."""
    by_user = defaultdict(list)
    for i, (u, t) in enumerate(zip(df["user"], df["ts"])):
        by_user[u].append((t, i))
    out = {}
    for i in idx:
        u, t = df.at[i, "user"], df.at[i, "ts"]
        prior = [j for tj, j in by_user[u] if tj < t][-k:]
        out[i] = [render(df.loc[j]) for j in prior]
    return out


def save(name, df, train_idx, test_idx, prompt, k, render_hist, meta):
    hist = histories(df, train_idx + test_idx, k, render_hist) if k else {}
    for variant in ("H", "WH"):
        def ex(i, test):
            p = prompt(df.loc[i], hist.get(i, []) if variant == "H" else None)
            return f"{df.at[i, 'user']}<s>[INST] {p}[/INST]</s>" if test else p
        dd = DatasetDict({"train": Dataset.from_dict({"example": [ex(i, False) for i in train_idx]}),
                          "test": Dataset.from_dict({"example": [ex(i, True) for i in test_idx]})})
        dd.save_to_disk(os.path.join(HERE, "data", f"{name}-{variant}"))
    with_hist = sum(1 for i in test_idx if (hist.get(i) if k else len(df.at[i, "hist"])))
    meta.update(n_interactions=len(df), n_users=int(df["user"].nunique()), n_items=int(df["item"].nunique()),
                n_train_sample=len(train_idx), n_test_sample=len(test_idx), history_k=k,
                test_with_history=with_hist,
                test_label_counts={str(k_): int(v) for k_, v in df.loc[test_idx, "label"].value_counts().items()})
    json.dump(meta, open(os.path.join(HERE, "data", f"{name}_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print(f"--- example test prompt ({name}-H):\n{prompt(df.loc[test_idx[0]], hist.get(test_idx[0], []))}")


def k_core(df, k):
    while True:
        n = len(df)
        df = df[df.groupby("user")["item"].transform("size") >= k]
        df = df[df.groupby("item")["user"].transform("size") >= k]
        if len(df) == n:
            return df


# --------------------------------------------------------------------------------------------- datasets

def yelp(args):
    src = os.path.join(RAW, "yelp")
    rows = []
    with open(os.path.join(src, "yelp_academic_dataset_review.json")) as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["user_id"], r["business_id"], int(r["stars"]), r["date"]))
    df = pd.DataFrame(rows, columns=["user", "item", "label", "ts"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = k_core(df, args.core)
    items, users = set(df["item"]), set(df["user"])
    biz = {}
    with open(os.path.join(src, "yelp_academic_dataset_business.json")) as f:
        for line in f:
            b = json.loads(line)
            if b["business_id"] in items:
                price = (b.get("attributes") or {}).get("RestaurantsPriceRange2")
                biz[b["business_id"]] = dict(name=b["name"], city=b["city"], state=b["state"],
                                             categories=b.get("categories") or "unknown",
                                             price=price if price not in (None, "None") else "unknown")
    since = {}
    with open(os.path.join(src, "yelp_academic_dataset_user.json")) as f:
        for line in f:  # lines are huge (friend lists): read the id and year without parsing the JSON
            uid = line[len('{"user_id":"'):len('{"user_id":"') + 22]
            if uid in users:
                i = line.find('"yelping_since":"')
                since[uid] = line[i + 17:i + 21]

    def render(r):
        b = biz[r["item"]]
        return f"Name: {b['name']}, Categories: {b['categories']}. The user rated it as: {r['label']} out of 5"

    def prompt(r, hist):
        b = biz[r["item"]]
        given = "the user's information, contextual information, business details" + (" and review history" if hist is not None else "")
        return HEAD.format(given=given, what="on a scale of 1 to 5 how much the user will like the business",
                           answer="rating") + build([
            ("User Information", f"Yelping since: {since.get(r['user'], 'unknown')}"),
            ("Contextual Information", f"{time_context(r['ts'])}, location: {b['city']}, {b['state']}"),
            ("Business Details", f"Name: {b['name']}, Categories: {b['categories']}, Price range: {b['price']}")],
            hist, r["label"])

    df, tr, te = split_and_sample(df, args.n_train, args.n_test)
    save("YELP", df, tr, te, prompt, 5, render, {"dataset": "YELP", "core": args.core})


def movielens(args):
    src = os.path.join(RAW, "movielens", "ml-1m")
    read = lambda f, cols: pd.read_csv(os.path.join(src, f), sep="::", engine="python", names=cols, encoding="latin-1")
    df = read("ratings.dat", ["user", "item", "label", "ts"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    movies = read("movies.dat", ["item", "title", "genres"]).set_index("item")
    users = read("users.dat", ["user", "gender", "age", "occupation", "zip"]).set_index("user")
    ages = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
    occ = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student", "customer service",
           "doctor/health care", "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer",
           "programmer", "retired", "sales/marketing", "scientist", "self-employed", "technician/engineer",
           "tradesman/craftsman", "unemployed", "writer"]

    def render(r):
        m = movies.loc[r["item"]]
        return f"Name: {m['title']}, Genre: {m['genres'].replace('|', ', ')}. The user rated it as: {r['label']} out of 5"

    def prompt(r, hist):
        m, u = movies.loc[r["item"]], users.loc[r["user"]]
        given = "the user's information, contextual information, movie details" + (" and watch history" if hist is not None else "")
        return HEAD.format(given=given, what="on a scale of 1 to 5 how much the user will like the movie",
                           answer="rating") + build([
            ("User Information", f"gender: {'Male' if u['gender'] == 'M' else 'Female'}, age: {ages[u['age']]}, "
                                 f"occupation: {occ[u['occupation']]}"),
            ("Contextual Information", time_context(r["ts"])),
            ("Movie Details", f"Name: {m['title']}, Genre: {m['genres'].replace('|', ', ')}")], hist, r["label"])

    df, tr, te = split_and_sample(df, args.n_train, args.n_test)
    save("MovieLens", df, tr, te, prompt, 10, render, {"dataset": "MovieLens-1M"})


def amazon(args):
    src = os.path.join(RAW, "amazon")
    meta = {}
    with gzip.open(os.path.join(src, "meta_Books.json.gz"), "rt") as f:
        for line in f:
            m = json.loads(line)
            title = html.unescape(m.get("title") or "").strip()
            brand = re.sub(r"^Visit Amazon's (.*) Page$", r"\1", html.unescape(m.get("brand") or "").strip())
            if title and len(title) < 300 and "<" not in title:
                meta[m["asin"]] = dict(title=title, brand=brand or "unknown")
    rows = []
    with gzip.open(os.path.join(src, "Books_5.json.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["asin"] in meta:  # Kang et al.: drop ratings of items without metadata
                rows.append((r["reviewerID"], r["asin"], int(r["overall"]), r["unixReviewTime"]))
    df = pd.DataFrame(rows, columns=["user", "item", "label", "ts"]).drop_duplicates(["user", "item"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s")

    def render(r):
        m = meta[r["item"]]
        return f"Name: {m['title']}, Brand: {m['brand']}. The user rated it as: {r['label']} out of 5"

    def prompt(r, hist):
        m = meta[r["item"]]
        given = "contextual information, book details" + (" and reading history" if hist is not None else "")
        return HEAD.format(given=given, what="on a scale of 1 to 5 how much the user will like the book",
                           answer="rating") + build([
            ("Contextual Information", time_context(r["ts"], with_hour=False)),
            ("Book Details", f"Name: {m['title']}, Brand: {m['brand']}")], hist, r["label"])

    df, tr, te = split_and_sample(df, args.n_train, args.n_test)
    save("AmazonBooks", df, tr, te, prompt, 10, render, {"dataset": "Amazon-Books (2018, 5-core)"})


def mind(args):
    src = os.path.join(RAW, "mind")
    news, rows = {}, []
    for part in ("MINDsmall_train", "MINDsmall_dev"):
        z = zipfile.ZipFile(os.path.join(src, part + ".zip"))
        for line in z.read("news.tsv").decode("utf-8").splitlines():
            nid, cat, sub, title = line.split("\t")[:4]
            news[nid] = dict(cat=cat, sub=sub, title=title)
        for line in z.read("behaviors.tsv").decode("utf-8").splitlines():
            imp, user, t, hist, cands = line.split("\t")
            rows.append((imp, user, t, hist.split() if hist else [], cands.split()))
    rng = random.Random(0)
    recs = []
    for imp, user, t, hist, cands in rows:
        pos = [c[:-2] for c in cands if c.endswith("-1")]
        neg = [c[:-2] for c in cands if c.endswith("-0")]
        if pos and neg:  # one clicked and one non-clicked article per impression
            for item, label in ((rng.choice(pos), 1), (rng.choice(neg), 0)):
                recs.append((user, item, label, t, imp, tuple(hist[-5:])))
    df = pd.DataFrame(recs, columns=["user", "item", "label", "ts", "imp", "hist"])
    df["ts"] = pd.to_datetime(df["ts"], format="%m/%d/%Y %I:%M:%S %p")
    title = lambda n: f"Title: {news[n]['title']}, Category: {news[n]['cat']}, {news[n]['sub']}"

    def prompt(r, hist):
        # MIND gives each impression the user's click history before the logging period; use it instead of
        # interactions rebuilt from the sampled rows (hist is only None / not None here).
        h = None if hist is None else [f"Clicked: {title(n)}" for n in r["hist"]]
        given = "contextual information, news article details" + (" and click history" if h is not None else "")
        return HEAD.format(given=given, what="whether the user will click on the news article: 1 for click, "
                                             "0 for no click", answer="answer") + build([
            ("Contextual Information", time_context(r["ts"])),
            ("News Details", title(r["item"]))], h, r["label"])

    df, tr, te = split_and_sample(df, args.n_train, args.n_test)
    save("MIND", df, tr, te, prompt, 0, None, {"dataset": "MIND-small (train+dev)", "label": "click 1 / no click 0",
                                              "history": "MIND click history before the logging period, last 5"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["yelp", "mind", "movielens", "amazon"])
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--core", type=int, default=20, help="YELP: k-core filter on users and items")
    a = ap.parse_args()
    globals()[a.dataset](a)
