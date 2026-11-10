"""ReAct with real tools (review: "the ReAct tool usage is not clearly described, nor its measured contribution").

The ReAct variant of the paper (original notebook) had no tools: the model wrote
its Thought / Action / Observation lines itself, so the observations were invented. Here the observations come
from tools that are executed on the TRAINING interactions of the dataset (never on the test set):

  user_ratings[]     number, mean and distribution of the user's training ratings, with up to 5 rated items
  item_ratings[]     number and mean of the training ratings of the target item (other users)
  similar_items[]    the user's training ratings of items that share an attribute with the target item: a genre
                     or an actor (LDOS: Genre1-3, Actor1-3) or the director or a cast member (DePaul); the
                     restaurant data has no item attributes, so the tool reports that none are available
  context_ratings[]  the user's mean training rating under the same context, per context variable and for the
                     full context, plus the mean rating of all users under the same full context

Loop: the model writes "Thought k: ... / Action k: tool[]"; generation stops at "Observation", the tool is run
and "Observation k: <result>" is appended; the episode ends with "Action k: Finish[r]" (r in 1..5, decimals
allowed) or after MAX_STEPS steps. Greedy decoding, Mistral-7B-Instruct-v0.2 without adapter (zero-shot).

Configurations (--config):
  direct          the dataset's Alpaca prompt, rating read from the generated text (as Table tab:prompting)
  react_notools   ReAct prompt without tools: the model writes its own observations (the previous variant)
  react_all       ReAct with the four tools
  react_minus_<t> ReAct with all tools except <t> (user, item, similar, context)
  direct_tools    no ReAct: the outputs of the four tools are written into the prompt, one generation (30 tokens)
Every output records its cost: LLM calls, prompt and generated tokens, wall-clock seconds (batch of --batch).

Output: results/react/<dataset>_<config>.json with truth, pred (None = no usable rating), steps, tool calls,
and the first 20 transcripts.

Usage: HF_HOME=../.hf_cache python react_tools.py --dataset LDOS --config react_all [--limit N]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TOOLS = {
    "user_ratings": "returns the user's past ratings (count, mean, distribution and a few rated items)",
    "item_ratings": "returns the ratings that other users gave to this item (count and mean)",
    "similar_items": "returns the user's past ratings of items similar to this one (same genre, actor, "
                     "director or cast member)",
    "context_ratings": "returns how the user rated items in the same context, and the mean rating of all users "
                       "in that context",
}
ABLATE = {"user": "user_ratings", "item": "item_ratings", "similar": "similar_items", "context": "context_ratings"}
MAX_STEPS = 6
ATTR_KEYS = {"LDOS": ["Genre1", "Genre2", "Genre3", "Actor1", "Actor2", "Actor3"], "Depaul": ["Director", "Cast"],
             "Tijuana": []}


def attributes(details, dataset):
    """Set of (kind, value) attributes of an item from its 'Key: value, ...' description."""
    keys = ATTR_KEYS[dataset]
    if not keys:
        return set()
    out = set()
    # split on ", Key:" boundaries so that comma-separated cast lists stay together
    parts = re.split(r",\s*(?=[A-Za-z0-9]+:)", details)
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k, v = k.strip(), v.strip()
        if k not in keys or not v or v.lower() in ("nan", "none", "unknown"):
            continue
        kind = re.sub(r"\d", "", k)
        for x in (v.split(",") if k == "Cast" else [v]):
            if x.strip():
                out.add((kind if kind != "Cast" else "cast member", x.strip()))
    return out


def item_name(details):
    m = re.match(r"Name:\s*(.*?)(,\s*[A-Za-z0-9]+:|$)", details)
    return m.group(1).strip() if m else details[:40]


class Toolbox:
    def __init__(self, train, dataset, ctx):
        self.t, self.ds, self.ctx = train, dataset, ctx
        self.attr = {i: attributes(i, dataset) for i in train["i"].unique()}

    def run(self, tool, row):
        t, u, i = self.t, row["u"], row["i"]
        f = lambda x: f"{x:.2f}"
        if tool == "user_ratings":
            d = t[t.u == u]
            if d.empty:
                return "The user has no past ratings."
            dist = Counter(int(r) for r in d.r)
            ex = "; ".join(f"{item_name(a)}: {int(b)}" for a, b in zip(d.i.head(5), d.r.head(5)))
            return (f"The user rated {len(d)} items, mean rating {f(d.r.mean())}; distribution "
                    + ", ".join(f"{k}: {dist.get(k, 0)}" for k in range(1, 6)) + f". Examples: {ex}.")
        if tool == "item_ratings":
            d = t[(t.i == i) & (t.u != u)]
            if d.empty:
                return "No other user has rated this item."
            return f"{len(d)} other users rated this item, mean rating {f(d.r.mean())}."
        if tool == "similar_items":
            a = attributes(i, self.ds)
            if not a:
                return "No item attributes (genre, actors, director) are available for this item."
            d = t[(t.u == u) & (t.i != i)]
            hits = [(x, r, self.attr.get(x, set()) & a) for x, r in zip(d.i, d.r)]
            hits = [h for h in hits if h[2]]
            if not hits:
                return "The user has not rated any item sharing a genre, actor, director or cast member with this one."
            shared = Counter(v for _, _, s in hits for v in s)
            top = ", ".join(f"{k} {v}" for (k, v), _ in shared.most_common(3))
            ex = "; ".join(f"{item_name(x)}: {int(r)}" for x, r, _ in hits[:5])
            return (f"The user rated {len(hits)} similar items (shared: {top}), mean rating "
                    f"{f(np.mean([r for _, r, _ in hits]))}. Examples: {ex}.")
        if tool == "context_ratings":
            d = t[t.u == u]
            parts = []
            for c in self.ctx:
                m = d[d[c] == row[c]]
                if len(m):
                    parts.append(f"{c[2:]} = {row[c]}: {len(m)} ratings, mean {f(m.r.mean())}")
            full = d[(d[self.ctx] == row[self.ctx].values).all(1)] if len(d) else d
            allc = t[(t[self.ctx] == row[self.ctx].values).all(1)]
            s = ("The user's ratings in the same context: " + "; ".join(parts) + "." if parts
                 else "The user has no past ratings in this context.")
            if len(full):
                s += f" Exactly the same context: {len(full)} ratings, mean {f(full.r.mean())}."
            if len(allc):
                s += f" All users in exactly the same context: {len(allc)} ratings, mean {f(allc.r.mean())}."
            return s
        return f"Unknown tool {tool}. Available tools are listed above."


def react_prompt(task, tools):
    if tools:
        lines = "\n".join(f"- {k}[]: {v}" for k, v in tools.items())
        return (f"{task}\n\nSolve this task in the ReAct format, alternating Thought, Action and Observation "
                f"steps. You can use these tools, which look up the past ratings of the recommender's database:\n"
                f"{lines}\n- Finish[r]: give the final answer, the predicted rating r between 1 and 5.\n"
                f"Write one Thought and one Action per step, for example:\nThought 1: ...\nAction 1: "
                f"{next(iter(tools))}[]\nThe Observation is then returned to you. Use at most {MAX_STEPS - 1} "
                f"tools, then finish with \"Action N: Finish[r]\".")
    return ("{task}\n\nAnswer in the ReAct format. Alternate Thought, Action and Observation lines, for example:\n"
            "Thought 1: I should analyze the user's profile and whether they typically enjoy this kind of item.\n"
            "Action 1: Check general preferences of similar users for this genre.\nObservation 1: ...\n"
            "Thought 2: Consider the context.\nAction 2: Assess context compatibility with the item.\n"
            "Observation 2: ...\nFinish with \"Action N: Finish[r]\" where r is the predicted rating, an integer "
            "from 1 to 5.").replace("{task}", task)


FINISH = re.compile(r"Finish\[?\s*([1-5](?:\.\d+)?)")
ACTION = re.compile(r"Action\s*\d*\s*:\s*([A-Za-z_]+)\s*\[")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["LDOS", "Depaul", "Tijuana"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="", help="suffix of the output file (e.g. _timed)")
    a = ap.parse_args()
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM
    from baselines_same_split import SMALL, recover_users
    from evaluate import prompt_of
    from history_sweep import extract_rating
    from tokenization import load_tokenizer


    path, source = SMALL[a.dataset]
    train, test, _ = recover_users(path, source)
    ctx = sorted(c for c in test.columns if c.startswith("c_"))
    for d in (train, test):
        for c in ctx:
            d[c] = d[c].fillna("NA").astype(str) if c in d else "NA"
        d["u"] = d["u"].astype(str)
    examples = load_from_disk(path)["test"]["example"]
    assert len(examples) == len(test)
    n = a.limit or len(test)
    test, examples = test.iloc[:n].reset_index(drop=True), examples[:n]
    box = Toolbox(train, a.dataset, ctx)

    base = "mistralai/Mistral-7B-Instruct-v0.2"
    tok = load_tokenizer(base)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda").eval()

    cost = {"llm_calls": 0, "prompt_tokens": 0, "new_tokens": 0}

    def generate(prompts, max_new_tokens, stop=None):
        out = []
        for k in range(0, len(prompts), a.batch):
            enc = tok(prompts[k:k + a.batch], return_tensors="pt", padding=True).to(model.device)
            kw = dict(stop_strings=stop, tokenizer=tok) if stop else {}
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tok.eos_token_id, **kw)
            new = g[:, enc["input_ids"].shape[1]:]
            out += tok.batch_decode(new, skip_special_tokens=True)
            cost["llm_calls"] += len(new)
            cost["prompt_tokens"] += int(enc["attention_mask"].sum())
            cost["new_tokens"] += int((new != tok.eos_token_id).sum())
        return out

    import time
    t0 = time.time()
    truth = test["r"].astype(float).tolist()
    steps, calls = [0] * n, [Counter() for _ in range(n)]
    if a.config == "direct":
        outs = generate([f"[INST] {prompt_of(e)} [/INST]" for e in examples], 30)
        pred = [extract_rating(o) for o in outs]
        transcripts = outs
    elif a.config == "direct_tools":  # the four tool outputs given directly in the prompt, one call, no ReAct
        def info(j):
            return "\n".join(f"- {t}: {box.run(t, test.iloc[j])}" for t in TOOLS)
        prompts = []
        for j, e in enumerate(examples):
            pr = prompt_of(e)
            prompts.append(f"[INST] {pr}\n\n### Information from the recommender's database (past ratings):\n{info(j)}"
                           f"\n\nOnly respond with the rating. [/INST]")
        outs = generate(prompts, 30)
        pred = [extract_rating(o) for o in outs]
        transcripts = outs
    elif a.config == "react_notools":
        outs = generate([f"[INST] {react_prompt(prompt_of(e), None)} [/INST]" for e in examples], 400)
        pred = []
        for o in outs:
            m = FINISH.search(o)
            pred.append(float(m.group(1)) if m else None)
        transcripts = outs
    else:
        tools = dict(TOOLS)
        if a.config.startswith("react_minus_"):
            del tools[ABLATE[a.config[len("react_minus_"):]]]
        elif a.config != "react_all":
            raise SystemExit(f"unknown config {a.config}")
        heads = [f"[INST] {react_prompt(prompt_of(e), tools)} [/INST]" for e in examples]
        trans = [""] * n
        pred = [None] * n
        open_ = list(range(n))
        for step in range(1, MAX_STEPS + 1):
            if not open_:
                break
            outs = generate([heads[j] + trans[j] for j in open_], 150, stop=["Observation"])
            still = []
            for j, o in zip(open_, outs):
                o = o.split("Observation")[0]
                trans[j] += o
                steps[j] = step
                m = FINISH.search(o)
                if m:
                    pred[j] = float(m.group(1))
                    continue
                act = ACTION.findall(o)
                if not act:
                    continue  # no action and no answer: the episode ends without a rating
                tool = act[-1]
                calls[j][tool] += 1
                obs = box.run(tool, test.iloc[j]) if tool in tools else \
                    f"Unknown tool {tool}. Available tools: {', '.join(tools)}."
                trans[j] = trans[j].rstrip() + f"\nObservation {step}: {obs}\n"
                still.append(j)
            open_ = still
        transcripts = trans
    os.makedirs(os.path.join(HERE, "results", "react"), exist_ok=True)
    pred = [p if p is not None and 1 <= p <= 5 else None for p in pred]
    ok = [(t, p) for t, p in zip(truth, pred) if p is not None]
    e = np.array([p - t for t, p in ok])
    print(f"{a.dataset} {a.config}: usable {len(ok)}/{n}, MAE {np.abs(e).mean():.3f}, RMSE {np.sqrt((e ** 2).mean()):.3f}"
          if ok else f"{a.dataset} {a.config}: no usable rating")
    json.dump({"dataset": a.dataset, "config": a.config, "truth": truth, "pred": pred, "steps": steps,
               "calls": [dict(c) for c in calls], "global_mean": float(train.r.mean()),
               "cost": {**cost, "seconds": time.time() - t0, "n": n, "batch": a.batch,
                        "gpu": torch.cuda.get_device_name(0)},
               "transcripts": transcripts[:20]},
              open(os.path.join(HERE, "results", "react", f"{a.dataset}_{a.config}{a.tag}.json"), "w"))


if __name__ == "__main__":
    main()
