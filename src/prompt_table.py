"""Turn the LLM prompt datasets back into interaction tables (user, item, context..., rating), so that the
classical baselines are trained and tested on exactly the train/test interactions the LLMs see.

user    = the id written before "<s>" at the start of every example
item    = the "Name: ..." line of the item-details section (the whole line, since titles may contain commas)
context = the key: value pairs of the "### Contextual Information:" section
rating  = the label after the answer marker
"""
import re

import pandas as pd
from datasets import load_from_disk

SECTION_RE = re.compile(r"### ([^\n:]+):\n(.*?)(?=\n\n###|\n\n\Z|\Z)", re.S)
ANSWER_RE = re.compile(r"### (?:Response|Prediction):[ \t]*\n?\s*(\d+)")


def parse(example):
    user = example.split("<s>", 1)[0].strip()
    secs = {k.strip(): v.strip() for k, v in SECTION_RE.findall(example.split("### History", 1)[0])}
    details = next(v for k, v in secs.items() if k.endswith("Details"))
    ctx_line = secs.get("Contextual Information", "")
    ctx = dict(kv.split(": ", 1) for kv in re.split(r",\s*(?=[A-Za-z_ ]+: )", ctx_line) if ": " in kv)
    answers = ANSWER_RE.findall(example)
    return {"u": user, "i": details.split("\n")[0], **{f"c_{k.strip()}": v.strip() for k, v in ctx.items()},
            "r": float(answers[-1])}


def table(path, split):
    return pd.DataFrame([parse(e) for e in load_from_disk(path)[split]["example"]]).fillna("NA")


def load(path):
    """(train, test, context columns) for a prompt DatasetDict."""
    tr, te = table(path, "train"), table(path, "test")
    ctx = sorted(c for c in set(tr.columns) | set(te.columns) if c.startswith("c_"))
    for d in (tr, te):
        for c in ctx:
            if c not in d:
                d[c] = "NA"
    return tr, te, ctx


def key(df):
    """Interaction identity used to compare test sets: user, item, context, rating."""
    cols = ["u", "i"] + sorted(c for c in df.columns if c.startswith("c_")) + ["r"]
    return df[cols].astype(str).agg(" | ".join, axis=1)
