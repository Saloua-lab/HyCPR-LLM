"""Remove test-set leakage from the history section of "-H" test prompts.

A history item is dropped when it is one of the same user's test interactions (same title, context and
rating as the target of any test example of that user), which includes the example's own interaction.
Training interactions are kept. A history left empty becomes "No history available", the notebooks'
text for users without history.

Training prompts carry no user id, so there the user is identified by the "### User Information:" text (users
with identical demographics are merged): a train history item is dropped when it is the example's own interaction
(its answer) or one of that user's test interactions. Prompts without user information (DePaul) cannot be
linked to a user, so only the example's own interaction is dropped there.

Usage: python rebuild_history.py ../small_data/LocationLDOSWithH [more test sets...]
Writes data/<name>_noleak/ next to this script and prints what was removed.
"""
import os
import re
import sys

from datasets import DatasetDict, load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_sweep import HISTORY_RE, extract_rating


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".")


def target_key(example):
    """(user, title, context, rating) of the example's own interaction."""
    user = example.split("<s>", 1)[0].strip()
    head = example[:HISTORY_RE.search(example).start()]
    ctx = re.search(r"### Contextual Information:\n(.*?)\n", head).group(1)
    name = re.findall(r"Name:\s*([^\n]+)", head)[-1].split(",")[0]
    return user, (norm(name), norm(ctx), extract_rating(example))


def profile(example):
    m = re.search(r"### User Information:\n(.*?)\n", example)
    return norm(m.group(1)) if m else None


def item_key(item):
    """(title, context, rating) of one history item, in any of the three dataset formats."""
    ctx = re.search(r"Contextual information:\s*(.*?)(?:\n|\\Restaurant|$)", item).group(1)
    name = re.search(r"Name\s*:\s*(.+?)(?:,| Genre:|\. The user|\n|$)", item).group(1)
    rating = re.search(r"as:? (\d) out of 5", item)
    return norm(name), norm(ctx), int(rating.group(1)) if rating else None


def rebuild(path):
    ds = load_from_disk(path)
    test = ds["test"]["example"]
    test_keys = {}
    for e in test:
        user, key = target_key(e)
        test_keys.setdefault(user, set()).add(key)

    out, removed, self_removed, emptied = [], 0, 0, 0
    for e in test:
        user, own = target_key(e)
        m = HISTORY_RE.search(e)
        items = [b for b in m.group(2).split("\n\n") if b.strip()]
        if items == ["No history available"]:
            out.append(e)
            continue
        keep = [b for b in items if item_key(b) not in test_keys[user]]
        removed += len(items) - len(keep)
        self_removed += any(item_key(b) == own for b in items)
        emptied += not keep
        body = "\n\n".join(keep) if keep else "No history available"
        out.append(e[:m.start(2)] + body + e[m.end(2):])

    train, t_removed, t_self, t_emptied = [], 0, 0, 0
    profile_keys = {}
    for e in test:
        if profile(e):
            profile_keys.setdefault(profile(e), set()).update(test_keys[target_key(e)[0]])
    for e in ds["train"]["example"]:
        own = target_key(e)[1]
        m = HISTORY_RE.search(e)
        items = [b for b in m.group(2).split("\n\n") if b.strip()]
        if items == ["No history available"]:
            train.append(e)
            continue
        keep = [b for b in items if item_key(b) != own and item_key(b) not in profile_keys.get(profile(e), ())]
        t_removed += len(items) - len(keep)
        t_self += any(item_key(b) == own for b in items)
        t_emptied += not keep
        body = "\n\n".join(keep) if keep else "No history available"
        train.append(e[:m.start(2)] + body + e[m.end(2):])

    name = os.path.basename(os.path.normpath(path))
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name + "_noleak")
    DatasetDict({"train": ds["train"].from_dict({"example": train}), "test": ds["test"].from_dict({"example": out})}).save_to_disk(dst)
    print(f"{name}: {len(test)} test prompts, {self_removed} contained their own interaction, "
          f"{removed} history items removed, {emptied} histories now empty; {len(train)} train prompts, "
          f"{t_self} contained their own answer, {t_removed} items removed, {t_emptied} emptied "
          f"-> {os.path.relpath(dst)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rebuild(p)
