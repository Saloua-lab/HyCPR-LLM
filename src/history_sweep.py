"""Measure MAE/RMSE as a function of the number of history items in the prompt (paper Fig. 6).

Usage:
  python history_sweep.py --data ../small_data/LocationLDOSWithHisAvecMistral \
      --adapter ../small_data/LDOS-modelWithHistMistral --out sweep_LDOS.json [--ks 0 1 2 3 4 5] [--limit N]

For each k, the "### History" section of every test prompt is cut to its first k
items (k = 0 -> "No history available", the notebook's text for users without
history); the model and adapter stay the same, so only the prompt changes.
Prompting and rating extraction follow NewEvaluation18Nov_Updated.ipynb cell 44:
"[INST] {prompt} [/INST]", greedy decoding, 30 new tokens, same regex list,
predictions outside 1..5 are dropped.
"""
import argparse
import json
import math
import re
import time

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM

from tokenization import load_tokenizer

PATTERNS = [
    r'Prediction:\s*(\d+)',
    r'Response:\s*(\d+)',
    r'rate the movie as: (\d+)(.*)',
    r'a (\d+) out of 5',
    r'rate the movie as (\d+)',
    r'(\d+)',
    r'rate the restaurant as: (\d+)(.*)',
    r'\s(\d+)',
]
HISTORY_RE = re.compile(r'(### History:?\n)(.*?)(\n+### (?:Response|Prediction):)', re.S)


def extract_rating(text):
    for pattern in PATTERNS:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def prompt_with_history(example, k):
    """Return the model input for `example` with its history cut to k items."""
    start = example.find("[INST]") + len("[INST]")
    end = example.find("### Prediction:")
    if end == -1:
        end = example.find("### Response:")
    end_marker = example[end:].split(":", 1)[0] + ":"
    body = example[start:end] + end_marker

    def cut(m):
        items = [b.strip() for b in m.group(2).split("\n\n") if b.strip()]
        kept = "\n\n".join(items[:k]) if k > 0 else "No history available"
        return m.group(1) + kept + m.group(3)

    body, n = HISTORY_RE.subn(cut, body)
    if n != 1:
        raise ValueError("history section not found in example")
    # cell 44 slices up to "### Response:" and strips; the marker stays out of the prompt
    return body[: body.rfind("###")].strip()


def history_length(example):
    m = HISTORY_RE.search(example)
    items = [b for b in m.group(2).split("\n\n") if b.strip()]
    return 0 if items == ["No history available"] else len(items)


def metrics(pairs):
    if not pairs:
        return None, None
    mae = sum(abs(t - p) for t, p in pairs) / len(pairs)
    rmse = math.sqrt(sum((t - p) ** 2 for t, p in pairs) / len(pairs))
    return mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    examples = load_from_disk(args.data)["test"]["example"]
    if args.limit:
        examples = examples[: args.limit]
    truths = [extract_rating(e) for e in examples]

    tokenizer = load_tokenizer(args.base)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    results = {"data": args.data, "adapter": args.adapter, "base": args.base,
               "history_lengths": [history_length(e) for e in examples], "runs": {}}
    for k in args.ks:
        t0 = time.time()
        prompts = [f"[INST] {prompt_with_history(e, k)} [/INST]" for e in examples]
        outputs = []
        for i in range(0, len(prompts), args.batch):
            batch = tokenizer(prompts[i:i + args.batch], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=30, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            outputs += tokenizer.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        preds = [extract_rating(o.strip()) for o in outputs]
        pairs = [(t, p) for t, p in zip(truths, preds)
                 if t is not None and p is not None and 1 <= t <= 5 and 1 <= p <= 5]
        mae, rmse = metrics(pairs)
        results["runs"][str(k)] = {"truth": truths, "pred": preds, "raw": outputs,
                                   "valid": len(pairs), "mae": mae, "rmse": rmse}
        print(f"k={k}: valid {len(pairs)}/{len(examples)}  MAE {mae:.3f}  RMSE {rmse:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(args.out, "w"))


if __name__ == "__main__":
    main()
