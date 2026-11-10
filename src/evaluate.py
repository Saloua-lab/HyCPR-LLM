"""Evaluate one LoRA adapter on one test set, with the same prompting and rating extraction as
history_sweep.py (NewEvaluation18Nov_Updated.ipynb cell 44), for test sets with or without history.

Usage:
  python evaluate.py --data ../small_data/LocationLDOSWithoutHis --adapter ../small_data/LDOS-modelSansHist \
      --base meta-llama/Llama-2-7b-chat-hf --out sweep/llama_LDOS-WH.json [--limit N]

The output has the same layout as history_sweep.py (one run, key "full"), so compare_with_log.py
works on it: python compare_with_log.py sweep/llama_LDOS-WH.json full "<results_*.txt>"
"""
import argparse
import json
import os
import sys
import time

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_sweep import extract_rating, metrics
from tokenization import load_tokenizer


def prompt_of(example):
    """Model input for `example`: the text between [INST] (or the start, if there is none) and the answer
    marker, as in cell 44."""
    start = example.find("[INST]")
    start = 0 if start == -1 else start + len("[INST]")
    end = example.find("### Prediction:")
    if end == -1:
        end = example.find("### Response:")
    return example[start:end].strip()


def raw_prompt_of(example):
    """Model input for adapters trained without the [INST] wrapper (FirstEval MovieLens): the example up to and
    including its answer marker."""
    marker = "### Prediction:" if "### Prediction:" in example else "### Response:"
    return example[: example.find(marker) + len(marker)] + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--labels", type=int, nargs=2, default=[1, 5], metavar=("LO", "HI"),
                    help="valid answer range (MIND clicks: 0 1)")
    ap.add_argument("--raw-prompt", action="store_true",
                    help="prompt with the example text up to its answer marker, without [INST] ... [/INST]")
    args = ap.parse_args()

    examples = load_from_disk(args.data)["test"]["example"]
    if args.limit:
        examples = examples[: args.limit]
    truths = [extract_rating(e) for e in examples]

    tokenizer = load_tokenizer(args.base)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    t0 = time.time()
    if args.raw_prompt:
        prompts = [raw_prompt_of(e) for e in examples]
    else:
        prompts = [f"[INST] {prompt_of(e)} [/INST]" for e in examples]
    outputs = []
    for i in range(0, len(prompts), args.batch):
        batch = tokenizer(prompts[i:i + args.batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        outputs += tokenizer.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    preds = [extract_rating(o.strip()) for o in outputs]
    lo, hi = args.labels
    pairs = [(t, p) for t, p in zip(truths, preds)
             if t is not None and p is not None and lo <= t <= hi and lo <= p <= hi]
    mae, rmse = metrics(pairs)
    json.dump({"data": args.data, "adapter": args.adapter, "base": args.base,
               "runs": {"full": {"truth": truths, "pred": preds, "raw": outputs,
                                 "valid": len(pairs), "mae": mae, "rmse": rmse}}},
              open(args.out, "w"))
    print(f"valid {len(pairs)}/{len(examples)}  MAE {mae:.3f}  RMSE {rmse:.3f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
