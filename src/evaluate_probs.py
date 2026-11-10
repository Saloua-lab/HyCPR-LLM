"""Probability-based evaluation of one LoRA adapter on one test set (the protocol of the revised paper,
Section "Evaluation protocol").

The prompt is the test example exactly as in fine-tuning, cut right after the answer marker ("### Response:\\n"
or "### Prediction:\\n"), with the rating removed. The model's next-token distribution is restricted to the
rating tokens and renormalised; the prediction is its expectation (`expected`), the most probable rating
(`argmax`) is kept for comparison. Every prompt gets a prediction.

Usage:
  python evaluate_probs.py --data ../small_data/LocationLDOSWithoutHis --adapter ../small_data/LDOS-modelSansHist \
      --base meta-llama/Llama-2-7b-chat-hf --out probs/llama_LDOS-WH.json [--limit N] [--batch 16]

Output JSON: {data, adapter, base, labels, truth, probs (n x K), expected, argmax, n, mae, rmse, mae_argmax,
rmse_argmax}.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_sweep import HISTORY_RE
from tokenization import load_tokenizer

MARKER_RE = re.compile(r"### (?:Response|Prediction):[ \t]*\n?")


def split_example(example):
    """(prompt, true label) for a training-format example: the prompt ends right after the last answer marker."""
    m = None
    for m in MARKER_RE.finditer(example):
        pass
    if m is None:
        raise ValueError("no answer marker in example: " + example[-200:])
    truth = re.match(r"\s*(\d+)", example[m.end():])
    return example[:m.end()], int(truth.group(1)) if truth else None


def cut_history(example, k):
    """The example with its history section cut to the first k items ("No history available" for k=0),
    as in history_sweep.prompt_with_history; examples without a history section are returned unchanged."""
    def cut(m):
        items = [b.strip() for b in m.group(2).split("\n\n") if b.strip()]
        if items == ["No history available"]:
            items = []
        kept = "\n\n".join(items[:k]) if k > 0 and items else "No history available"
        return m.group(1) + kept + m.group(3)
    return HISTORY_RE.sub(cut, example, count=1)


def label_token_ids(tokenizer, prompt, labels):
    """Token id of each label as it follows `prompt`: tokenise prompt + label and check that this is the
    prompt's tokens plus exactly one new token, so the scored token is the one the model was trained on."""
    base = tokenizer(prompt)["input_ids"]
    ids = []
    for lab in labels:
        full = tokenizer(prompt + str(lab))["input_ids"]
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"label {lab} is not a single token after the prompt for {tokenizer.name_or_path}: "
                             f"{tokenizer.convert_ids_to_tokens(full[len(base) - 2:])}")
        ids.append(full[-1])
    return ids


def metrics(truth, pred):
    err = [p - t for t, p in zip(truth, pred)]
    return sum(abs(e) for e in err) / len(err), math.sqrt(sum(e * e for e in err) / len(err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="4-bit nf4 base with the adapter on top (as in training) instead of a merged bf16 model")
    ap.add_argument("--labels", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="MIND clicks: --labels 0 1")
    ap.add_argument("--ks", type=int, nargs="+", default=None,
                    help="history curve: also score each prompt with its history cut to the first k items; "
                         "writes <out>_k<k>.json per k (skipped if it exists), same model for every k")
    args = ap.parse_args()

    examples = load_from_disk(args.data)[args.split]["example"]
    if args.limit:
        examples = examples[: args.limit]
    if args.ks:
        jobs = [(args.out.replace(".json", f"_k{k}.json"), [cut_history(e, k) for e in examples]) for k in args.ks]
        jobs = [(o, ex) for o, ex in jobs if not os.path.exists(o)]
    else:
        jobs = [(args.out, examples)]
    first = jobs[0][1] if jobs else examples
    prompts, truth = zip(*(split_example(e) for e in first))
    keep = [i for i, t in enumerate(truth) if t in args.labels]
    if len(keep) < len(truth):
        print(f"skipping {len(truth) - len(keep)} examples whose true label is not in {args.labels}", flush=True)
    prompts, truth = [prompts[i] for i in keep], [truth[i] for i in keep]

    tokenizer = load_tokenizer(args.base)
    tokenizer.padding_side = "left"
    ids = label_token_ids(tokenizer, prompts[0], args.labels)
    for p in prompts[1:50]:
        assert label_token_ids(tokenizer, p, args.labels) == ids
    print("label tokens:", dict(zip(args.labels, tokenizer.convert_ids_to_tokens(ids))), flush=True)

    if args.four_bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant, dtype=torch.bfloat16,
                                                     device_map="cuda")
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    for out, exs in jobs:
        prompts, truth = zip(*(split_example(e) for e in exs))
        score(model, tokenizer, ids, list(prompts), list(truth), args, out)


def score(model, tokenizer, ids, prompts, truth, args, out):
    t0 = time.time()
    keep = [i for i, t in enumerate(truth) if t in args.labels]
    prompts, truth = [prompts[i] for i in keep], [truth[i] for i in keep]
    probs = []
    idx = torch.tensor(ids)
    for i in range(0, len(prompts), args.batch):
        batch = tokenizer(list(prompts[i:i + args.batch]), return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :].float()
        probs += torch.softmax(logits[:, idx.to(logits.device)], dim=-1).cpu().tolist()
    labels = args.labels
    expected = [sum(l * p for l, p in zip(labels, row)) for row in probs]
    argmax = [labels[max(range(len(row)), key=row.__getitem__)] for row in probs]
    mae, rmse = metrics(truth, expected)
    mae_a, rmse_a = metrics(truth, argmax)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump({"data": args.data, "ks": args.ks, "out": out, "adapter": args.adapter, "base": args.base, "four_bit": args.four_bit, "labels": labels,
               "truth": truth, "probs": probs, "expected": expected, "argmax": argmax, "n": len(truth),
               "mae": mae, "rmse": rmse, "mae_argmax": mae_a, "rmse_argmax": rmse_a}, open(out, "w"))
    print(f"{os.path.basename(out)}: n {len(truth)}  expected MAE {mae:.3f} RMSE {rmse:.3f}  |  argmax MAE {mae_a:.3f} RMSE {rmse_a:.3f}"
          f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
