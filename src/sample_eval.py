"""Black-box (sampling-based) rating prediction, without access to the token probabilities
(second review: "the approach requires access to LLM token probabilities ... may not generalize to black-box APIs").

Same adapters, prompts and 4-bit setup as evaluate_probs.py, but the rating is obtained only from generated text, as
through an API without logit access: N answers are sampled at temperature 1 (unmodified next-token distribution,
max 4 new tokens), the first digit 1-5 of each answer is parsed, and the prediction is the mean of the first n valid
samples (a Monte-Carlo estimate of the expected rating; n = 1, 3, 5, 10, 20). Unparseable answers are counted and
skipped; a prompt without any valid sample gets the mean training rating of the column (counted).

Usage: HF_HOME=../.hf_cache python sample_eval.py --data <DatasetDict> --adapter <adapter> \
          --base <hf id> --out probs/sampled/<name>.json [--split test] [--n 20] [--limit N]
Output: {truth, samples (n x N, None = unparseable), ...}
"""
import argparse
import json
import os
import re
import sys

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_probs import split_example  # noqa: E402
from tokenization import load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    ex = load_from_disk(a.data)[a.split]["example"]
    if a.limit:
        ex = ex[: a.limit]
    prompts, truth = zip(*(split_example(e) for e in ex))
    keep = [i for i, t in enumerate(truth) if t in (1, 2, 3, 4, 5)]
    prompts, truth = [prompts[i] for i in keep], [truth[i] for i in keep]
    tok = load_tokenizer(a.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=quant, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    torch.manual_seed(0)
    samples, raw0 = [], []
    for i in range(0, len(prompts), a.batch):
        enc = tok(list(prompts[i:i + a.batch]), return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=4,
                               num_return_sequences=a.n, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j in range(len(txt) // a.n):
            row = []
            for t in txt[j * a.n:(j + 1) * a.n]:
                m = re.match(r"\s*([1-5])(?!\d)", t)
                row.append(int(m.group(1)) if m else None)
            samples.append(row)
            raw0.append(txt[j * a.n])
        print(f"{i + a.batch}/{len(prompts)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"data": a.data, "adapter": a.adapter, "base": a.base, "split": a.split, "n": a.n, "truth": truth,
               "samples": samples, "first_raw": raw0[:50]}, open(a.out, "w"))
    valid = sum(x is not None for r in samples for x in r) / (len(samples) * a.n)
    print(f"done: {len(samples)} prompts, valid samples {valid:.1%}")


if __name__ == "__main__":
    main()
