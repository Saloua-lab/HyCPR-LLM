"""LoRA fine-tuning with the recipe shared by every adapter in small_data/ (FinetuningWithprompts23Nov*.ipynb):
4-bit nf4 base model, LoRA r=32 / alpha=64 / dropout 0.05 on all attention and MLP projections plus lm_head,
10 epochs, batch 16, lr 2e-4, cosine schedule with 3% warm-up, weight decay 0.001, max grad norm 0.3,
paged AdamW, loss on the whole `example` text of the train split (SFTTrainer with dataset_text_field, no packing).

Usage:
  python train.py --data ../small_data/LocationLDOSWithH --base deepseek-ai/deepseek-llm-7b-chat \
      --out adapters/deepseek_LDOS-H [--epochs 10] [--seed 42] [--limit N] [--no-4bit]

Writes the final adapter to --out (adapter_config.json + weights) and train_log.json with the loss curve.

Options of the improvement experiments (defaults reproduce the recipe above):
  --loss rating        loss on the answer only (rating token and what follows it), not on the prompt
  --eval-split NAME    evaluate the rating-only loss on that split after every epoch and keep the best epoch
"""
import argparse
import json
import os
import sys

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_probs import split_example
from tokenization import load_tokenizer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
# MPT (RecGPT-7B): fused query/key/value and output projections, MLP up/down; the output layer is tied to the embeddings
TARGETS_MPT = ["Wqkv", "out_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training examples")
    ap.add_argument("--no-4bit", action="store_true", help="bf16 base model instead of 4-bit nf4")
    ap.add_argument("--loss", choices=["all", "rating"], default="all")
    ap.add_argument("--eval-split", default=None)
    args = ap.parse_args()
    set_seed(args.seed)

    tokenizer = load_tokenizer(args.base)
    tokenizer.padding_side = "right"

    data = load_from_disk(args.data)
    train = data["train"]
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))

    def encode(batch, answer_only):
        enc = tokenizer(batch["example"], truncation=True, max_length=args.max_len, return_offsets_mapping=True)
        if answer_only:
            enc["labels"] = []
            for ex, ids, offs in zip(batch["example"], enc["input_ids"], enc["offset_mapping"]):
                cut = len(split_example(ex)[0])  # characters of the prompt, up to and including the answer marker
                lab, on = [], False
                for t, (st, en) in zip(ids, offs):
                    if en > st:          # a real text token: in the answer iff it starts after the prompt
                        on = st >= cut
                    lab.append(t if on else -100)   # special tokens (e.g. a final </s>) follow the previous token
                assert any(l != -100 for l in lab), "answer truncated away"
                enc["labels"].append(lab)
        enc.pop("offset_mapping")
        return enc

    train = train.map(lambda b: encode(b, args.loss == "rating"), batched=True, remove_columns=train.column_names)
    evals = None
    if args.eval_split:
        evals = data[args.eval_split].map(lambda b: encode(b, True), batched=True,
                                          remove_columns=data[args.eval_split].column_names)

    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant,
                                                 dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    if quant:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                                             target_modules=TARGETS_MPT if model.config.model_type == "mpt" else TARGETS, task_type="CAUSAL_LM"))

    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "trainer"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, learning_rate=args.lr, weight_decay=0.001,
        max_grad_norm=0.3, lr_scheduler_type="cosine", warmup_steps=0.03,  # float = ratio of total steps
        optim="paged_adamw_32bit", bf16=True, train_sampling_strategy="group_by_length", logging_steps=25,
        # checkpoints so that a run killed with the UCloud job resumes instead of restarting
        save_strategy="epoch" if evals is not None else "steps", save_steps=200,
        save_total_limit=2 if evals is not None else 1, report_to=[],
        eval_strategy="epoch" if evals is not None else "no", per_device_eval_batch_size=args.batch,
        load_best_model_at_end=evals is not None, metric_for_best_model="eval_loss", greater_is_better=False,
        seed=args.seed, gradient_checkpointing=True)
    collator = (DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100) if args.loss == "rating" or evals is not None
                else DataCollatorForLanguageModeling(tokenizer, mlm=False))
    if args.loss == "all" and evals is not None:  # the train set needs labels for the seq2seq collator
        train = train.map(lambda b: {"labels": b["input_ids"]}, batched=True)
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=evals, data_collator=collator)
    resume = os.path.isdir(targs.output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(targs.output_dir))
    trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(args.out)
    json.dump({"args": vars(args), "n_train": len(train), "best_checkpoint": trainer.state.best_model_checkpoint,
               "log": trainer.state.log_history},
              open(os.path.join(args.out, "train_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
