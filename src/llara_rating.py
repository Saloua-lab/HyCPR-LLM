"""LLaRA (Liao et al., SIGIR 2024; official code https://github.com/ljy0ustc/LLaRA, cloned in ../external/LLaRA)
adapted to rating prediction on MovieLens-1M and Amazon-Books (review 4: "missing head-to-head comparisons with
LLaRA, CoLLM, A-LLMRec ...").

Taken unchanged from the official code (imported from ../external/LLaRA):
  - the sequential recommender: recommender/A_SASRec_final_bce_llm.SASRec (hidden 64, state size 10, item id
    `item_num` = padding), whose item embeddings (`cacu_x`) give the item representations;
  - the projector: model/mlp_projector.MlpProjector (64 -> 4096 -> 4096, GELU);
  - the special tokens [PH], [HistoryEmb], [CansEmb], [ItemEmb] added to the tokenizer, and wrap_emb's replacement of
    the [HistoryEmb] / [ItemEmb] token embeddings by the projected item embeddings;
  - the curriculum of data_interface.TrainCollater: at training step s of S, the hybrid prompt (title + [HistoryEmb])
    is used with probability s/S, otherwise the text-only prompt (title + [PH]) with the projector frozen;
  - LoRA r = 8, alpha = 32, dropout 0.1 on q, k, v, o, gate, up, down; Adam, lr 8e-4 with linear warm-up from 8e-6
    over 5% of the steps and cosine decay to 8e-6 (train_movielens.sh); batch 8 x 16 accumulation; bf16 LLM;
    LM loss on the answer only.
Adapted to our task and protocol:
  - the answer is the rating (a single token) instead of a candidate title, and the prompt is our Alpaca prompt
    (the same as for CPR-LLM), with " [HistoryEmb]" after each history item's description and " [ItemEmb]" after the
    target item's description (no candidate list: rating prediction has no candidates);
  - backbone Mistral-7B-Instruct-v0.2 (the LLM of our other comparisons) instead of Llama-2-7B;
  - SASRec is trained on the training period of our chronological split (the repository ships only checkpoints for
    its own item ids, not the training script): every training-period interaction with at least one earlier item is
    a sequence (last 10 earlier items -> next item), BCE on s_fc with one sampled negative, as in its name;
  - 2 epochs over the 50,000 training prompts (the budget of our CPR-LLM models) instead of 5 over their small sets;
  - the rating is read as for every other model: expected rating under the next-token distribution over "1".."5".

Usage: HF_HOME=../.hf_cache python llara_rating.py --dataset movielens [--limit N] [--epochs 2]
Output: cf/<Name>_llara.npz, adapters/llara_mistral_<Name>-H/, probs/llara_mistral_<Name>-H.json
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LLARA = os.path.join(HERE, "..", "external", "LLaRA")
sys.path.insert(0, HERE)
sys.path.insert(0, LLARA)
from evaluate_probs import label_token_ids, metrics, split_example  # noqa: E402
from model.mlp_projector import MlpProjector  # noqa: E402  (official)
from recommender.A_SASRec_final_bce_llm import SASRec  # noqa: E402  (official)

NAMES = {"movielens": "MovieLens", "amazon": "AmazonBooks"}
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
STATE = 10


def prepare(dataset, dev):
    """Item ids of the target and history items of every prompt (checked against the prompt text) and the
    official SASRec trained on the training period."""
    name = NAMES[dataset]
    path = os.path.join(HERE, "cf", f"{name}_llara.npz")
    if os.path.exists(path):
        d = dict(np.load(path))
        return d
    import large_baselines as lb
    from datasets import load_from_disk
    cap = lb.rebuild(dataset)
    df, tr, te = cap["df"], list(cap["tr"]), list(cap["te"])
    lb.check(name, df, te)
    cut = int(len(df) * 0.9)
    iid = {x: k for k, x in enumerate(df["item"].unique())}
    n_items = len(iid)
    I = df["item"].map(iid).to_numpy(np.int64)
    R = df["label"].to_numpy(np.int64)
    by_user = defaultdict(list)
    for j, (u, t) in enumerate(zip(df["user"], df["ts"])):
        by_user[u].append((t, j))
    data = load_from_disk(os.path.join(HERE, "data", f"{name}-H"))
    out = {"n_items": np.array(n_items)}
    for split, idx, exs in (("train", tr, data["train"]["example"]), ("test", te, data["test"]["example"])):
        H = np.full((len(idx), cap["k"]), -1, np.int64)
        for r, (i, e) in enumerate(zip(idx, exs)):
            u, t = df.at[i, "user"], df.at[i, "ts"]
            h = [j for tj, j in by_user[u] if tj < t][-cap["k"]:]
            written = [int(x) for x in re.findall(r"rated it as: (\d) out of 5", split_example(e)[0])]
            assert written == [int(R[j]) for j in h], (split, r)
            H[r, :len(h)] = I[h]
        out[f"{split}_hist"], out[f"{split}_target"] = H, I[idx]
        print(f"{split}: {len(idx)} prompts aligned", flush=True)
    # SASRec (official class) on training-period sequences: last STATE earlier items -> next item
    states, targets = [], []
    for v in by_user.values():
        items = [int(I[j]) for tj, j in v if j < cut]
        for p in range(1, len(items)):
            states.append(items[max(0, p - STATE):p])
            targets.append(items[p])
    rng = np.random.default_rng(0)
    if len(states) > 3_000_000:
        keep = rng.choice(len(states), 3_000_000, replace=False)
        states, targets = [states[k] for k in keep], [targets[k] for k in keep]
    S = np.full((len(states), STATE), n_items, np.int64)  # right padding with item_num, as the official data
    L = np.array([len(s) for s in states], np.int64)
    for k, s in enumerate(states):
        S[k, :len(s)] = s
    Tg = np.array(targets, np.int64)
    torch.manual_seed(0)
    rec = SASRec(64, n_items, STATE, 0.1, dev).to(dev)
    opt = torch.optim.Adam(rec.parameters(), lr=1e-3)
    epochs = 10 if dataset == "movielens" else 2
    St, Lt, Tt = map(torch.as_tensor, (S, L, Tg))
    for ep in range(epochs):
        perm, tot = torch.randperm(len(St)), 0.0
        rec.train()
        for b in range(0, len(St), 1024):
            ix = perm[b:b + 1024]
            logits = rec(St[ix].to(dev), Lt[ix].to(dev))
            pos = logits.gather(1, Tt[ix].to(dev)[:, None])
            neg = logits.gather(1, torch.randint(0, n_items, (len(ix), 1), device=dev))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ix)
        print(f"SASRec epoch {ep + 1}/{epochs}: BCE {tot / len(St):.4f}", flush=True)
    rec.eval()
    out["item_emb"] = rec.item_embeddings.weight.detach().cpu().numpy()  # cacu_x = item_embeddings
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


def hybrid(prompt, text_only):
    """Our prompt with LLaRA's item tokens: [HistoryEmb] after each history item, [ItemEmb] after the target item;
    the text-only curriculum variant uses [PH] instead (official TrainCollater)."""
    tok_h, tok_i = ("[PH]", "[PH]") if text_only else ("[HistoryEmb]", "[ItemEmb]")
    p = re.sub(r"(### (?:Movie|Book) Details:\n[^\n]*)", lambda m: m.group(1) + " " + tok_i, prompt, count=1)
    return p.replace(". The user rated it as:", f" {tok_h}. The user rated it as:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(NAMES))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--lr", type=float, default=8e-4, help="peak lr (official 8e-4 diverged on Mistral-7B)")
    a = ap.parse_args()
    dev = "cuda"
    name = NAMES[a.dataset]
    cf = prepare(a.dataset, dev)
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from tokenization import load_tokenizer
    data = load_from_disk(os.path.join(HERE, "data", f"{name}-H"))
    tr_p, tr_y = zip(*(split_example(e) for e in data["train"]["example"]))
    te_p, te_y = zip(*(split_example(e) for e in data["test"]["example"]))
    n_tr, n_te = a.limit or len(tr_p), a.limit or len(te_p)
    tok = load_tokenizer(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": ["[PH]", "[HistoryEmb]", "[CansEmb]", "[ItemEmb]"]})
    his_id, item_id = tok.convert_tokens_to_ids("[HistoryEmb]"), tok.convert_tokens_to_ids("[ItemEmb]")
    lab = torch.tensor(label_token_ids(tok, hybrid(te_p[0], False), [1, 2, 3, 4, 5]), device=dev)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.resize_token_embeddings(len(tok))
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1,
                                             target_modules=["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    proj = MlpProjector(rec_size=64, llm_size=model.config.hidden_size).to(dev, torch.bfloat16)
    item_emb = torch.as_tensor(cf["item_emb"], device=dev, dtype=torch.bfloat16)

    def wrap_emb(enc, hist, target):
        """Official wrap_emb: token embeddings, then [HistoryEmb] / [ItemEmb] replaced by projected item embeddings."""
        emb = model.get_input_embeddings()(enc["input_ids"]).clone()  # clone: in-place replacement below
        ids = enc["input_ids"]
        for r in range(len(ids)):
            h = hist[r][hist[r] >= 0]
            pos = (ids[r] == his_id).nonzero().view(-1)
            if len(pos):
                assert len(pos) == len(h), (len(pos), len(h))
                emb[r, pos] = proj(item_emb[torch.as_tensor(h, device=dev)]).to(emb.dtype)
            pos = (ids[r] == item_id).nonzero().view(-1)
            if len(pos):
                emb[r, pos] = proj(item_emb[int(target[r])][None]).to(emb.dtype)
        return emb

    tag = "" if a.lr == 8e-4 else f"_lr{a.lr:g}"
    out_dir = os.path.join(HERE, "adapters", f"llara{tag}_mistral_{name}-H" + ("_smoke" if a.limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "projector.pt")
    if os.path.exists(done):
        from peft import set_peft_model_state_dict
        proj.load_state_dict(torch.load(done))
        set_peft_model_state_dict(model, torch.load(os.path.join(out_dir, "lora.pt")))
        print("loaded", out_dir, flush=True)
    else:
        params = [{"params": proj.parameters(), "lr": a.lr, "weight_decay": 1e-5},
                  {"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr}]
        opt = torch.optim.Adam(params)
        steps = a.epochs * (n_tr // a.batch) // a.accum
        warm, lo, hi = max(1, steps // 20), a.lr / 100, a.lr

        def set_lr(s):  # official LinearWarmupCosineLRScheduler
            lr = lo + (hi - lo) * s / warm if s < warm else lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
            for g in opt.param_groups:
                g["lr"] = lr

        random.seed(1234)
        torch.manual_seed(1234)
        cur, max_batches, step, run = 1, a.epochs * (n_tr // a.batch), 0, []
        skipped, skipped_steps = 0, 0
        model.train()
        for ep in range(a.epochs):
            order = list(range(n_tr))
            random.shuffle(order)
            for k in range(0, n_tr - a.batch + 1, a.batch):
                b = order[k:k + a.batch]
                text_only = random.random() >= cur / max_batches  # official curriculum
                cur += 1
                prompts = [hybrid(tr_p[j], text_only) for j in b]
                enc = tok([p + str(tr_y[j]) + "\n" for p, j in zip(prompts, b)], return_tensors="pt", padding=True).to(dev)
                plen = [len(tok(p)["input_ids"]) for p in prompts]
                labels = enc["input_ids"].clone()
                for r, n in enumerate(plen):
                    labels[r, :n] = -100
                labels[enc["attention_mask"] == 0] = -100
                for p in proj.parameters():
                    p.requires_grad_(not text_only)
                emb = wrap_emb(enc, cf["train_hist"][b], cf["train_target"][b])
                loss = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], labels=labels).loss / a.accum
                if torch.isfinite(loss):  # guard: a non-finite batch is skipped (as in collm.py)
                    loss.backward()
                    run.append(float(loss.detach()) * a.accum)
                else:
                    skipped += 1
                if (k // a.batch + 1) % a.accum == 0:
                    set_lr(step)
                    trainable = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
                    if all(torch.isfinite(p.grad).all() for p in trainable):
                        opt.step()
                    else:
                        skipped_steps += 1
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 10 == 0:
                        print(f"epoch {ep + 1} step {step}/{steps} loss {np.mean(run[-160:]):.4f} "
                              f"hybrid share {cur / max_batches:.2f} skipped {skipped}/{skipped_steps}", flush=True)
        from peft import get_peft_model_state_dict
        torch.save(proj.state_dict(), done)
        torch.save(get_peft_model_state_dict(model), os.path.join(out_dir, "lora.pt"))
    model.eval()
    probs = []
    with torch.no_grad():
        for k in range(0, n_te, 16):
            b = list(range(k, min(k + 16, n_te)))
            enc = tok([hybrid(te_p[j], False) for j in b], return_tensors="pt", padding=True).to(dev)
            last = enc["attention_mask"].sum(1) - 1
            emb = wrap_emb(enc, cf["test_hist"][b], cf["test_target"][b])
            logits = model(inputs_embeds=emb, attention_mask=enc["attention_mask"]).logits[torch.arange(len(b), device=dev), last].float()
            probs += torch.softmax(logits[:, lab], -1).cpu().tolist()
    truth = list(te_y[:n_te])
    expected = [sum((r + 1) * p for r, p in enumerate(row)) for row in probs]
    mae, rmse = metrics(truth, expected)
    out = os.path.join(HERE, "probs", f"llara{tag}_mistral_{name}-H{'_limit' if a.limit else ''}.json")
    json.dump({"data": f"data/{name}-H", "method": "LLaRA (official modules) adapted to rating prediction",
               "epochs": a.epochs, "truth": truth, "probs": probs, "expected": expected, "n": len(truth),
               "mae": mae, "rmse": rmse}, open(out, "w"))
    print(f"{name} LLaRA: n {len(truth)}  MAE {mae:.4f}  RMSE {rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
