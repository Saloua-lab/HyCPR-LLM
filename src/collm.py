"""Input-level LLM-CF fusion in the style of CoLLM (Zhang et al., TKDE 2025), as a controlled comparison with the
prediction-level fusion of HyCPR-LLM (review 4: "missing head-to-head comparisons with LLM-CF integration methods").

CoLLM recipe, adapted to rating prediction:
  stage 1  LoRA fine-tuning of the LLM on the text prompts only: this is exactly our CPR-LLM adapter
           (adapters/mistral_full_<Name>-H), reused unchanged;
  stage 2  the LLM and its LoRA adapter are frozen; the user and item embeddings of a collaborative model
           (BiasedMF, 32 dimensions, trained on the whole training part of the chronological split, as the BiasedMF
           baseline of Table 7) are mapped by a two-layer MLP (32 -> 512 -> hidden size, shared by users and items)
           to two soft tokens inserted in the prompt ("### Collaborative Information: user embedding: <u>, item
           embedding: <i>" before "### User Information:"); only the MLP is trained, on the same 50,000 training
           prompts, with the cross-entropy of the rating token (the LM loss on the answer).
The rating is read exactly as for CPR-LLM: expected rating under the next-token distribution restricted to "1".."5".
Same split, same test prompts, same adapter and same CF model family as HyCPR-LLM, so the only difference is where
the collaborative signal enters (inside the LLM input vs. at the level of the predicted rating).

Usage: HF_HOME=../.hf_cache python collm.py --dataset movielens [--epochs 1] [--limit N] [--batch 8]
Output: cf/<Name>_mf.npz (CF embeddings and prompt-to-id alignment), adapters/collm_mistral_<Name>-H/mapper.pt,
        probs/collm_mistral_<Name>-H.json (same format as evaluate_probs.py)
"""
import argparse
import json
import math
import re
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate_probs import label_token_ids, metrics, split_example  # noqa: E402
from tokenization import load_tokenizer  # noqa: E402

NAMES = {"movielens": "MovieLens", "amazon": "AmazonBooks"}
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
SLOT = "### Collaborative Information:\nuser embedding: <unk>, item embedding: <unk>\n\n### User Information:"


def prepare(dataset, dim=32):
    """BiasedMF embeddings (training part only) and the (user, item) index of every training / test prompt."""
    name = NAMES[dataset]
    path = os.path.join(HERE, "cf", f"{name}_mf.npz" if dim == 32 else f"{name}_mf{dim}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    import large_baselines as lb
    from datasets import load_from_disk
    cap = lb.rebuild(dataset)
    df, tr, te = cap["df"], np.array(cap["tr"]), np.array(cap["te"])
    lb.check(name, df, te)
    cut = int(len(df) * 0.9)
    uid = {x: k for k, x in enumerate(df["user"].unique())}
    iid = {x: k for k, x in enumerate(df["item"].unique())}
    U = df["user"].map(uid).to_numpy(np.int64)
    I = df["item"].map(iid).to_numpy(np.int64)
    R = df["label"].to_numpy(np.float64)
    train_ex = load_from_disk(os.path.join(HERE, "data", f"{name}-H"))["train"]["example"]
    assert len(train_ex) == len(tr)
    assert all(split_example(e)[1] == int(R[i]) for e, i in zip(train_ex, tr)), "train prompts not aligned"
    epochs = json.load(open(os.path.join(HERE, "results", f"large_baselines_{name}.json")))["info"]["BiasedMF"]["epochs"]
    torch.manual_seed(0)
    mu = float(R[:cut].mean())
    P, Q = torch.nn.Embedding(len(uid), dim), torch.nn.Embedding(len(iid), dim)
    bu, bi = torch.nn.Embedding(len(uid), 1), torch.nn.Embedding(len(iid), 1)
    for e in (P, Q):
        torch.nn.init.normal_(e.weight, std=0.01)
    for e in (bu, bi):
        torch.nn.init.zeros_(e.weight)
    mods = torch.nn.ModuleList([P, Q, bu, bi])
    opt = torch.optim.Adam(mods.parameters(), lr=1e-3, weight_decay=1e-5 if dim == 32 else 1e-4)
    Ut, It, Rt = (torch.as_tensor(x[:cut]) for x in (U, I, R.astype(np.float32)))
    f = lambda uu, ii: mu + bu(uu)[:, 0] + bi(ii)[:, 0] + (P(uu) * Q(ii)).sum(1)
    bs = 4096 if cut < 2e5 else 16384
    for ep in range(epochs):
        perm = torch.randperm(cut)
        for s in range(0, cut, bs):
            b = perm[s:s + bs]
            loss = ((f(Ut[b], It[b]) - Rt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = f(torch.as_tensor(U[te]), torch.as_tensor(I[te])).clamp(1, 5).numpy()
    print(f"{name}: BiasedMF ({epochs} epochs) test RMSE {np.sqrt(np.mean((p - R[te]) ** 2)):.4f}", flush=True)
    out = {"P": P.weight.detach().numpy(), "Q": Q.weight.detach().numpy(), "train_u": U[tr], "train_i": I[tr],
           "test_u": U[te], "test_i": I[te], "test_r": R[te], "mf_test": p}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


def insert_slot(p):
    """CF slot before the user section (MovieLens) or, when the prompt has none (Amazon-Books), before the context."""
    if "### User Information:" in p:
        return p.replace("### User Information:", SLOT, 1)
    assert "### Contextual Information:" in p
    return p.replace("### Contextual Information:", SLOT.replace("### User Information:", "### Contextual Information:"), 1)


def insert_slot_official(p):
    """Official CoLLM wording (prompts/collm_movie.txt): the user's preferences 'encoded in the feature <UserID>' and the
    target item 'with the feature <TargetItemID>', both replaced by <unk> and then by the mapped embeddings."""
    m = re.search(r"### (?:Movie|Book) Details:\n[^\n]*", p)
    assert m
    p = p[:m.end()] + " with the feature <unk>" + p[m.end():]
    return p[:m.start()] + ("### Collaborative Information:\nWe have information about the user's preferences encoded "
                            "in the feature <unk>.\n\n") + p[m.start():]


class OfficialProj(torch.nn.Module):
    """Official CoLLM llama_proj: Linear(d, d * proj_mid) - ReLU - Linear(d * proj_mid, hidden), proj_mid = 10."""
    def __init__(self, d_cf, d_model, mid=10):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_cf, d_cf * mid), torch.nn.ReLU(), torch.nn.Linear(d_cf * mid, d_model))

    def forward(self, x):
        return self.net(x)


class Mapper(torch.nn.Module):
    def __init__(self, d_cf, d_model):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_cf, 512), torch.nn.GELU(), torch.nn.Linear(512, d_model))

    def forward(self, x):
        return self.net(x)


def embed_batch(model, tok, mapper, prompts, cu, ci, device):
    """Token embeddings of the prompts with the two <unk> slots replaced by the mapped CF embeddings."""
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    emb = model.get_input_embeddings()(enc["input_ids"])
    slots = enc["input_ids"] == tok.unk_token_id
    assert int(slots.sum()) == 2 * len(prompts), "each prompt must contain exactly two CF slots"
    soft = mapper(torch.stack([cu, ci], 1).to(device)).to(emb.dtype)  # (B, 2, d)
    emb = emb.masked_scatter(slots[..., None].expand_as(emb), soft.reshape(-1, soft.shape[-1]))
    last = enc["attention_mask"].sum(1) - 1  # right padding: position of the last real token (answer marker)
    return emb, enc["attention_mask"], last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(NAMES))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--official", action="store_true", help="official CoLLM settings: MF d=256, llama_proj, wording, schedule")
    a = ap.parse_args()
    name = NAMES[a.dataset]
    cf = prepare(a.dataset, 256 if a.official else 32)
    from datasets import load_from_disk
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    data = load_from_disk(os.path.join(HERE, "data", f"{name}-H"))
    tr_p, tr_y = zip(*(split_example(e) for e in data["train"]["example"]))
    te_p, te_y = zip(*(split_example(e) for e in data["test"]["example"]))
    assert np.array_equal(np.array(te_y), cf["test_r"].astype(int))
    ins = insert_slot_official if a.official else insert_slot
    tr_p, te_p = [ins(p) for p in tr_p], [ins(p) for p in te_p]
    n_tr, n_te = (a.limit or len(tr_p)), (a.limit or len(te_p))
    tok = load_tokenizer(BASE)
    tok.padding_side = "right"  # left padding gave NaN losses (fully masked rows with gradients)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = label_token_ids(tok, te_p[0], [1, 2, 3, 4, 5])
    assert label_token_ids(tok, tr_p[0], [1, 2, 3, 4, 5]) == ids
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=quant, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, os.path.join(HERE, "adapters", f"mistral_full_{name}-H")).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dev = model.device
    P, Q = torch.as_tensor(cf["P"]), torch.as_tensor(cf["Q"])
    scale = float(model.get_input_embeddings().weight.float().std())
    if a.official:
        mapper = OfficialProj(P.shape[1], model.config.hidden_size).to(dev)
    else:
        mapper = Mapper(P.shape[1], model.config.hidden_size).to(dev)
        with torch.no_grad():  # start with soft tokens of the size of ordinary token embeddings
            mapper.net[2].weight.mul_(scale / 0.05)
    lab = torch.tensor(ids, device=dev)
    tag = "collm_official" if a.official else "collm"
    out_dir = os.path.join(HERE, "adapters", f"{tag}_mistral_{name}-H" + ("_smoke" if a.limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "mapper.pt")
    if os.path.exists(ckpt):
        mapper.load_state_dict(torch.load(ckpt))
        print("loaded", ckpt, flush=True)
    else:
        steps = a.epochs * math.ceil(n_tr / (a.batch * a.accum))
        if a.official:  # linear_warmup_cosine_lr: warmup 1e-5 -> 1e-3 over 200 steps, cosine to 8e-5, wd 1e-3
            opt = torch.optim.AdamW(mapper.parameters(), lr=1e-3, weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1e-5 + (1e-3 - 1e-5) * s / 200) / 1e-3 if s < 200 else
                                                      (8e-5 + 0.5 * (1e-3 - 8e-5) * (1 + math.cos(math.pi * (s - 200) / max(1, steps - 200)))) / 1e-3)
        else:
            opt = torch.optim.AdamW(mapper.parameters(), lr=a.lr, weight_decay=0.0)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * steps))) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
        tu, ti = torch.as_tensor(cf["train_u"]), torch.as_tensor(cf["train_i"])
        g = torch.Generator().manual_seed(0)
        step, t0, run = 0, time.time(), []
        model.train()
        for ep in range(a.epochs):
            perm = torch.randperm(n_tr, generator=g).tolist()
            for k in range(0, n_tr, a.batch):
                b = perm[k:k + a.batch]
                emb, att, last = embed_batch(model, tok, mapper, [tr_p[j] for j in b], P[tu[b]], Q[ti[b]], dev)
                logits = model(inputs_embeds=emb, attention_mask=att).logits[torch.arange(len(b), device=dev), last].float()
                y = lab[torch.tensor([tr_y[j] - 1 for j in b], device=dev)]
                loss = torch.nn.functional.cross_entropy(logits, y) / a.accum
                if not torch.isfinite(loss):
                    print("non-finite loss, batch skipped", flush=True)
                    continue
                loss.backward()
                run.append(float(loss.detach()) * a.accum)
                if (k // a.batch + 1) % a.accum == 0:
                    if not all(torch.isfinite(g_.grad).all() for g_ in mapper.parameters() if g_.grad is not None):
                        print("non-finite gradient, step skipped", flush=True)
                        opt.zero_grad()
                        continue
                    torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    step += 1
                    if step % 50 == 0:
                        print(f"epoch {ep + 1} step {step}/{steps} loss {np.mean(run[-100:]):.4f} "
                              f"({time.time() - t0:.0f}s)", flush=True)
            torch.save(mapper.state_dict(), ckpt + f".ep{ep + 1}")
        torch.save(mapper.state_dict(), ckpt)
    model.eval()
    mapper.eval()
    probs = []
    eu, ei = torch.as_tensor(cf["test_u"]), torch.as_tensor(cf["test_i"])
    with torch.no_grad():
        for k in range(0, n_te, 16):
            b = list(range(k, min(k + 16, n_te)))
            emb, att, last = embed_batch(model, tok, mapper, [te_p[j] for j in b], P[eu[b]], Q[ei[b]], dev)
            logits = model(inputs_embeds=emb, attention_mask=att).logits[torch.arange(len(b), device=dev), last].float()
            probs += torch.softmax(logits[:, lab], -1).cpu().tolist()
    truth = list(te_y[:n_te])
    expected = [sum((r + 1) * p for r, p in enumerate(row)) for row in probs]
    mae, rmse = metrics(truth, expected)
    out = os.path.join(HERE, "probs", f"{tag}_mistral_{name}-H{'_limit' if a.limit else ''}.json")
    json.dump({"data": f"data/{name}-H", "method": "CoLLM-style input-level fusion (stage 2 on our CPR-LLM adapter)",
               "epochs": a.epochs, "truth": truth, "probs": probs, "expected": expected, "n": len(truth),
               "mae": mae, "rmse": rmse}, open(out, "w"))
    print(f"{name} CoLLM-style: n {len(truth)}  MAE {mae:.4f}  RMSE {rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
