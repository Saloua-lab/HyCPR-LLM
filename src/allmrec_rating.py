"""A-LLMRec (Kim et al., KDD 2024; official code https://github.com/ghdtjr/A-LLMRec, cloned in ../external/A-LLMRec)
adapted to rating prediction on MovieLens-1M and Amazon-Books (review 4).

Taken from the official code:
  - the collaborative model: pre_train/sasrec/model.SASRec (imported), with its pre-training recipe (hidden 50,
    max length 50, 2 blocks, 1 head, dropout 0.5, Adam lr 1e-3 betas (0.9, 0.98), batch 128, BCE with one negative
    per position, users sampled with their last 50 items);
  - stage 1 (pre_train_phase1, re-implemented line by line): the item encoder two_layer_mlp(hidden) (imported) on the
    SASRec item embeddings and two_layer_mlp(768) on SBERT (nq-distilbert-base-v1, mean pooling, fine-tuned) text
    embeddings, loss = BCE recommendation loss with the user representation + matching MSE + 0.5 item reconstruction
    + 0.2 text reconstruction, on the last position of each user sequence, chunks of 60, Adam lr 1e-4, batch 32;
  - stage 2 (pre_train_phase2): the LLM is frozen; the user representation (SASRec log_only, last position) is
    projected by log_emb_proj (Linear-LayerNorm-LeakyReLU-Linear) and prepended to the prompt "... is a user
    representation."; the 128-d item codes (frozen stage-1 encoder) are projected by item_emb_proj (Linear-LayerNorm-
    GELU-Linear) into the [HistoryEmb] / [CandidateEmb] tokens that follow the item titles; xavier initialisation;
    LM loss on the answer + eos; Adam lr 1e-4 betas (0.9, 0.98); only the two projectors are trained.
Adapted: the answer is the rating (the target item takes the [CandidateEmb] token, there is no candidate list); the
prompt is our Alpaca prompt (same as CPR-LLM) after the user-representation sentence; the LLM is Mistral-7B-Instruct
(bf16, frozen) instead of OPT-6.7B; the item text is the description shown in our prompts (title and genre / brand);
SASRec is trained on the training period of our chronological split (200 epochs on MovieLens-1M; 3 on Amazon-Books,
whose 1.86M users make 200 epochs impractical); stage 1 uses 10 epochs on MovieLens-1M and 2 epochs on 200,000 sampled
users of Amazon-Books; stage 2 uses 2 epochs over the 50,000 training prompts with batches of 8 (budget of our
CPR-LLM models). The rating is read as for every other model (expected rating over the tokens "1".."5").

Usage: HF_HOME=../.hf_cache python allmrec_rating.py --dataset movielens [--limit N]
Output: cf/<Name>_allmrec.npz, adapters/allmrec_mistral_<Name>-H/, probs/allmrec_mistral_<Name>-H.json
"""
import argparse
import json
import os
import random
import re
import sys
import types
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ALLM = os.path.join(HERE, "..", "external", "A-LLMRec")
sys.path.insert(0, HERE)
sys.path.insert(0, ALLM)
sys.modules.setdefault("sentence_transformers", types.SimpleNamespace(SentenceTransformer=None))  # SBERT done below
from evaluate_probs import label_token_ids, metrics, split_example  # noqa: E402
from models.a_llmrec_model import two_layer_mlp  # noqa: E402  (official)
from pre_train.sasrec.model import SASRec  # noqa: E402  (official)

NAMES = {"movielens": "MovieLens", "amazon": "AmazonBooks"}
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
SBERT = "sentence-transformers/nq-distilbert-base-v1"
MAXLEN = 50


def sasrec_args(dev):
    return SimpleNamespace(hidden_units=50, maxlen=MAXLEN, num_blocks=2, num_heads=1, dropout_rate=0.5, device=dev)


def pad_left(seq, n=MAXLEN):
    out = np.zeros(n, np.int64)
    seq = seq[-n:]
    if len(seq):
        out[-len(seq):] = seq
    return out


def prepare(dataset, dev):
    """Training-period user sequences (official ids: item + 1, 0 = padding), the prior sequence of every prompt, the
    official SASRec, and the item texts."""
    name = NAMES[dataset]
    path = os.path.join(HERE, "cf", f"{name}_allmrec.npz")
    sas_path = os.path.join(HERE, "cf", f"{name}_allmrec_sasrec.pt")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    import large_baselines as lb
    cap = lb.rebuild(dataset)
    df, tr, te = cap["df"], list(cap["tr"]), list(cap["te"])
    cut = int(len(df) * 0.9)
    iid = {x: k for k, x in enumerate(df["item"].unique())}
    n_items = len(iid)
    I = df["item"].map(iid).to_numpy(np.int64) + 1
    by_user = defaultdict(list)
    for j, (u, t) in enumerate(zip(df["user"], df["ts"])):
        by_user[u].append((t, j))
    user_seqs = [np.array([I[j] for tj, j in v if j < cut]) for v in by_user.values()]
    user_seqs = [s for s in user_seqs if len(s) >= 2]
    out = {"n_items": np.array(n_items)}
    for split, idx in (("train", tr), ("test", te)):
        P = np.zeros((len(idx), MAXLEN), np.int64)
        for r, i in enumerate(idx):
            u, t = df.at[i, "user"], df.at[i, "ts"]
            P[r] = pad_left([I[j] for tj, j in by_user[u] if tj < t and j < cut])
        out[f"{split}_prior"] = P
    # official SASRec pre-training
    torch.manual_seed(0)
    model = SASRec(len(user_seqs), n_items, sasrec_args(dev)).to(dev)
    for p in model.parameters():  # official init (xavier_normal_, errors ignored)
        try:
            torch.nn.init.xavier_normal_(p.data)
        except ValueError:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    bce = torch.nn.BCEWithLogitsLoss()
    epochs = 200 if dataset == "movielens" else 3
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(len(user_seqs))
        model.train()
        tot, nb = 0.0, 0
        for b in range(0, len(order), 128):
            seqs = [user_seqs[k] for k in order[b:b + 128]]
            seq = np.stack([pad_left(s[:-1]) for s in seqs])
            pos = np.stack([pad_left(s[1:]) for s in seqs])
            neg = np.where(pos > 0, rng.integers(1, n_items + 1, pos.shape), 0)
            pl, nl = model(None, seq, pos, neg)
            ix = np.where(pos != 0)
            loss = bce(pl[ix], torch.ones_like(pl[ix])) + bce(nl[ix], torch.zeros_like(nl[ix]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + float(loss), nb + 1
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"SASRec epoch {ep + 1}/{epochs}: loss {tot / nb:.4f}", flush=True)
    torch.save(model.state_dict(), sas_path)
    # item texts (the description shown in our prompts)
    first = df.drop_duplicates("item").set_index("item")
    inv = {v: k for k, v in iid.items()}
    texts = []
    for n in range(n_items):
        r = first.loc[inv[n]].copy()
        r["item"] = inv[n]
        texts.append(cap["render"](r).split(". The user rated it as:")[0].replace("Name: ", "", 1))
    out["texts"] = np.array(texts, dtype=object)
    out["n_users"] = np.array(len(user_seqs))
    np.save(os.path.join(HERE, "cf", f"{name}_allmrec_userseqs.npy"), np.array(user_seqs, dtype=object), allow_pickle=True)
    np.savez(path, **out)
    return dict(np.load(path, allow_pickle=True))


def load_sasrec(name, n_users, n_items, dev):
    m = SASRec(int(n_users), int(n_items), sasrec_args(dev)).to(dev)
    m.load_state_dict(torch.load(os.path.join(HERE, "cf", f"{name}_allmrec_sasrec.pt")))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def stage1(dataset, name, cf, dev):
    """Official pre_train_phase1: align SASRec item embeddings and SBERT text embeddings."""
    path = os.path.join(HERE, "cf", f"{name}_allmrec_stage1.pt")
    if os.path.exists(path):
        return torch.load(path)
    from transformers import AutoModel, AutoTokenizer
    rec = load_sasrec(name, cf["n_users"], cf["n_items"], dev)
    user_seqs = list(np.load(os.path.join(HERE, "cf", f"{name}_allmrec_userseqs.npy"), allow_pickle=True))
    texts = cf["texts"]
    stok = AutoTokenizer.from_pretrained(SBERT)
    sbert = AutoModel.from_pretrained(SBERT).to(dev).train()
    mlp, mlp2 = two_layer_mlp(50).to(dev), two_layer_mlp(768).to(dev)
    params = list(sbert.parameters()) + list(mlp.parameters()) + list(mlp2.parameters())
    opt = torch.optim.Adam(params, lr=1e-4, betas=(0.9, 0.98))
    bce, mse = torch.nn.BCEWithLogitsLoss(), torch.nn.MSELoss()

    def embed(item_ids):  # official text: '"title, description"'
        t = stok([f'"{texts[i - 1]}"' for i in item_ids], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        h = sbert(**t).last_hidden_state
        a = t["attention_mask"][..., None].float()
        return (h * a).sum(1) / a.sum(1)

    rng = np.random.default_rng(1)
    users = np.arange(len(user_seqs))
    epochs = 10
    if dataset == "amazon":
        users, epochs = rng.choice(users, 200_000, replace=False), 2
    for ep in range(epochs):
        order = rng.permutation(users)
        tot, nb = 0.0, 0
        for b in range(0, len(order), 32):
            seqs = [user_seqs[k] for k in order[b:b + 32]]
            seq = np.stack([pad_left(s[:-1]) for s in seqs])
            pos = np.stack([pad_left(s[1:]) for s in seqs])
            neg = np.where(pos > 0, rng.integers(1, int(cf["n_items"]) + 1, pos.shape), 0)
            with torch.no_grad():
                log_emb, pos_emb, neg_emb = rec(None, seq, pos, neg, mode="item")
            ix = [MAXLEN * (k + 1) - 1 for k in range(len(seqs))]
            log_emb, pos_emb, neg_emb = log_emb[ix], pos_emb[ix], neg_emb[ix]
            p_ids, n_ids = pos[:, -1], neg[:, -1]
            for s in range(0, len(ix), 60):  # official chunks of 60
                opt.zero_grad()
                pt, nt = embed(p_ids[s:s + 60]), embed(n_ids[s:s + 60])
                pm, pp = mlp(pos_emb[s:s + 60])
                nm, np_ = mlp(neg_emb[s:s + 60])
                pmt, ppt = mlp2(pt)
                nmt, npt = mlp2(nt)
                le = log_emb[s:s + 60]
                pl, nl = (le * pp).mean(1), (le * np_).mean(1)
                loss = bce(pl, torch.ones_like(pl)) + bce(nl, torch.zeros_like(nl))
                loss = (loss + mse(pm, pmt) + mse(nm, nmt) + 0.5 * (mse(pp, pos_emb[s:s + 60]) + mse(np_, neg_emb[s:s + 60]))
                        + 0.2 * (mse(ppt, pt.data) + mse(npt, nt.data)))
                loss.backward()
                opt.step()
                tot, nb = tot + float(loss), nb + 1
        print(f"stage 1 epoch {ep + 1}/{epochs}: loss {tot / nb:.4f}", flush=True)
    state = {k: v.cpu() for k, v in mlp.state_dict().items()}
    torch.save(state, path)
    return state


def hybrid(prompt):
    p = re.sub(r"(### (?:Movie|Book) Details:\n[^\n]*)", lambda m: m.group(1) + " [CandidateEmb]", prompt, count=1)
    return " is a user representation. " + p.replace(". The user rated it as:", " [HistoryEmb]. The user rated it as:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(NAMES))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    dev = "cuda"
    name = NAMES[a.dataset]
    cf = prepare(a.dataset, dev)
    mlp_state = stage1(a.dataset, name, cf, dev)
    llara = dict(np.load(os.path.join(HERE, "cf", f"{name}_llara.npz")))  # history / target item ids, checked
    rec = load_sasrec(name, cf["n_users"], cf["n_items"], dev)
    mlp = two_layer_mlp(50).to(dev)
    mlp.load_state_dict(mlp_state)
    for p in mlp.parameters():
        p.requires_grad_(False)
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM
    from tokenization import load_tokenizer
    data = load_from_disk(os.path.join(HERE, "data", f"{name}-H"))
    tr_p, tr_y = zip(*(split_example(e) for e in data["train"]["example"]))
    te_p, te_y = zip(*(split_example(e) for e in data["test"]["example"]))
    n_tr, n_te = a.limit or len(tr_p), a.limit or len(te_p)
    tok = load_tokenizer(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": ["[UserRep]", "[HistoryEmb]", "[CandidateEmb]"]})
    his_id, can_id = tok.convert_tokens_to_ids("[HistoryEmb]"), tok.convert_tokens_to_ids("[CandidateEmb]")
    lab = torch.tensor(label_token_ids(tok, hybrid(te_p[0]), [1, 2, 3, 4, 5]), device=dev)
    llm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    llm.resize_token_embeddings(len(tok))
    for p in llm.parameters():
        p.requires_grad_(False)
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    H = llm.config.hidden_size
    log_emb_proj = torch.nn.Sequential(torch.nn.Linear(50, H), torch.nn.LayerNorm(H), torch.nn.LeakyReLU(), torch.nn.Linear(H, H))
    item_emb_proj = torch.nn.Sequential(torch.nn.Linear(128, H), torch.nn.LayerNorm(H), torch.nn.GELU(), torch.nn.Linear(H, H))
    for m in (log_emb_proj, item_emb_proj):
        torch.nn.init.xavier_normal_(m[0].weight)
        torch.nn.init.xavier_normal_(m[3].weight)
    log_emb_proj, item_emb_proj = log_emb_proj.to(dev), item_emb_proj.to(dev)

    def item_codes(ids):  # official get_item_emb: frozen SASRec item_emb -> stage-1 encoder (128-d)
        with torch.no_grad():
            return mlp(rec.item_emb(torch.as_tensor(ids + 1, device=dev)))[0]

    def batch_embeds(prompts, answers, prior, hist, target):
        texts = prompts if answers is None else [p + str(y) + tok.eos_token for p, y in zip(prompts, answers)]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
        emb = llm.get_input_embeddings()(enc["input_ids"]).clone()
        ids = enc["input_ids"]
        for r in range(len(ids)):
            h = hist[r][hist[r] >= 0]
            pos = (ids[r] == his_id).nonzero().view(-1)
            assert len(pos) == len(h)
            if len(pos):
                emb[r, pos] = item_emb_proj(item_codes(h)).to(emb.dtype)
            pos = (ids[r] == can_id).nonzero().view(-1)
            emb[r, pos] = item_emb_proj(item_codes(np.array([target[r]]))).to(emb.dtype)
        with torch.no_grad():
            u = rec(None, prior, prior, prior, mode="log_only")
        u = log_emb_proj(u).to(emb.dtype)[:, None]
        emb = torch.cat([u, emb], 1)
        att = torch.cat([torch.ones_like(enc["attention_mask"][:, :1]), enc["attention_mask"]], 1)
        return emb, att, enc

    out_dir = os.path.join(HERE, "adapters", f"allmrec_mistral_{name}-H" + ("_smoke" if a.limit else ""))
    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.join(out_dir, "proj.pt")
    if os.path.exists(ck):
        st = torch.load(ck)
        log_emb_proj.load_state_dict(st["log"])
        item_emb_proj.load_state_dict(st["item"])
        print("loaded", ck, flush=True)
    else:
        opt = torch.optim.Adam(list(log_emb_proj.parameters()) + list(item_emb_proj.parameters()), lr=1e-4, betas=(0.9, 0.98))
        random.seed(0)
        step, run = 0, []
        steps = a.epochs * (n_tr // a.batch)
        for ep in range(a.epochs):
            order = list(range(n_tr))
            random.shuffle(order)
            for k in range(0, n_tr - a.batch + 1, a.batch):
                b = order[k:k + a.batch]
                prompts = [hybrid(tr_p[j]) for j in b]
                emb, att, enc = batch_embeds(prompts, [tr_y[j] for j in b], cf["train_prior"][b], llara["train_hist"][b], llara["train_target"][b])
                labels = torch.full(att.shape, -100, device=dev)
                for r, p in enumerate(prompts):
                    n = len(tok(p)["input_ids"])
                    L = int(enc["attention_mask"][r].sum())
                    labels[r, 1 + n:1 + L] = enc["input_ids"][r, n:L]  # answer + eos (+1: user representation)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = llm(inputs_embeds=emb, attention_mask=att, labels=labels).loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                run.append(float(loss.detach()))
                if step % 100 == 0:
                    print(f"epoch {ep + 1} step {step}/{steps} loss {np.mean(run[-100:]):.4f}", flush=True)
        torch.save({"log": log_emb_proj.state_dict(), "item": item_emb_proj.state_dict()}, ck)
    probs = []
    with torch.no_grad():
        for k in range(0, n_te, 16):
            b = list(range(k, min(k + 16, n_te)))
            emb, att, _ = batch_embeds([hybrid(te_p[j]) for j in b], None, cf["test_prior"][b], llara["test_hist"][b], llara["test_target"][b])
            last = att.sum(1) - 1
            logits = llm(inputs_embeds=emb, attention_mask=att).logits[torch.arange(len(b), device=dev), last].float()
            probs += torch.softmax(logits[:, lab], -1).cpu().tolist()
    truth = list(te_y[:n_te])
    expected = [sum((r + 1) * p for r, p in enumerate(row)) for row in probs]
    mae, rmse = metrics(truth, expected)
    out = os.path.join(HERE, "probs", f"allmrec_mistral_{name}-H{'_limit' if a.limit else ''}.json")
    json.dump({"data": f"data/{name}-H", "method": "A-LLMRec (official modules) adapted to rating prediction",
               "epochs": a.epochs, "truth": truth, "probs": probs, "expected": expected, "n": len(truth),
               "mae": mae, "rmse": rmse}, open(out, "w"))
    print(f"{name} A-LLMRec: n {len(truth)}  MAE {mae:.4f}  RMSE {rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
