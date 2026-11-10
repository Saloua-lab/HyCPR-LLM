"""Head-to-head with a domain-pretrained LLM recommender (review 6: "a head-to-head against a domain-pretrained LLM
could test whether continual pretraining subsumes some of the hybrid gains").

RecGPT-7B-Instruct (Ngo & Nguyen, ACL 2024; MPT-7B continually pretrained on 20.5B tokens of recommendation data,
then instruction-tuned) is fine-tuned with exactly the recipe of the other LLMs (train.py: 4-bit NF4, LoRA r = 32 on
all attention and MLP projections, 10 / 2 epochs, jobs/recgpt.json) on the same prompts, read out by the expected
rating over the five rating tokens, and fused with the same CF pool by the same rule (CF model and w on the grid chosen
on validation RMSE, fusion_robust.select). Reported per -H setting: RecGPT alone, RecGPT + fusion, and the paired
bootstrap differences with CPR-LLM (Mistral, the LLM inside HyCPR-LLM) and HyCPR-LLM (Mistral).

Usage: python recgpt_compare.py → results/recgpt.{json,md}, probs/hybrid/recgpt_hycpr_<setting>.json
"""
import json

import numpy as np

from fusion_robust import jobs, select
from holm import test


def main():
    out, md = {}, ["| Setting | RecGPT alone | CPR-LLM (Mistral) | RecGPT - CPR-LLM (MAE, RMSE) | RecGPT + fusion (b*, w) | HyCPR-LLM (Mistral) | RecGPT fusion - HyCPR (MAE, RMSE) |",
                   "|---|---|---|---|---|---|---|"]
    for m, s, names, yv, yt, Lv, Lt, Bv, Bt, hp in jobs():
        if m != "mistral":
            continue
        if s in ("MovieLens-H", "AmazonBooks-H"):
            rv, rt = json.load(open(f"probs/recgpt_full_{s}_val.json")), json.load(open(f"probs/recgpt_full_{s}.json"))
        else:
            rv, rt = (json.load(open(f"probs/variants/recgpt_v0_{s}_{sp}.json")) for sp in ("validation", "test"))
        assert np.array_equal(np.array(rv["truth"], float), yv) and np.array_equal(np.array(rt["truth"], float), yt), s
        Rv, Rt = np.array(rv["expected"], float), np.array(rt["expected"], float)
        j, w = select(Rv, Bv, yv)
        fused = w * Rt + (1 - w) * Bt[j]
        hy = np.array(json.load(open(hp))["expected"], float)
        f = lambda x: [float(np.mean(np.abs(x - yt))), float(np.sqrt(np.mean((x - yt) ** 2)))]
        rec = {"recgpt": f(Rt), "cpr_mistral": f(Lt), "recgpt_fused": f(fused), "b": names[j], "w": float(w), "hycpr": f(hy),
               "recgpt_vs_cpr": {k: test(Rt, Lt, yt, k) for k in ("mae", "rmse")},
               "fused_vs_hycpr": {k: test(fused, hy, yt, k) for k in ("mae", "rmse")},
               "fused_vs_recgpt": {k: test(fused, Rt, yt, k) for k in ("mae", "rmse")}}
        out[s] = rec
        json.dump({"truth": yt.tolist(), "expected": fused.tolist(), "baseline": names[j], "w": float(w)},
                  open(f"probs/hybrid/recgpt_hycpr_{s}.json", "w"))
        g = lambda x: f"{x[0]:.3f} / {x[1]:.3f}"
        sg = lambda d: f"{d['diff']:+.3f}{'*' if d['p'] < 0.05 else ''}"
        md.append(f"| {s} | {g(rec['recgpt'])} | {g(rec['cpr_mistral'])} | {sg(rec['recgpt_vs_cpr']['mae'])}, {sg(rec['recgpt_vs_cpr']['rmse'])} | "
                  f"{g(rec['recgpt_fused'])} ({names[j]}, {w:.2f}) | {g(rec['hycpr'])} | {sg(rec['fused_vs_hycpr']['mae'])}, {sg(rec['fused_vs_hycpr']['rmse'])} |")
        print(md[-1], flush=True)
    json.dump(out, open("results/recgpt.json", "w"), indent=1)
    open("results/recgpt.md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
