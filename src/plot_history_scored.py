"""Figures/<Dataset>_history_scored.pdf of the paper (fig:history-impact): MAE and RMSE of the leak-free Mistral -H
model against the number k of history items kept in the test prompt, from probs/history/mistral_clean_<setting>_k<k>.json.

Usage: python plot_history_scored.py   → figures/{LDOS,Restaurant,Depaul}_history_scored.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
for name, setting in [("LDOS", "LDOS-H"), ("Restaurant", "Restaurant-H"), ("Depaul", "Depaul-H")]:
    ks = list(range(6))
    d = [json.load(open(f"probs/history/mistral_clean_{setting}_k{k}.json")) for k in ks]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(ks, [x["mae"] for x in d], "o-", label="MAE")
    ax.plot(ks, [x["rmse"] for x in d], "s--", label="RMSE")
    ax.set_xlabel("history items in the prompt ($k$)")
    ax.set_ylabel("error")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"figures/{name}_history_scored.pdf")
    fig.savefig(f"figures/{name}_history_scored.png", dpi=150)
    plt.close(fig)
print("written figures/*_history_scored.{pdf,png}")
