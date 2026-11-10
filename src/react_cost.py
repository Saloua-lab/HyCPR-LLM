"""Cost per test interaction of the ReAct configurations (review 3, point 2): LLM calls, prompt and generated
tokens, wall-clock seconds (Mistral-7B-Instruct 4-bit, one H100, batch as recorded), from the timed runs.

Usage: python react_cost.py   → results/react_cost.{json,md}
"""
import json
import os

DS = ["LDOS", "Depaul", "Tijuana"]
CF = [("direct", "direct_timed"), ("direct_tools", "direct_tools"), ("react_notools", "react_notools_timed"),
      ("react_all", "react_all_timed")]
out, md = {}, ["| config | dataset | LLM calls | prompt tokens | generated tokens | seconds | batch |", "|---|---|---|---|---|---|---|"]
for c, f in CF:
    for ds in DS:
        p = f"results/react/{ds}_{f}.json"
        if not os.path.exists(p):
            continue
        k = json.load(open(p))["cost"]
        n = k["n"]
        r = {x: k[x] / n for x in ("llm_calls", "prompt_tokens", "new_tokens", "seconds")}
        r.update(batch=k["batch"], gpu=k.get("gpu"))
        out.setdefault(c, {})[ds] = r
        md.append(f"| {c} | {ds} | {r['llm_calls']:.2f} | {r['prompt_tokens']:.0f} | {r['new_tokens']:.0f} | {r['seconds']:.3f} | {k['batch']} |")
json.dump(out, open("results/react_cost.json", "w"), indent=1)
open("results/react_cost.md", "w").write("\n".join(md))
print("\n".join(md))
