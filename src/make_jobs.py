"""Write the job lists for run_queue.py (one JSON file per phase under jobs/).

Usage: python make_jobs.py
Each job trains one adapter with train.py and evaluates it with evaluate.py; "-H" adapters are evaluated on the
original test set and on the leak-free copy in data/<name>_noleak (rebuild_history.py).
"""
import json
import os

PY = "python"
FE = "../small_data/"

INSTRUCT = {"mistral": "mistralai/Mistral-7B-Instruct-v0.2", "llama": "meta-llama/Llama-2-7b-chat-hf",
            "eurollm": "utter-project/EuroLLM-9B-Instruct", "deepseek": "deepseek-ai/deepseek-llm-7b-chat"}
BASE = {"mistral": "mistralai/Mistral-7B-v0.1", "llama": "meta-llama/Llama-2-7b-hf",
        "eurollm": "utter-project/EuroLLM-9B", "deepseek": "deepseek-ai/deepseek-llm-7b-base"}

# Test sets of the small datasets. Mistral and Llama have their own -H prompt files (the ones behind their
# Table 5 cells); EuroLLM and DeepSeek use the generic ones (the EuroLLM re-runs in report.md).
SMALL = {
    "LDOS-WH": "LocationLDOSWithoutHis", "Depaul-WH": "Datasets/DepaulNoHistory",
    "Tijuana-WH": "LocationRestaurantSansHis", "LDOS-H": "LocationLDOSWithH",
    "Depaul-H": "LocationDepaulAvecHis", "Restaurant-H": "LocationRestaurantAvecHis",
}
OWN_H = {"mistral": {"LDOS-H": "LocationLDOSWithHisAvecMistral", "Depaul-H": "LocationDepaulWithHisAvecMistral",
                     "Restaurant-H": "LocationRestaurantWithHisAvecMistral"},
         "llama": {"LDOS-H": "Datasets/LDOSWithHistoryLlama", "Depaul-H": "LocationDepaulWithHisAvecLLama",
                   "Restaurant-H": "LocationRestaurantWithHisAvecLLama"}}


def dataset(model, setting):
    return OWN_H.get(model, {}).get(setting, SMALL[setting])


def job(name, data, base, noleak=None, train_args="", eval_args=""):
    adapter = f"adapters/{name}"
    path = data if data.startswith("data/") else FE + data
    cmds = [f"{PY} train.py --data {path} --base {base} --out {adapter} {train_args}".strip(),
            f"{PY} evaluate.py --data {path} --adapter {adapter} --base {base} --out results/{name}.json "
            f"{eval_args}".strip()]
    done = f"results/{name}.json"
    if noleak:
        cmds.insert(1, f"{PY} evaluate.py --data data/{noleak}_noleak --adapter {adapter} --base {base} "
                       f"--out results/{name}_noleak.json")
    return {"name": name, "cmd": " && ".join(cmds), "done": done}


def small_jobs(model, settings, base_kind):
    base = (INSTRUCT if base_kind == "full" else BASE)[model]
    out = []
    for s in settings:
        d = dataset(model, s)
        noleak = os.path.basename(d) if s.endswith("-H") else None
        out.append(job(f"{model}_{base_kind}_{s}", d, base, noleak))
    return out


def large_jobs(models, datasets, base_kind):
    """Large datasets built by prepare_large.py: 50,000 training prompts, 2 epochs instead of 10."""
    out = []
    for m in models:
        base = (INSTRUCT if base_kind == "full" else BASE)[m]
        for d in datasets:
            labels = "--labels 0 1" if d.startswith("MIND") else ""
            out.append(job(f"{m}_{base_kind}_{d}", f"data/{d}", base, train_args="--epochs 2", eval_args=labels))
    return out


def clean_jobs(model):
    """-H adapters retrained on leak-free histories (train and test split of data/<name>_noleak)."""
    return [job(f"{model}_clean_{s}", f"data/{os.path.basename(dataset(model, s))}_noleak", INSTRUCT[model])
            for s in H]


def eval_job(name, data, adapter, base):
    """Evaluation of an existing FirstEval adapter, no training."""
    return {"name": name, "done": f"results/{name}.json",
            "cmd": f"{PY} evaluate.py --data {FE}{data} --adapter {FE}{adapter} --base {base} --out results/{name}.json"}


H = ["LDOS-H", "Depaul-H", "Restaurant-H"]
PHASES = {
    # Table 5 Mistral Tijuana-WH: adapter and test set exist, only a running log value behind the paper number
    "phase5_clean": [eval_job("mistral_full_Tijuana-WH", "LocationTRestaurantWithoutHis",
                              "Restaurant-recommendation-model1WithoutHist", INSTRUCT["mistral"])]
                    + clean_jobs("mistral") + clean_jobs("deepseek"),
    "phase6_clean_gated": clean_jobs("llama") + clean_jobs("eurollm"),
    # no token needed
    "phase1_small": small_jobs("deepseek", list(SMALL), "full")
                    + small_jobs("mistral", H, "noit") + small_jobs("deepseek", H, "noit"),
    "phase3_large": large_jobs(["mistral", "deepseek"], ["YELP-H", "YELP-WH"], "full")
                    + large_jobs(["mistral", "deepseek"], ["YELP-H"], "noit")
                    + large_jobs(["mistral"], ["MovieLens-H"], "full"),
    "phase4_amazon": large_jobs(["mistral"], ["AmazonBooks-H"], "full"),
    # gated base models (HF token)
    "phase2_small_gated": small_jobs("llama", H, "noit") + small_jobs("eurollm", H, "noit"),
}

if __name__ == "__main__":
    os.makedirs("jobs", exist_ok=True)
    for name, jobs in PHASES.items():
        json.dump(jobs, open(f"jobs/{name}.json", "w"), indent=1)
        print(name, len(jobs), "jobs")
