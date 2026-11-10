"""Recompute the CPR-LLM paper's metrics from the saved prediction logs.

Usage: python3 recompute.py ../small_data  > report.md

Reads every results_*.txt under the given folder (via parse_runs.parse), then:
  1. final MAE/RMSE for each run that backs a cell of Table 5,
  2. the earliest point in each run where the running MAE/RMSE equals the
     value printed in the paper (i.e. whether the paper reports an early stop),
  3. conformal prediction on the real absolute errors |r - r_hat|:
     split CP (30% calibration, as in the notebook, 1000 random splits) and
     sliding-window CP (window of the 100 most recent scores, in test order).

Validity rule copied from NewEvaluation18Nov_Updated.ipynb cell 44: a pair is
kept only if the prediction parses and lies in 1..5.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_runs import parse

EPSILONS = (0.1, 0.05, 0.01)
CAL_RATIO = 0.3
N_SPLITS = 1000
WINDOW = 100

# Paper cell -> (results file, paper MAE, paper RMSE).
# The file is the complete run for that model/dataset/history setting; when a
# setting was run more than once, the latest complete run is used.
TABLE5 = [
    ("Mistral", "LDOS-WH", "results_2024-11-13_06-37-44_mistralai_Mistral-7B-Instruct-v0.2_1.txt", 0.848, 1.197),
    ("Mistral", "Depaul-WH", "results_2024-12-09_15-29-24_Datasets_DepaulTrainAndTestNoHistory_H.txt", 1.595, 2.144),
    ("Mistral", "LDOS-H", "results_2025-01-21_17-31-06, mistralai_Mistral-7B-Instruct-v0.2, LocationLDOSWithHisAvecMistral.txt", 0.768, 1.155),
    ("Mistral", "Depaul-H", "results_2024-12-24_14-31-23, mistralai_Mistral-7B-Instruct-v0.2, LocationDepaulWithHisAvecMistral.txt", 0.666, 1.122),
    ("Mistral", "Restaurant-H", "results_2024-12-24_13-15-33, mistralai_Mistral-7B-Instruct-v0.2, LocationRestaurantWithHisAvecMistral.txt", 0.589, 1.249),
    ("EuroLLM-9B", "LDOS-WH", "results_2024-12-17_18-22-13_Datasets_LDOSTrainAndTest_t.txt", 0.905, 1.197),
    ("EuroLLM-9B", "Depaul-WH", "results_2024-12-18_09-51-32_Datasets_DepaulNoHistory_t.txt", 1.427, 1.872),
    ("EuroLLM-9B", "Tijuana-WH", "results_2024-12-21_12-06-18_LocationRestaurantSansHis_t.txt", 1.204, 1.790),
    ("EuroLLM-9B", "Depaul-H", "results_2024-12-21_08-50-25_LocationDepaulAvecHis_t.txt", 0.800, 1.290),
    ("EuroLLM-9B", "Restaurant-H", "results_2024-12-21_14-54-49_LocationRestaurantAvecHis_t.txt", 0.950, 1.602),
    ("Llama-2-7B", "LDOS-WH", "results_2024-12-22_19-16-49, meta-llama_Llama-2-7b-chat-hf, LocationLDOSWithoutHis.txt", 0.967, 1.440),
    ("Llama-2-7B", "Depaul-WH", "results_2025-01-13_05-58-47, meta-llama_Llama-2-7b-chat-hf, LocationDepaulWithoutHisAvecLLama.txt", 1.175, 1.673),
    ("Llama-2-7B", "Tijuana-WH", "results_2024-12-22_21-44-08, meta-llama_Llama-2-7b-chat-hf, LocationRestaurantWithoutHisAvecLLama.txt", 0.923, 1.575),
    ("Llama-2-7B", "LDOS-H", "results_2024-12-22_21-17-25, meta-llama_Llama-2-7b-chat-hf, Datasets_LDOSWithHistoryLlama.txt", 0.500, 0.707),
    ("Llama-2-7B", "Depaul-H", "results_2025-01-22_13-55-41, meta-llama_Llama-2-7b-chat-hf, LocationDepaulWithHisAvecLLama.txt", 0.714, 1.253),
    ("Llama-2-7B", "Restaurant-H", "results_2025-01-22_15-07-31, meta-llama_Llama-2-7b-chat-hf, LocationRestaurantWithHisAvecLLama.txt", 0.818, 1.480),
    ("DeepSeek-coder-33B", "LDOS-WH", "results_2025-01-25_15-51-29, deepseek-ai_deepseek-coder-33b-instruct, LocationLDOSNOHisAvecDeepS.txt", 0.890, 1.310),
]

# Settings used for Table 7 (the "-H" datasets with Mistral and Llama).
TABLE7 = [(m, d, f) for m, d, f, _, _ in TABLE5 if d.endswith("-H") and m in ("Mistral", "Llama-2-7B")]


def valid_pairs(run):
    return [(t, p) for t, p in run if t is not None and p is not None and 1 <= t <= 5 and 1 <= p <= 5]


def load(folder, name):
    _, runs = parse(os.path.join(folder, name))
    run = max(runs, key=len)  # files with appended restarts: keep the longest run
    return len(run), valid_pairs(run)


def mae_rmse(pairs):
    n = len(pairs)
    return (sum(abs(t - p) for t, p in pairs) / n,
            math.sqrt(sum((t - p) ** 2 for t, p in pairs) / n))


def first_match(run_len, pairs, mae, rmse):
    """Smallest prefix (in logged examples) whose running MAE/RMSE rounds to the paper value."""
    for k in range(1, len(pairs) + 1):
        m, r = mae_rmse(pairs[:k])
        if abs(m - mae) < 0.0015 and abs(r - rmse) < 0.0015:
            return k
    return None


def cp_quantile(scores, eps):
    """Finite-sample conformal quantile: ceil((n+1)(1-eps))-th smallest score."""
    s = sorted(scores)
    k = math.ceil((len(s) + 1) * (1 - eps))
    return float("inf") if k > len(s) else s[k - 1]


def clipped_width(pred, tau):
    """Width of [pred - tau, pred + tau] intersected with the 1..5 rating scale."""
    return min(5.0, pred + tau) - max(1.0, pred - tau)


def split_cp(pairs, eps, rng):
    widths, clipped, covs = [], [], []
    idx = list(range(len(pairs)))
    n_cal = max(int(len(pairs) * CAL_RATIO), 1)
    for _ in range(N_SPLITS):
        rng.shuffle(idx)
        cal, test = idx[:n_cal], idx[n_cal:]
        tau = cp_quantile([abs(pairs[i][0] - pairs[i][1]) for i in cal], eps)
        covs.append(sum(abs(pairs[i][0] - pairs[i][1]) <= tau for i in test) / len(test))
        widths.append(2 * tau)
        clipped.append(sum(clipped_width(pairs[i][1], tau) for i in test) / len(test))
    return sum(widths) / N_SPLITS, sum(clipped) / N_SPLITS, 100 * sum(covs) / N_SPLITS


def sliding_cp(pairs, eps):
    scores = [abs(t - p) for t, p in pairs]
    clipped, hits = [], []
    for i in range(WINDOW, len(scores)):
        tau = cp_quantile(scores[i - WINDOW:i], eps)
        clipped.append(clipped_width(pairs[i][1], tau))
        hits.append(scores[i] <= tau)
    if not hits:
        return None, None
    return sum(clipped) / len(clipped), 100 * sum(hits) / len(hits)


def fmt(x, d=3):
    return "–" if x is None else ("∞" if x == float("inf") else f"{x:.{d}f}")


def main(folder):
    print("# Recomputed CPR-LLM metrics\n")
    print(f"Source logs: `{os.path.abspath(folder)}`. Generated by `recompute.py`.\n")

    print("## Table 5: final MAE/RMSE vs. paper\n")
    print("| Model | Setting | Test size | Valid preds | Paper MAE / RMSE | Final MAE / RMSE | Paper value first reached after | Results file |")
    print("|---|---|---|---|---|---|---|---|")
    for model, setting, name, p_mae, p_rmse in TABLE5:
        n, pairs = load(folder, name)
        mae, rmse = mae_rmse(pairs)
        k = first_match(n, pairs, p_mae, p_rmse)
        when = "never" if k is None else ("end of run" if k == len(pairs) else f"{k} of {len(pairs)} valid preds")
        print(f"| {model} | {setting} | {n} | {len(pairs)} ({100 * len(pairs) / n:.0f}%) | "
              f"{p_mae:.3f} / {p_rmse:.3f} | {mae:.3f} / {rmse:.3f} | {when} | `{name}` |")

    print("\n## Table 7: conformal prediction on the real errors\n")
    print(f"Nonconformity score |r − r̂|. Split CP: {int(CAL_RATIO * 100)}% calibration, mean over {N_SPLITS} random splits. "
          f"Sliding-window CP: τ from the previous {WINDOW} scores, evaluated on the remaining examples in log order. "
          "Raw width is 2τ as in the paper; clipped width intersects each interval with the 1–5 scale. "
          "Scores are integers, so τ moves in whole steps. ∞ means the calibration set is too small for that ε "
          "(the interval is the whole scale). Sliding CP needs more than 100 valid predictions.\n")
    print("| ε | Model | Setting | Valid preds | Split CP raw width | Split CP clipped width | Split CP coverage % | Sliding CP clipped width | Sliding CP coverage % |")
    print("|---|---|---|---|---|---|---|---|---|")
    rng = random.Random(0)
    for eps in EPSILONS:
        for model, setting, name in TABLE7:
            _, pairs = load(folder, name)
            sw, sk, sc = split_cp(pairs, eps, rng)
            ww, wc = sliding_cp(pairs, eps)
            print(f"| {eps} | {model} | {setting} | {len(pairs)} | {fmt(sw, 2)} | {fmt(sk, 2)} | {fmt(sc, 2)} | {fmt(ww, 2)} | {fmt(wc, 2)} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "small_data"))
