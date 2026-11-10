"""LaTeX rows of the baseline part of Table 4 (tab:mae_rmse_rotated_mindH) from the same-split baseline runs.

The -H columns share their test interactions with the -WH columns (LDOS-H/LDOS-WH, Depaul-H/Depaul-WH,
Restaurant-H/Tijuana-WH, YELP-H/YELP-WH; checked in baselines_same_split.py / prompt_table.py), and the baselines
do not read the prompt, so their -H values equal the -WH ones.

Usage: python3 table4_rows.py [results/baselines_small.json results/baselines_yelp.json]
"""
import json
import sys

COLS = [("LDOS", "LDOS-WH"), ("Depaul", "Depaul-WH"), ("Tijuana", "Tijuana-WH"), ("LDOS", "LDOS-H"),
        ("Depaul", "Depaul-H"), ("Tijuana", "Restaurant-H"), ("YELP", "YELP-WH"), ("YELP", "YELP-H")]
GROUPS = [("Heuristic", ["GAR", "IAR", "UAR"]),
          ("Traditional", ["ItemKNN", "UserKNN", "SVD++", "BiasedMF"]),
          ("CA", ["FM", "CAMF_CI", "CAMF_CU", "CAMF_C", "CAMF_CUCI"]),
          ("Neural CA", ["NFM", "DeepFM", "AutoInt"])]


def main(paths):
    res = {}
    for p in paths:
        for k, v in json.load(open(p)).items():
            res.setdefault(k, {"models": {}})["models"].update(v["models"])
    for group, models in GROUPS:
        print("\\midrule")
        print(f"\\multirow{{{len(models)}}}{{*}}{{{group}}}")
        for m in models:
            cells = []
            for ds, _ in COLS:
                v = res.get(ds, {}).get("models", {}).get(m)
                cells += [f"{v['mae']:.3f}", f"{v['rmse']:.3f}"] if v else ["--", "--"]
            print(f"& {m.replace('_', chr(92) + '_'):10s}& " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/baselines_small.json", "results/baselines_yelp.json",
                         "results/baselines_neural_small.json", "results/baselines_neural_yelp.json"])
