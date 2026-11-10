# HyCPR-LLM

Code for the paper **"HyCPR-LLM: Enhancing LLM-based Rating Predictions via Prompt Engineering, Collaborative
Filtering Fusion and Conformal Prediction"** (Saloua Zammali, Sadok Ben Yahia).

HyCPR-LLM fine-tunes an instruction-tuned LLM with LoRA to predict a 1–5 rating for a (user, item, context)
prompt. It reads the rating as the expected value of the next-token distribution over the tokens "1"…"5", and
fuses it with a collaborative-filtering (CF) model: `w · LLM + (1 − w) · CF`, with the CF model and `w` chosen on
a validation split. Conformal prediction (CP) then turns each prediction into an interval or a set of ratings
with a coverage guarantee. **CPR-LLM** is the LLM alone, without the CF fusion.

The repository contains all the code behind the paper, and the per-prompt predictions of every model. With the
predictions, every table and statistical test of the paper can be rebuilt on a CPU, without retraining
(see [Rebuilding the tables without a GPU](#rebuilding-the-tables-without-a-gpu)).

## Layout

```
HyCPR-LLM/
├── src/            all the code; every command below is run from this folder
│   ├── jobs/       the exact training / evaluation commands used for the paper (JSON job lists)
│   ├── probs/      per-prompt predictions of every model (rating-token probabilities, expected rating)
│   ├── results/    result files of every analysis (JSON, Markdown, LaTeX table bodies)
│   └── figures/    figures of the paper made by the code
├── raw_data/       public datasets, to download (see raw_data/README.md)
├── small_data/     prompt datasets of the small context-aware datasets (see small_data/README.md)
└── external/       official code of LLaRA, CoLLM and A-LLMRec, to clone (see external/README.md)
```

The scripts in `src/` import each other, and read and write `data/`, `adapters/`, `cf/`, `probs/` and `results/`
relative to `src/`. `data/`, `adapters/` and `cf/` are created by the scripts and are not part of the repository.

## Setup

Python 3.12, a CUDA GPU (we used H100 80 GB) for training and LLM evaluation.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login     # Llama-2 and EuroLLM are gated models on Hugging Face
```

LLMs: `mistralai/Mistral-7B-Instruct-v0.2` (main model), `meta-llama/Llama-2-7b-chat-hf`,
`utter-project/EuroLLM-9B-Instruct`, `deepseek-ai/deepseek-llm-7b-chat`, and their non-instruct base models for
the ablation (`BASE` dict in `src/make_jobs.py`).

## Pipeline

### 1. Data and prompts

| Datasets | Script | Output |
|---|---|---|
| MovieLens-1M, Amazon-Books, Yelp, MIND | `python prepare_large.py {movielens,amazon,yelp,mind}` | `data/<Name>-H`, `data/<Name>-WH` |
| LDOS-CoMoDa, DePaul Movies, Tijuana Restaurants | prompt datasets in `small_data/` | – |
| leak-free histories of the small -H test sets | `python rebuild_history.py ../small_data/<test set> ...` | `data/<test set>_noleak` |
| validation splits for the fusion | `python make_variants.py ...` | `data/variants/` |

`prepare_large.py` uses a global chronological 90/10 split, with histories taken strictly before the target
interaction. `-H` prompts include the user's history, `-WH` prompts do not.

### 2. Fine-tuning and evaluation (GPU)

```bash
python train.py --data <DatasetDict> --base <hf id> --out adapters/<name> [--epochs 10]
python evaluate_probs.py --4bit --data <DatasetDict> --adapter adapters/<name> --base <hf id> --out probs/<name>.json
```

- `train.py`: 4-bit NF4 base model, LoRA r = 32 / α = 64 on all projections and `lm_head`, lr 2e-4.
- `evaluate_probs.py`: the evaluation protocol of the paper. It cuts the prompt after the answer marker,
  restricts the next-token softmax to the tokens "1"…"5", and predicts the expected rating.

Every training and evaluation run of the paper is listed in `jobs/*.json`, with its exact command.
`python run_queue.py jobs/<file>.json --gpus 0 1 2` runs a job list, one job per GPU, and skips finished jobs.
`make_jobs.py` regenerates the job lists.

### 3. Baselines

| Baselines | Script |
|---|---|
| CF / CARS baselines on the same split (small datasets, Yelp) | `baselines_same_split.py`, `baselines.py`, `neural_cars.py` (NFM, DeepFM, AutoInt) |
| baselines on MovieLens-1M / Amazon-Books / Yelp | `large_baselines.py fit <dataset>` |
| CoLLM (official settings) | `collm.py --dataset <d> --official` |
| LLaRA (official modules) | `llara_rating.py --dataset <d> --lr 2e-4` |
| A-LLMRec (official modules) | `allmrec_rating.py --dataset <d>` |

The three LLM–CF methods are adapted to rating prediction with Mistral-7B, on the same split and with the same
rating read-out as HyCPR-LLM. `llara_rating.py` and `allmrec_rating.py` import modules from the official
repositories in `external/`. The settings taken from the official code are listed in each script's docstring.

### 4. Fusion (HyCPR-LLM)

- `hybrid.py <llm> --only v0 --tag hycpr`: small datasets. It fits every CF model, and chooses the CF model and
  `w` on validation RMSE.
- `large_baselines.py hybrid`: the same for MovieLens-1M, Amazon-Books and Yelp.
- Fused predictions: `probs/hybrid/<llm>_hycpr_<setting>.json`.

## Paper items

| Paper item | Script(s) | Output |
|---|---|---|
| Table `tab:mae_rmse_rotated_mindH` (main results, small datasets and Yelp) | `table4.py`, `table4_rows.py`, `bootstrap.py` | `results/table5_hycpr_body.tex` |
| Table `tab:baselines-user-rating` (MovieLens-1M, Amazon-Books) | `large_baselines.py compare` / `hybrid`, `table6_rows.py`, `llmcf_compare.py` | `results/table6_rows.tex`, `results/llmcf.json` |
| Table `tab:gen-vs-prob` (expected rating vs. parsed text) | `gen_vs_prob.py` | `results/gen_vs_prob.md` |
| Table `tab:stacking` (learned and gated fusion) | `stacking.py`, `gating.py` | `results/stacking.md`, `results/gating.md` |
| Table `tab:seeds` (training seeds) | `seed_summary_hycpr.py`, `seed_ensemble.py` | `results/seed_table.tex` |
| Table `tab:contamination` (pretraining memorization) | `contamination.py`, `contamination_impact.py` | `results/contamination_impact.md` |
| Table `tab:sampling` (black-box sampling) | `sample_eval.py`, `sample_analysis.py` | `results/sampling.md` |
| Table `tab:segments` (where the fusion helps) | `segments.py` | `results/segments_table.tex` |
| Multiple comparisons (Holm), effect sizes | `holm.py` | `results/holm.md` |
| Table `tab:conformal_prediction_by_significance` (RQ2) | `cp_hybrid.py`, `cp_probs.py` | `results/cp_hybrid.md` |
| Tables `tab:cp-conditional`, `tab:cp-adaptive`, `tab:cp-groups` | `cp_extended.py`, `cp_adaptive.py`, `cp_groups.py`, `cp_actionable.py` | `results/cp_*.md`, `results/cp_*_table.tex` |
| Figure `fig:history-impact`, Table `tab:history-measured` (RQ3) | `evaluate_probs.py --ks 0 1 2 3 4 5` (`jobs/history_clean.json`), `plot_history_scored.py` | `figures/*_history_scored.pdf` |
| Ablation (RQ4), tables `tab:retrained-clean`, `tab:training-variants` | `bootstrap.py` on the predictions of `jobs/noit_clean.json`, `jobs/phase5_clean.json`, `jobs/improve_mistral.json`; `hybrid.py` | `results/bootstrap.md`, `results/improve_mistral.md` (values copied into the tables by hand) |
| Tables `tab:react-tools`, `tab:react-cost` (RQ5) | `react_tools.py`, `react_summary.py`, `react_cost.py` | `results/react_table.tex`, `results/react_cost.md` |
| Appendix tables `tab:ci`, `tab:ci2` (bootstrap intervals) | `ci_table.py` | `results/ci_table_hycpr.tex` |
| Table `tab:stacking`, BMA and pseudo-BMA+ columns | `bma.py` | `results/bma.md` |
| Cold-start-aware fusion weight (RQ1, learned fusion) | `cold_aware.py` | `results/cold_aware.json` |
| Criterion for when the fusion wins (RQ1) | `when_wins.py` | `results/when_wins.json` |
| Table `tab:cold-start` (cold-start regimes) | `cold_start.py` (after `text_knn.py`) | `results/cold_start.md` |
| Table `tab:textknn` (semantic Text-KNN baseline) | `text_knn.py` | `results/text_knn.md`, `probs/textknn_*.json` |
| Table `tab:cp-adaptive`, localized CP column | `cp_localized.py` | `results/cp_localized.md` |
| Table `tab:recgpt` (domain-pretrained LLM, RecGPT-7B-Instruct) | `jobs/recgpt.json` via `run_queue.py`, then `recgpt_compare.py` | `results/recgpt.md`, `probs/recgpt_*`, `probs/hybrid/recgpt_hycpr_*` |
| Table `tab:cprae` (CPR-AE, autoencoder-based conformal recommender, re-implementation) | `cpr_ae.py` | `results/cpr_ae.md`, `probs/cprae_*.json` |
| Calibrated / ordinal read-outs (RQ1 text) | `ordinal_calib.py` | `results/ordinal_calib.md` |
| Robustness of the fusion weight (RQ1 text) | `fusion_robust.py` | `results/fusion_robust.md` |
| Learned CQR and CF-only conformal baselines (RQ2 text) | `cp_cqr.py` | `results/cp_cqr.md` |
| Conformal intervals without logits (RQ2 text) | `cp_sampled.py` | `results/cp_sampled.md` |

All significance tests are paired bootstraps over the test interactions: 10,000 replicates, with a Holm
correction in `holm.py`.

The `run_*.sh` and `resume_improve.sh` files in `src/` are the shell chains we used to run groups of jobs, kept
as a record of the order of the runs.

## Rebuilding the tables without a GPU

`probs/` and `results/` hold the outputs of the GPU runs, so the analysis scripts run on a CPU:

```bash
cd src
python table4.py > results/table5_hycpr_body.tex
python holm.py; python cp_adaptive.py; python cp_hybrid.py; python stacking.py; python gating.py
python llmcf_compare.py; python seed_summary_hycpr.py; python react_summary.py; python plot_history_scored.py
```

Scripts that refit a CF model or read the prompts (`hybrid.py`, `segments.py`, `cp_groups.py`, `large_baselines.py`)
also need the datasets of `raw_data/` and `small_data/`.

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets keep their own licences. The files in
`external/` come from their authors' repositories, under those repositories' licences.

## Citation

If you use this code, please cite the paper (the reference will be added on publication).
