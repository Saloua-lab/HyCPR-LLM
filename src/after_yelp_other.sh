#!/bin/sh
# Waits for jobs/yelp_other_llms.json, then rebuilds the Yelp fusion, bootstrap, Table 5 body and CI table.
# Re-runnable: run_queue.py skips finished jobs and train.py resumes from its last checkpoint.
cd "$(dirname "$0")"
export HF_HOME=../.hf_cache
python run_queue.py jobs/yelp_other_llms.json --gpus ${GPUS:-0 1 2} &&
python large_baselines.py hybrid > logs/large_hybrid_yelp.log 2>&1 &&
python bootstrap.py > logs/bootstrap.log 2>&1 &&
python table4.py > results/table5_hycpr_body.tex &&
python ci_table.py > results/ci_table_hycpr.tex &&
echo "after_yelp_other done $(date)"
