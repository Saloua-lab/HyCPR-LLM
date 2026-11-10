#!/bin/sh
# Restart the improvement + Table 6 pipeline after a UCloud disconnect. Safe to run any time: finished jobs
# are skipped (run_queue.py checks each job's output file) and training resumes from its last checkpoint.
#   GPUS="0 1 2" sh src/resume_improve.sh > logs/resume_improve.log 2>&1 &
# Check the GPUs first with nvidia-smi -L and set GPUS to the ones the new job has.
cd "$(dirname "$0")"
export HF_HOME=../.hf_cache
G=${GPUS:-0 1 2}
python run_queue.py jobs/v0_seeds_mistral.json --gpus $G &&
python hybrid.py mistral > /dev/null &&
echo "improve_mistral updated $(date)" &&
for d in movielens amazon; do
  n=$([ $d = movielens ] && echo MovieLens || echo AmazonBooks)
  [ -f results/large_baselines_$n.json ] || CUDA_VISIBLE_DEVICES= python large_baselines.py fit $d
done &&
python run_queue.py jobs/table6_mistral.json --gpus $G &&
python large_baselines.py compare &&
python run_queue.py jobs/table6_val_mistral.json --gpus $G &&
python large_baselines.py hybrid > /dev/null &&
python run_queue.py jobs/v0_other_llms.json --gpus $G &&
[ -f results/large_baselines_YELP.json ] || CUDA_VISIBLE_DEVICES= python large_baselines.py fit yelp &&
python run_queue.py jobs/yelp_val_mistral.json --gpus $G &&
python large_baselines.py hybrid > /dev/null &&
for m in mistral llama eurollm deepseek; do python hybrid.py $m --only v0 --tag hycpr > /dev/null; done &&
echo "ALL IMPROVE + TABLE6 JOBS FINISHED $(date)"
