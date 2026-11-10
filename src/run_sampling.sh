#!/bin/bash
# Review 2, point 2 (2026-11-07): sampling-based (black-box) predictions, Mistral. Skip-if-done. Detached:
#   setsid nohup bash run_sampling.sh > logs/run_sampling.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export HF_HOME=../.hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=mistralai/Mistral-7B-Instruct-v0.2
run() { # gpu data adapter out
  [ -f probs/sampled/$4.json ] && return
  CUDA_VISIBLE_DEVICES=$1 python sample_eval.py --data $2 --adapter $3 --base $B --out probs/sampled/$4.json --batch 2 > logs/sample_$4.log 2>&1
  tail -1 logs/sample_$4.log
}
( run 0 data/LocationLDOSWithHisAvecMistral_noleak adapters/mistral_clean_LDOS-H LDOS-H
  run 0 data/LocationDepaulWithHisAvecMistral_noleak adapters/mistral_clean_Depaul-H Depaul-H
  run 0 data/LocationRestaurantWithHisAvecMistral_noleak adapters/mistral_clean_Restaurant-H Restaurant-H ) &
( run 1 data/MovieLens-H adapters/mistral_full_MovieLens-H MovieLens-H
  run 1 data/MovieLens-H_val adapters/mistral_full_MovieLens-H MovieLens-H_val ) &
( run 2 data/AmazonBooks-H adapters/mistral_full_AmazonBooks-H AmazonBooks-H
  run 2 data/AmazonBooks-H_val adapters/mistral_full_AmazonBooks-H AmazonBooks-H_val ) &
wait
echo ALL_DONE
