#!/bin/bash
# Weakness 4 (2026-11-07): pretraining-contamination probes on the 4 base models + impact analysis. Skip-if-done.
#   setsid nohup bash run_contamination.sh > logs/run_contamination.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export HF_HOME=../.hf_cache
probe() {  # model gpu
  if [ -f results/contamination/$1.json ] && python3 -c "import json,sys;d=json.load(open('results/contamination/$1.json'));sys.exit(0 if d['ml_ratings']['n']>100 else 1)"; then
    return
  fi
  CUDA_VISIBLE_DEVICES=$2 python contamination.py --model $1 > logs/contamination_$1.log 2>&1
  grep -E "ml_items|ml_ratings|az_items|Error" logs/contamination_$1.log | cut -c1-300
}
(probe mistral 0; probe deepseek 0) & probe llama 1 & probe eurollm 2 & wait
python contamination_impact.py > logs/contamination_impact.log 2>&1
tail -3 logs/contamination_impact.log
echo ALL_DONE
