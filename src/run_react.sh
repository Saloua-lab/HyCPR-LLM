#!/bin/bash
# Weakness 3 (2026-11-06): ReAct with real tools + ablation. Skip-if-done. Detached:
#   setsid nohup bash run_react.sh > logs/run_react.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export HF_HOME=../.hf_cache
run_ds() {  # dataset gpu
  for c in direct react_notools react_all react_minus_user react_minus_item react_minus_similar react_minus_context; do
    [ -f results/react/$1_$c.json ] && [ "$(python3 -c "import json;print(len(json.load(open('results/react/$1_$c.json'))['truth']))")" -gt 10 ] && continue
    CUDA_VISIBLE_DEVICES=$2 python react_tools.py --dataset $1 --config $c > logs/react_$1_$c.log 2>&1
    tail -1 logs/react_$1_$c.log
  done
}
run_ds LDOS 0 & run_ds Depaul 1 & run_ds Tijuana 2 & wait
echo ALL_DONE
