#!/bin/bash
# Review 3, point 2 (2026-11-08): tool outputs given directly in the prompt + cost of each strategy. Detached:
#   setsid nohup bash run_react2.sh > logs/run_react2.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
export HF_HOME=../.hf_cache
run_ds() { # dataset gpu
  for c in direct_tools direct react_notools react_all; do
    tag=""; [ "$c" != direct_tools ] && tag="_timed"
    [ -f results/react/$1_$c$tag.json ] && continue
    CUDA_VISIBLE_DEVICES=$2 python react_tools.py --dataset $1 --config $c --tag "$tag" > logs/react2_$1_$c.log 2>&1
    tail -1 logs/react2_$1_$c.log
  done
}
run_ds LDOS 0 & run_ds Depaul 1 & run_ds Tijuana 2 & wait
echo ALL_DONE
