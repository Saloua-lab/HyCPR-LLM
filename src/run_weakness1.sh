#!/bin/bash
# Weakness 1 (2026-11-06): neural CARS baselines in the fusion pool + refit of the chosen CF model on the full
# training data. Detached runner: setsid nohup bash run_weakness1.sh > logs/run_weakness1.log 2>&1 &
cd "$(dirname "$0")"
PY=python
set -x
CUDA_VISIBLE_DEVICES=1 THREADS=16 $PY large_baselines.py fit amazon > logs/lb_fit_amazon.log 2>&1 &
AMZ=$!
run() { CUDA_VISIBLE_DEVICES=0 $PY hybrid.py "$@" --neural --refit; }
run llama --only v0 --tag hycpr_seed2 --seed 2 > logs/hy_llama_s2.log 2>&1
for m in eurollm deepseek; do
  run $m --only v0 --tag hycpr > logs/hy_$m.log 2>&1
  for k in 1 2; do run $m --only v0 --tag hycpr_seed$k --seed $k > logs/hy_${m}_s$k.log 2>&1; done
done
wait $AMZ
$PY large_baselines.py hybrid > logs/lb_hybrid.log 2>&1
$PY large_baselines.py compare > logs/lb_compare.log 2>&1
$PY bootstrap.py > logs/bootstrap.log 2>&1
$PY table4.py > results/table4_body.tex 2> logs/table4.log
$PY ci_table.py > results/ci_table_hycpr.tex 2> logs/ci_table.log
$PY cp_hybrid.py > logs/cp_hybrid.log 2>&1
$PY seed_summary_hycpr.py > logs/seed_summary.log 2>&1
$PY table6_rows.py > results/table6_rows.tex 2> logs/table6_rows.log
echo ALL_DONE
