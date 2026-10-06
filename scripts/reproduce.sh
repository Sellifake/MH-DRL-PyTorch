#!/usr/bin/env bash
# 全部实验：Indian 60 / PaviaU 30 / Houston 40 个波段，种子 0–2
# 数据放在 ./data（或设置 HSI_DATA），结果写到 ./runs（或设置 HSI_RUNS）
set -e
METHODS="mhdrl bsnets sicnn twcnn abcnn mrsvm drlbs ddcnn"
for ds in indian paviau houston; do
  for s in 0 1 2; do
    for m in $METHODS; do
      python -m mhdrl.run --ds $ds --method $m --seed $s
    done
    python -m mhdrl.boost --ds $ds --json runs/$ds/mhdrl/seed$s.json \
      --members penet:21:200:0:0,penet:21:200:0:1,penet:25:200:0:0,penet:25:200:0:1
  done
done
python -m mhdrl.report --seeds 0,1,2 --out results
python -m mhdrl.figures --seeds 0,1,2 --pick indian:0,paviau:2,houston:2 --out results/figs
