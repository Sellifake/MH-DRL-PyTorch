#!/usr/bin/env bash
# 全部实验：Indian 60 / PaviaU 30 / Houston 40 个波段，种子 0–2
# 数据放在 ./data（或设置 HSI_DATA），结果写到 ./runs（或设置 HSI_RUNS）
set -e
RUNS=${HSI_RUNS:-runs}
BASELINES="bsnets sicnn twcnn abcnn mrsvm drlbs ddcnn"
MEMBERS=penet:21:200:0:0,penet:21:200:0:1,penet:25:200:0:0,penet:25:200:0:1
for ds in indian paviau houston; do
  for s in 0 1 2; do
    for m in $BASELINES; do
      python -m mhdrl.run --ds $ds --method $m --seed $s
    done
    # MH-DRL：先选波段，再用 4 个 PE-Net 的集成评测（输出目录即 infer 的 --weights）
    python -m mhdrl.run --ds $ds --method mhdrl --seed $s --no_eval
    python -m mhdrl.boost --json $RUNS/$ds/mhdrl/seed$s.json --members $MEMBERS
  done
done
python -m mhdrl.report --seeds 0,1,2 --out results
for p in indian:0 paviau:2 houston:2; do
  ds=${p%:*}; s=${p#*:}
  python -m mhdrl.infer --ds $ds --weights $RUNS/$ds/boost/mhdrl_seed$s --map results/maps/$ds.png
done
python -m mhdrl.figures --seeds 0,1,2 --pick indian:0,paviau:2,houston:2 --maps results/maps --out results/figs
