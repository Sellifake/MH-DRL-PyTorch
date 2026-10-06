"""给定一组波段，训练若干分类器成员，把每个成员在测试集上的概率存盘，集成方式事后离线组合。

python -m mhdrl.boost --ds indian --json runs/indian/mhdrl/seed0.json --members penet:21:200:0:0,penet:21:200:0:1,penet:25:200:0:0,penet:25:200:0:1
成员格式 arch:patch:epochs:balanced:init，概率存到 runs/<ds>/boost/<名>/<成员>.npy，权重（fp16）存到同名 .pt
"""
import argparse
import json
import os

import numpy as np
import torch

from .classify import _fit, metrics, predict
from .data import DATASETS, PatchSampler, load, split

ap = argparse.ArgumentParser()
ap.add_argument('--ds', default='indian')
ap.add_argument('--json', required=True)
ap.add_argument('--members', required=True)
ap.add_argument('--name', default='')
a = ap.parse_args()
torch.backends.cudnn.benchmark = True
RUNS = os.environ.get('HSI_RUNS', 'runs')
d = json.load(open(a.json))
seed = int(os.path.basename(a.json)[4:-5])
name = a.name or f'{a.json.split("/")[-2]}_seed{seed}'
out = os.path.join(RUNS, a.ds, 'boost', name)
os.makedirs(out, exist_ok=True)
cube, gt, n_cls = load(a.ds)
tr, ytr, te, yte = split(gt, DATASETS[a.ds]['ratio'], seed)
np.save(os.path.join(out, 'y_test.npy'), yte)
sub = cube[:, :, np.asarray(d['bands'])]
for mem in a.members.split(','):
    f = os.path.join(out, mem.replace(':', '_') + '.npy')
    if os.path.exists(f):
        continue
    arch, patch, epochs, bal, init = mem.split(':')
    sampler = PatchSampler(sub, int(patch))
    model = _fit(sampler, sub.shape[2], n_cls, tr, ytr, seed * 1000 + 500 + int(init), arch, int(patch), int(epochs),
                 64, 1e-3, 1e-4, 0.1, 'cuda', balanced=bool(int(bal)))
    p = predict(model, sampler, te, True)
    np.save(f, p.astype(np.float32))
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
               f.replace('.npy', '.pt'))
    m = metrics(yte, p.argmax(1), n_cls)
    print(f'{mem}: OA {m["oa"] * 100:.2f} AA {m["aa"] * 100:.2f}', flush=True)
    del model, sampler
    torch.cuda.empty_cache()

# 全部成员的 softmax 平均 = 最终集成结果
probs = [np.load(os.path.join(out, mem.replace(':', '_') + '.npy')) for mem in a.members.split(',')]
m = metrics(yte, sum(probs).argmax(1), n_cls)
m['members'] = a.members.split(',')
m['bands'] = d['bands']
m['seed'] = seed
m['ds'] = a.ds
json.dump(m, open(os.path.join(out, 'ensemble.json'), 'w'), indent=1)
print(f'ENSEMBLE ({len(probs)} members): OA {m["oa"] * 100:.2f} AA {m["aa"] * 100:.2f} Kappa {m["kappa"] * 100:.2f}',
      flush=True)
