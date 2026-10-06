"""给定一组波段，训练若干分类器成员，把每个成员在测试集上的概率和权重存盘，softmax 平均为最终集成结果。

python -m mhdrl.boost --json runs/indian/mhdrl/seed0.json --members penet:21:200:0:0,penet:21:200:0:1,penet:25:200:0:0,penet:25:200:0:1
成员格式 arch:patch:epochs:balanced:init，概率存到 runs/<ds>/boost/<名>/<成员>.npy，权重（fp16）存到同名 .pt；
输出目录可直接作为 mhdrl.infer 的 --weights。
"""
import argparse
import json
import os

import numpy as np
import torch

from .classify import _fit, metrics, predict
from .data import DATASETS, PatchSampler, load, split

RUNS = os.environ.get('HSI_RUNS', 'runs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='mhdrl.run 的结果文件（含 bands 与 args.ds / args.seed）')
    ap.add_argument('--members', required=True)
    ap.add_argument('--name', default='', help='输出子目录名，默认 <method>_seed<k>')
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    d = json.load(open(a.json))
    ds, seed = d['args']['ds'], d['args']['seed']
    name = a.name or f'{d["args"]["tag"] or d["args"]["method"]}_seed{seed}'
    out = os.path.join(RUNS, ds, 'boost', name)
    os.makedirs(out, exist_ok=True)
    members = a.members.split(',')

    cube, gt, n_cls = load(ds)
    tr, ytr, te, yte = split(gt, DATASETS[ds]['ratio'], seed)
    np.save(os.path.join(out, 'y_test.npy'), yte)
    sub = cube[:, :, np.asarray(d['bands'])]
    for mem in members:
        f = os.path.join(out, mem.replace(':', '_') + '.npy')
        if os.path.exists(f):
            continue
        arch, patch, epochs, bal, init = mem.split(':')
        sampler = PatchSampler(sub, int(patch))
        model = _fit(sampler, sub.shape[2], n_cls, tr, ytr, seed * 1000 + 500 + int(init), arch, int(patch),
                     int(epochs), 64, 1e-3, 1e-4, 0.1, 'cuda', balanced=bool(int(bal)))
        p = predict(model, sampler, te, True)
        np.save(f, p.astype(np.float32))
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   f.replace('.npy', '.pt'))
        m = metrics(yte, p.argmax(1), n_cls)
        print(f'{mem}: OA {m["oa"] * 100:.2f} AA {m["aa"] * 100:.2f}', flush=True)
        del model, sampler
        torch.cuda.empty_cache()

    probs = [np.load(os.path.join(out, mem.replace(':', '_') + '.npy')) for mem in members]
    m = metrics(yte, sum(probs).argmax(1), n_cls)
    m.update(members=members, bands=d['bands'], seed=seed, ds=ds)
    json.dump(m, open(os.path.join(out, 'ensemble.json'), 'w'), indent=1)
    print(f'ENSEMBLE ({len(probs)} members): OA {m["oa"] * 100:.2f} AA {m["aa"] * 100:.2f} '
          f'Kappa {m["kappa"] * 100:.2f}', flush=True)


if __name__ == '__main__':
    main()
