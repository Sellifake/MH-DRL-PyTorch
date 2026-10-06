"""一次运行 = 一个数据集 × 一个方法 × 一个划分种子：在该划分的训练集上选波段，再在测试集上评测。
结果写到 $HSI_RUNS/<ds>/<tag>/seed<k>.json。

python -m mhdrl.run --ds indian --method mhdrl --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .agent import corr_matrix, run_mhdrl
from .classify import train_eval
from .data import DATASETS, load, split
from .evaluate import own_eval
from .masked import MaskedEvaluator
from . import baselines as B
from . import teachers as T

RUNS = os.environ.get('HSI_RUNS', 'runs')


def cached_scores(ds, name, seed, fn, log, key=''):
    path = os.path.join(RUNS, ds, 'scores', f'{name}{key}_seed{seed}.npy')
    if os.path.exists(path):
        return np.load(path)
    t0 = time.time()
    s = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, s)
    log(f'  {name} scores computed in {time.time() - t0:.0f}s')
    return s


def main():
    torch.backends.cudnn.benchmark = True
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='indian')
    ap.add_argument('--method', default='mhdrl')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nbands', type=int, default=0, help='0 = 论文默认（Indian 60 / PaviaU 30 / Houston 40）')
    ap.add_argument('--tag', default='', help='结果子目录名，默认等于 method')
    # 最终评测分类器
    ap.add_argument('--cls', default='penet')
    ap.add_argument('--cls_patch', type=int, default=25)
    ap.add_argument('--cls_pca', type=int, default=0)
    ap.add_argument('--cls_epochs', type=int, default=200)
    ap.add_argument('--cls_ens', type=int, default=3)
    ap.add_argument('--no_eval', action='store_true', help='只选波段不评测（MH-DRL 的最终评测交给 mhdrl.boost）')
    # MH-DRL
    ap.add_argument('--pe_patch', type=int, default=15)
    ap.add_argument('--pe_iters', type=int, default=2000)
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--guide', type=int, default=300)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--reward', default='bsoft')
    ap.add_argument('--no_center', action='store_true')
    ap.add_argument('--learn_in_guide', action='store_true')
    ap.add_argument('--teachers', default='bsnets,sicnn,twcnn')
    ap.add_argument('--dqn_lr', type=float, default=1e-3)
    ap.add_argument('--act', default='topk')
    ap.add_argument('--updates', type=int, default=4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--rescale', type=int, default=0)
    a = ap.parse_args()

    tag = a.tag or a.method
    out = os.path.join(RUNS, a.ds, tag, f'seed{a.seed}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    logf = open(out.replace('.json', '.log'), 'w')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    cube, gt, n_cls = load(a.ds)
    tr, ytr, te, yte = split(gt, DATASETS[a.ds]['ratio'], a.seed)
    D = cube.shape[2]
    nd = a.nbands or DATASETS[a.ds]['nbands']
    log(f'{a.ds} {cube.shape} train={len(tr)} test={len(te)} method={a.method} nd={nd} seed={a.seed}')
    log(json.dumps(vars(a)))

    score_fns = {
        'bsnets': lambda: T.bsnets_scores(cube, a.seed),
        'twcnn': lambda: T.twcnn_scores(cube, tr, ytr, n_cls, nd, a.seed),
        'sicnn': lambda: T.sicnn_scores(cube, tr, ytr, n_cls, a.seed, log=log),
        'abcnn': lambda: B.abcnn_scores(cube, tr, ytr, n_cls, a.seed),
        'mrsvm': lambda: B.mrsvm_scores(cube, tr, ytr, a.seed),
    }
    t0 = time.time()
    extra = {}
    if a.method == 'all':
        bands = np.arange(D)
    elif a.method == 'random':
        bands = np.sort(np.random.RandomState(a.seed).choice(D, nd, replace=False))
    elif a.method == 'uniform':
        bands = np.unique(np.round(np.linspace(0, D - 1, nd)).astype(int))
    elif a.method in score_fns:
        s = cached_scores(a.ds, a.method, a.seed, score_fns[a.method], log, f'_k{nd}' if a.method == 'twcnn' else '')
        bands = np.sort(np.argsort(-s)[:nd])
    elif a.method == 'drlbs':
        bands = np.sort(B.drlbs_bands(cube, nd, a.seed))
    elif a.method == 'ddcnn':
        bands = B.ddcnn_bands(cube, tr, ytr, n_cls, nd, a.seed, log=log)
    elif a.method == 'mhdrl':
        teach = [(n, cached_scores(a.ds, n, a.seed, score_fns[n], log, f'_k{nd}' if n == 'twcnn' else ''))
                 for n in a.teachers.split(',') if n]
        t1 = time.time()
        ev = MaskedEvaluator(cube, tr, ytr, n_cls, 'penet', a.pe_patch, a.folds, a.pe_iters, seed=a.seed, log=log,
                             rescale=bool(a.rescale),
                             cache=os.path.join(RUNS, a.ds, 'evaluator',
                                                f'penet_p{a.pe_patch}_it{a.pe_iters}_rs{a.rescale}_bnb_seed{a.seed}'))
        log(f'  PE-Net trained in {time.time() - t1:.0f}s')
        bands, hist = run_mhdrl(ev, corr_matrix(cube), nd, teach, a.seed, a.steps, a.guide, a.alpha, a.reward,
                                not a.no_center, lr=a.dqn_lr, batch=a.batch, learn_in_guide=a.learn_in_guide,
                                act=a.act, updates=a.updates, log=log)
        extra['hist'] = hist
    else:
        raise ValueError(a.method)
    sel_time = time.time() - t0
    log(f'selected {len(bands)} bands in {sel_time:.0f}s: {bands.tolist()}')

    if a.no_eval:
        m = None
    elif a.method in ('mhdrl', 'all'):
        m, _, _ = train_eval(cube, n_cls, bands, tr, ytr, te, yte, a.seed, a.cls, a.cls_patch, a.cls_pca,
                             a.cls_epochs, ens=a.cls_ens)
    else:
        m = own_eval(a.method, cube, np.asarray(bands), tr, ytr, te, yte, n_cls, a.seed)
    if m:
        log(f'RESULT OA={m["oa"] * 100:.2f} AA={m["aa"] * 100:.2f} Kappa={m["kappa"] * 100:.2f}')
    json.dump(dict(args=vars(a), bands=np.asarray(bands).tolist(), select_time_s=sel_time, metrics=m, **extra),
              open(out, 'w'))


if __name__ == '__main__':
    main()
