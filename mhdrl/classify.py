"""给定波段子集，在同一划分上训练分类器并在测试集上算 OA/AA/Kappa。所有波段选择方法共用这一套评测。"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .data import PatchSampler, pca_cube
from .nets import PENet, SSFTT


def dihedral(x, k):
    """8 种翻转/旋转之一，x: [B, C, p, p]。"""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, (-2, -1))


def metrics(y, pred, n_cls):
    cm = confusion_matrix(y, pred, labels=np.arange(n_cls))
    per = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return dict(oa=float(cm.diagonal().sum() / cm.sum()), aa=float(per.mean()),
                kappa=float(cohen_kappa_score(y, pred)), per_class=per.tolist())


def build(arch, n_bands, n_cls, patch):
    if arch == 'ssftt':
        return SSFTT(n_bands, n_cls)
    if arch == 'penet':
        return PENet(n_bands, n_cls, patch, mask_aware=False)
    raise ValueError(arch)


@torch.no_grad()
@torch.autocast('cuda', torch.bfloat16)
def predict(model, sampler, coords, tta=True, bs=128):
    model.eval()
    out = []
    for i in range(0, len(coords), bs):
        x = sampler(coords[i:i + bs])
        p = sum(F.softmax(model(dihedral(x, k))[0].float(), 1) for k in (range(8) if tta else [0]))
        out.append(p.cpu())
    return torch.cat(out).numpy()


def train_eval(cube, n_cls, bands, tr, ytr, te, yte, seed, arch='ssftt', patch=13, pca=30,
               epochs=200, bs=64, lr=1e-3, wd=1e-4, smooth=0.1, tta=True, ens=1, device='cuda'):
    """ens>1：ens 个不同初始化的分类器，softmax 取平均。"""
    t0 = time.time()
    sub = cube[:, :, np.asarray(bands)]
    if pca and sub.shape[2] > pca:
        sub = pca_cube(sub, pca)
    sampler = PatchSampler(sub, patch, device)
    prob, singles, models = 0, [], []
    for e in range(ens):
        model = _fit(sampler, sub.shape[2], n_cls, tr, ytr, seed * 1000 + e, arch, patch, epochs, bs, lr, wd, smooth,
                     device)
        p = predict(model, sampler, te, tta)
        singles.append(metrics(yte, p.argmax(1), n_cls)['oa'])
        prob = prob + p
        models.append(model)
    m = metrics(yte, prob.argmax(1), n_cls)
    m['single_oa'] = singles
    m['time_s'] = time.time() - t0
    return m, models, sampler


def _fit(sampler, n_bands, n_cls, tr, ytr, seed, arch, patch, epochs, bs, lr, wd, smooth, device, balanced=False):
    """balanced=True：按类别频率的平方根倒数加权采样，小类多采（只改采样，不改损失）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(arch, n_bands, n_cls, patch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * max(1, int(np.ceil(len(tr) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    ytr_t = torch.as_tensor(ytr, device=device)
    xtr = sampler(tr)
    if balanced:
        w = 1.0 / np.sqrt(np.bincount(ytr, minlength=n_cls)[ytr])
        w_t = torch.as_tensor(w / w.sum(), dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        perm = torch.multinomial(w_t, len(tr), replacement=True) if balanced else torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            x = dihedral(xtr[idx], np.random.randint(8))
            with torch.autocast('cuda', torch.bfloat16):
                loss = F.cross_entropy(model(x)[0].float(), ytr_t[idx], label_smoothing=smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model
