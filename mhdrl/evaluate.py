"""对比方法的分类器（论文 4.3 节）：逐像素输入的方法（BS-Nets、MR-SVM、ABCNN、DDCNN）用 RBF-SVM，
空间窗口输入的方法（TWCNN、SICNN、DRLBS）用 15×15 窗口的 2D CNN。MH-DRL 用内嵌的 PE-Net（论文 4.4 节）。"""
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .classify import metrics, predict, dihedral
from .data import PatchSampler
from .masked import CNN2D

PIXEL = {'bsnets', 'mrsvm', 'abcnn', 'ddcnn', 'random', 'uniform'}
SPATIAL = {'twcnn', 'sicnn', 'drlbs'}


def svm_eval(cube, bands, tr, ytr, te, yte, n_cls, seed):
    x = cube[:, :, bands]
    xtr, xte = x[tr[:, 0], tr[:, 1]], x[te[:, 0], te[:, 1]]
    gs = GridSearchCV(SVC(kernel='rbf'), {'C': [1, 10, 100, 1000], 'gamma': ['scale', 0.01, 0.1, 1]}, cv=3, n_jobs=4)
    gs.fit(xtr, ytr)
    return metrics(yte, gs.predict(xte), n_cls)


def cnn2d_eval(cube, bands, tr, ytr, te, yte, n_cls, seed, patch=15, epochs=200, bs=64, lr=1e-3, device='cuda'):
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler = PatchSampler(cube[:, :, bands], patch, device)
    net = CNN2D(len(bands), n_cls).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    xtr, y = sampler(tr), torch.as_tensor(ytr, device=device)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(net(dihedral(xtr[idx], np.random.randint(8)))[0], y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return metrics(yte, predict(net, sampler, te, tta=False).argmax(1), n_cls)


def own_eval(method, cube, bands, tr, ytr, te, yte, n_cls, seed):
    if method in PIXEL:
        return svm_eval(cube, bands, tr, ytr, te, yte, n_cls, seed)
    if method in SPATIAL:
        return cnn2d_eval(cube, bands, tr, ytr, te, yte, n_cls, seed)
    raise ValueError(method)
