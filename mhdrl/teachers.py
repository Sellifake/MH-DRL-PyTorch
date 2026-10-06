"""三个教师（也是对比算法）：BS-Nets（过滤式）、SICNN（包裹式）、TWCNN（嵌入式）。
每个函数返回长度 D 的波段得分，越大越好；教师在 MH-DRL 里对参与波段集合按得分取前 k 个。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classify import dihedral
from .data import PatchSampler
from .masked import CNN2D, MaskedEvaluator


# ---------------- BS-Nets (Cai et al., TGRS 2020, BSNet-Conv) ----------------
class BSNetConv(nn.Module):
    """波段注意力模块 BAM 产生逐波段权重，加权后的块经卷积自编码器重建原块；L1 让权重稀疏。"""

    def __init__(self, D):
        super().__init__()
        self.bam = nn.Sequential(nn.Conv2d(D, 64, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, D), nn.Sigmoid())

        def cbr(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.rec = nn.Sequential(*cbr(D, 128), *cbr(128, 64), *cbr(64, 128), nn.Conv2d(128, D, 1))

    def forward(self, x):
        w = self.bam(x)
        return self.rec(x * w[:, :, None, None]), w


def bsnets_scores(cube, seed, patch=7, iters=3000, bs=64, lr=1e-3, l1=0.01, device='cuda'):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    H, W, D = cube.shape
    sampler = PatchSampler(cube, patch, device)
    net = BSNetConv(D).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(iters):
        c = np.stack([rng.randint(H, size=bs), rng.randint(W, size=bs)], 1)
        x = sampler(c)
        y, w = net(x)
        loss = F.mse_loss(y, x) + l1 * w.abs().sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        c = np.stack([rng.randint(H, size=4096), rng.randint(W, size=4096)], 1)
        ws = torch.cat([net(sampler(c[i:i + 512]))[1] for i in range(0, 4096, 512)])
    return ws.mean(0).cpu().numpy()


# ---------------- TWCNN (Feng et al., IGARSS 2019) ----------------
class TWNet(nn.Module):
    """第一层是逐波段深度卷积权重，前向时三值化 {-1,0,1}（阈值 th，直通估计回传梯度），权重为 0 的波段即未选。"""

    def __init__(self, D, n_cls, th=0.5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(D).uniform_(0.48, 0.5))
        self.th = th
        self.body = CNN2D(D, n_cls)

    def wq(self):
        t = torch.where(self.w > self.th, 1.0, torch.where(self.w < -self.th, -1.0, 0.0))
        return self.w + (t - self.w).detach()

    def forward(self, x):
        return self.body(x * self.wq()[None, :, None, None])


def twcnn_scores(cube, tr, ytr, n_cls, k, seed, patch=15, iters=3000, bs=64, lr=1e-3, lam=0.01,
                 band_loss=True, device='cuda'):
    torch.manual_seed(seed)
    D = cube.shape[2]
    x = PatchSampler(cube, patch, device)(tr)
    y = torch.as_tensor(ytr, device=device)
    net = TWNet(D, n_cls).to(device)
    opt = torch.optim.Adam([{'params': net.body.parameters()}, {'params': [net.w], 'lr': lr * 10}], lr=lr)
    for _ in range(iters):
        b = torch.randint(len(tr), (bs,), device=device)
        loss = F.cross_entropy(net(dihedral(x[b], np.random.randint(8)))[0], y[b])
        if band_loss:
            loss = loss + lam * 0.5 * (net.wq().abs().sum() - k) ** 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            net.w.clamp_(-1, 1)
    return net.w.detach().abs().cpu().numpy()


# ---------------- SICNN（包裹式：2D CNN 评估候选子集） ----------------
def sicnn_scores(cube, tr, ytr, n_cls, seed, patch=11, folds=4, iters=2000, frac=0.05, log=print, device='cuda'):
    """随机波段组合训练的 2D CNN 当评估器，逐轮删掉“删了掉分最少”的若干波段（每轮删剩余数的 frac），
    越晚被删的波段得分越高。"""
    ev = MaskedEvaluator(cube, tr, ytr, n_cls, arch='cnn2d', patch=patch, folds=folds, iters=iters,
                         seed=seed, device=device, log=log)
    D = cube.shape[2]
    alive = list(range(D))
    score = np.zeros(D)
    rank = 0
    while alive:
        if len(alive) == 1:
            score[alive[0]] = rank
            break
        base = np.zeros(D)
        base[alive] = 1
        masks = np.repeat(base[None], len(alive), 0)
        masks[np.arange(len(alive)), alive] = 0
        s = ev.evaluate_many(masks)              # 去掉第 i 个波段后的得分，越高说明该波段越可有可无
        n_rm = max(1, int(len(alive) * frac))
        order = np.argsort(-s)[:n_rm]
        for j, o in enumerate(order):
            score[alive[o]] = rank + j
        rank += n_rm
        alive = [b for i, b in enumerate(alive) if i not in set(order.tolist())]
    return score
