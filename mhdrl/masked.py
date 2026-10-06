"""随机波段组合训练的评估网络（论文 Stage I）。

训练时每个 batch 随机保留 U[1, D] 个波段、其余置零（Algorithm I 的 random band combination），
训练好后对任意波段子集直接前向评估，不再微调。
训练集按类分层切成 K 折，每折一个网络，评估时每个训练样本只由没见过它的那一折网络打分
（out-of-fold），这样奖励不会被训练集上 100% 的准确率淹没，也不碰测试集。
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classify import dihedral
from .data import PatchSampler
from .nets import PENet


class CNN2D(nn.Module):
    """轻量 2D CNN（SICNN 教师的评估网络）：1×1 光谱卷积 + 两层 3×3 + GAP。"""

    def __init__(self, n_bands, n_classes, width=128):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_bands, width, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mask_emb = nn.Linear(n_bands, width, bias=False)
        self.out = nn.Linear(width, n_classes)

    def forward(self, x, mask=None):
        m = torch.ones(len(x), x.shape[1], device=x.device) if mask is None else mask.expand(len(x), -1)
        f = F.relu(self.f(x) + self.mask_emb(m))
        return self.out(f), f


def stratified_folds(y, k, seed):
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


class MaskedEvaluator:
    def __init__(self, cube, tr, ytr, n_cls, arch='penet', patch=25, folds=4, iters=3000, bs=64,
                 lr=1e-3, seed=0, device='cuda', log=print, cache=None, rescale=False, norm='bn',
                 bn_batch_eval=True):
        self.D, self.n_cls, self.device = cube.shape[2], n_cls, device
        self.rescale = rescale      # True：置零后按 D/保留数 放大（同 inverted dropout），保持输入幅度稳定
        # bn_batch_eval：评估时 BN 用当批统计量。训练时一个 batch 只有一种掩码，BN 用的就是该掩码下的统计量；
        # 评估时每次也是同一掩码下的一批样本，两者一致
        self.bn_batch_eval = bn_batch_eval
        self.x = PatchSampler(cube, patch, device)(tr)          # [N, D, p, p]
        self.y = torch.as_tensor(ytr, device=device)
        self.fold = torch.as_tensor(stratified_folds(ytr, folds, seed), device=device)
        self.nets = []
        for k in range(folds):
            torch.manual_seed(seed * 100 + k)
            net = (PENet(self.D, n_cls, patch, norm=norm) if arch == 'penet' else CNN2D(self.D, n_cls)).to(device)
            ck = cache and f'{cache}_fold{k}.pt'
            if ck and os.path.exists(ck):
                net.load_state_dict(torch.load(ck, map_location=device))
            else:
                self._train(net, torch.where(self.fold != k)[0], iters, bs, lr)
                if ck:
                    os.makedirs(os.path.dirname(ck), exist_ok=True)
                    torch.save(net.state_dict(), ck)
            self._set_eval(net)
            self.nets.append(net)
            log(f'  evaluator fold {k}: full-band OOF acc {self._fold_acc(k):.4f}')

    def _train(self, net, idx, iters, bs, lr):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
        net.train()
        for _ in range(iters):
            b = idx[torch.randint(len(idx), (bs,), device=self.device)]
            n_keep = np.random.randint(1, self.D + 1)
            mask = torch.zeros(self.D, device=self.device)
            mask[torch.randperm(self.D, device=self.device)[:n_keep]] = 1
            x = dihedral(self.x[b], np.random.randint(8)) * self._scale(mask)[None, :, None, None]
            with torch.autocast('cuda', torch.bfloat16):
                loss = F.cross_entropy(net(x, mask[None])[0].float(), self.y[b], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    def _set_eval(self, net):
        net.eval()
        if self.bn_batch_eval:
            for m in net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.track_running_stats = False
                    m.running_mean = m.running_var = None

    def _scale(self, mask):
        return mask * (self.D / mask.sum(-1, keepdim=True).clamp(min=1)) if self.rescale else mask

    @torch.no_grad()
    @torch.autocast('cuda', torch.bfloat16)
    def _fold_acc(self, k):
        idx = torch.where(self.fold == k)[0]
        return (self.nets[k](self.x[idx])[0].argmax(1) == self.y[idx]).float().mean().item()

    @torch.no_grad()
    @torch.autocast('cuda', torch.bfloat16)
    def evaluate(self, mask):
        """mask: [D] 0/1。返回 (OOF 准确率, OOF 真类平均概率, 状态特征 = 全部训练样本 F_l 的均值)。"""
        mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        correct, pt, feats, ys = 0.0, [], [], []
        for k, net in enumerate(self.nets):
            idx = torch.where(self.fold == k)[0]
            logits, f = net(self.x[idx] * self._scale(mask)[None, :, None, None], mask[None])
            p, f = logits.float().softmax(1), f.float()
            correct += (p.argmax(1) == self.y[idx]).sum().item()
            pt.append(p.gather(1, self.y[idx, None])[:, 0])
            ys.append(self.y[idx])
            feats.append(f)
        pt, ys = torch.cat(pt), torch.cat(ys)
        # 类别均衡的真类平均概率：先类内平均再跨类平均
        self.last_bsoft = torch.zeros(self.n_cls, device=self.device).index_add_(0, ys, pt).div_(
            torch.bincount(ys, minlength=self.n_cls).clamp(min=1)).mean().item()
        n = len(self.y)
        return correct / n, pt.sum().item() / n, torch.cat(feats).mean(0)

    @torch.no_grad()
    @torch.autocast('cuda', torch.bfloat16)
    def evaluate_many(self, masks, chunk=16):
        """一次评估多个掩码 [M, D]，返回 OOF 真类平均概率 [M]（用于包裹式逐步淘汰）。"""
        masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        if self.bn_batch_eval:
            chunk = 1        # 当批统计量要求一批里只有一种掩码
        out = torch.zeros(len(masks), device=self.device)
        for k, net in enumerate(self.nets):
            idx = torch.where(self.fold == k)[0]
            xk, yk = self.x[idx], self.y[idx]
            for i in range(0, len(masks), chunk):
                m = masks[i:i + chunk]
                xb = (xk[None] * self._scale(m)[:, None, :, None, None]).flatten(0, 1)
                mb = m.repeat_interleave(len(idx), 0)
                p = net(xb, mb)[0].float().softmax(1).view(len(m), len(idx), -1)
                out[i:i + chunk] += p.gather(2, yk[None, :, None].expand(len(m), -1, 1)).squeeze(2).sum(1)
        return (out / len(self.y)).cpu().numpy()
