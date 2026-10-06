"""其余对比算法（都返回长度 D 的波段得分，越大越好；或直接返回波段下标）。
ABCNN 依据官方实现（ESA-PhiLab/hypernet, hsi_attention/Model2）；MR-SVM、DRLBS、DDCNN 依据各自原论文实现。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

from .masked import MaskedEvaluator


def center_spectra(cube, coords):
    return cube[coords[:, 0], coords[:, 1]]


# ---------------- ABCNN (Lorenzo et al., IEEE Access 2020) ----------------
class _Att(nn.Module):
    def __init__(self, ch, length, n_cls):
        super().__init__()
        self.soft = nn.Sequential(nn.Conv1d(ch, 1, 1), nn.ReLU(), nn.Softmax(dim=2))
        self.conf = nn.Sequential(nn.Linear(length, 1), nn.Tanh())
        self.att = nn.Linear(length, n_cls)

    def forward(self, z):
        h = self.soft(z)                                   # [B, 1, L] 光谱位置上的注意力热图
        c = (h * z).mean(1)                                # [B, L]
        return self.att(c) * self.conf(c), h[:, 0]


class ABCNN(nn.Module):
    def __init__(self, D, n_cls):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(o), nn.MaxPool1d(2))

        self.b1, self.b2 = blk(1, 96), blk(96, 54)
        self.a1, self.a2 = _Att(96, D // 2, n_cls), _Att(54, D // 4, n_cls)
        flat = 54 * (D // 4)
        self.cls = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, n_cls))
        self.cconf = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

    def forward(self, x):
        z1 = self.b1(x[:, None])
        p1, h1 = self.a1(z1)
        z2 = self.b2(z1)
        p2, h2 = self.a2(z2)
        flat = z2.flatten(1)
        return self.cls(flat) * self.cconf(flat) + p1 + p2, (h1, h2)


def abcnn_scores(cube, tr, ytr, n_cls, seed, iters=4000, bs=64, lr=1e-4, device='cuda'):
    torch.manual_seed(seed)
    D = cube.shape[2]
    x = torch.as_tensor(center_spectra(cube, tr), device=device)
    y = torch.as_tensor(ytr, device=device)
    net = ABCNN(D, n_cls).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(iters):
        b = torch.randint(len(x), (bs,), device=device)
        loss = F.binary_cross_entropy(net(x[b])[0].softmax(1), F.one_hot(y[b], n_cls).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        _, (h1, h2) = net(x)
        # 官方做法：热图插值回 D 个波段，先按类平均再跨类平均
        hm = [F.interpolate(h[:, None], size=D, mode='linear', align_corners=True)[:, 0] for h in (h1, h2)]
        hm = (hm[0] + hm[1]) / 2
        per_cls = torch.stack([hm[y == c].mean(0) for c in range(n_cls) if (y == c).any()])
    return per_cls.mean(0).cpu().numpy()


# ---------------- MR-SVM（递归 SVM 特征消除，Zhang et al. 2009） ----------------
def mrsvm_scores(cube, tr, ytr, seed):
    x = center_spectra(cube, tr)
    rfe = RFE(LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=seed), n_features_to_select=1, step=1)
    rfe.fit(x, ytr)
    return -rfe.ranking_.astype(float)


# ---------------- DRLBS（Mou et al., TGRS 2022） ----------------
def drlbs_bands(cube, nd, seed, episodes=300, lr=1e-3, gamma=0.99, eps=0.9, batch=100, memory=5000, device='cuda'):
    """单智能体 DQN 逐个往子集里加波段（动作空间 = D 个波段），奖励 = 子集“平均信息熵 − 平均相关系数”的增量，
    无监督；训练完按策略贪心走一遍得到 nd 个波段。"""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    D = cube.shape[2]
    px = cube.reshape(-1, D)[rng.permutation(cube.shape[0] * cube.shape[1])[:50000]]
    ie = np.zeros(D)
    for i in range(D):
        h, _ = np.histogram(px[:, i], bins=256)
        p = h[h > 0] / h.sum()
        ie[i] = -(p * np.log(p)).sum()
    ie = (ie - ie.min()) / (ie.max() - ie.min() + 1e-12)
    C = np.abs(np.corrcoef(px.T))

    def value(mask):
        idx = np.where(mask > 0)[0]
        if len(idx) == 0:
            return 0.0
        corr = 0.0 if len(idx) < 2 else (C[np.ix_(idx, idx)].sum() - len(idx)) / (len(idx) * (len(idx) - 1))
        return ie[idx].mean() - corr

    q = nn.Sequential(nn.Linear(D, 400), nn.ReLU(), nn.Linear(400, D)).to(device)
    tgt = nn.Sequential(nn.Linear(D, 400), nn.ReLU(), nn.Linear(400, D)).to(device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    mem = []

    def act(s, greedy):
        free = np.where(s == 0)[0]
        if not greedy and rng.rand() > eps:
            return rng.choice(free)
        with torch.no_grad():
            v = q(torch.as_tensor(s, dtype=torch.float32, device=device)[None])[0].cpu().numpy()
        v[s > 0] = -1e9
        return int(v.argmax())

    for ep in range(episodes):
        s = np.zeros(D, np.float32)
        for k in range(nd):
            a = act(s, False)
            s2 = s.copy(); s2[a] = 1
            r = (value(s2) - value(s)) * 10
            mem.append((s, a, r, s2, float(k == nd - 1)))
            if len(mem) > memory:
                mem.pop(0)
            s = s2
            if len(mem) >= batch:
                bt = [mem[i] for i in rng.randint(len(mem), size=batch)]
                bs_ = torch.as_tensor(np.stack([b[0] for b in bt]), device=device)
                ba = torch.as_tensor([b[1] for b in bt], device=device)
                br = torch.as_tensor([b[2] for b in bt], dtype=torch.float32, device=device)
                bs2 = torch.as_tensor(np.stack([b[3] for b in bt]), device=device)
                bd = torch.as_tensor([b[4] for b in bt], device=device)
                with torch.no_grad():
                    qn = tgt(bs2).masked_fill(bs2 > 0, -1e9).max(1)[0]
                    yt = br + gamma * qn * (1 - bd)
                loss = F.mse_loss(q(bs_).gather(1, ba[:, None])[:, 0], yt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if ep % 10 == 0:
            tgt.load_state_dict(q.state_dict())
    s = np.zeros(D, np.float32)
    for _ in range(nd):
        s[act(s, True)] = 1
    return np.where(s > 0)[0]


# ---------------- DDCNN（Zhan et al., GRSL 2017） ----------------
def ddcnn_bands(cube, tr, ytr, n_cls, nd, seed, n_cand=60, log=print, device='cuda'):
    """先用全波段（随机波段组合）训好一个逐像素 CNN；再按波段间的距离密度（密度峰：ρ 局部密度、δ 到更高密度波段的距离）
    生成一批候选子集，用训好的 CNN 不重训直接评估，取最好的一组。"""
    rng = np.random.RandomState(seed)
    D = cube.shape[2]
    px = cube.reshape(-1, D)[rng.permutation(cube.shape[0] * cube.shape[1])[:20000]].T   # [D, P]
    dist = cdist(px, px) / np.sqrt(px.shape[1])          # 波段间逐像素差的均方根
    dc = np.sort(dist[np.triu_indices(D, 1)])[int(0.02 * D * (D - 1) / 2)]
    rho = np.exp(-(dist / dc) ** 2).sum(1) - 1
    delta = np.zeros(D)
    for i in range(D):
        higher = np.where(rho > rho[i])[0]
        delta[i] = dist[i, higher].min() if len(higher) else dist[i].max()
    rho_n = (rho - rho.min()) / (np.ptp(rho) + 1e-12)
    delta_n = (delta - delta.min()) / (np.ptp(delta) + 1e-12)
    ev = MaskedEvaluator(cube, tr, ytr, n_cls, arch='cnn2d', patch=1, folds=4, iters=2000, seed=seed, log=log)
    cands = []
    for a in np.linspace(0.25, 4, n_cand // 2):
        g = rho_n ** a * delta_n
        cands.append(np.argsort(-g)[:nd])
        noisy = g * np.exp(0.3 * rng.randn(D))
        cands.append(np.argsort(-noisy)[:nd])
    masks = np.zeros((len(cands), D), np.float32)
    for i, c in enumerate(cands):
        masks[i, c] = 1
    s = ev.evaluate_many(masks)
    return np.sort(cands[int(s.argmax())])
