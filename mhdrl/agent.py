"""MH-DRL：多智能体 DQN + 混合教师引导（论文 3.1、3.3、3.4 节与 Algorithm I 的 Stage II/III）。"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDQN(nn.Module):
    """D 个相互独立的全连接 DQN（每个波段一个智能体），参数堆叠成张量一次前向。
    输入共享状态 s [B, S]，输出 Q [B, D, 2]（0=不选，1=选）。"""

    def __init__(self, n_agents, s_dim, hidden=64):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_agents, s_dim, hidden) * (1 / s_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n_agents, hidden))
        self.w2 = nn.Parameter(torch.randn(n_agents, hidden, 2) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(n_agents, 2))

    def forward(self, s):
        h = F.relu(torch.einsum('bs,ash->bah', s, self.w1) + self.b1)
        return torch.einsum('bah,aho->bao', h, self.w2) + self.b2


def corr_matrix(cube, max_px=50000, seed=0):
    x = cube.reshape(-1, cube.shape[2])
    x = x[np.random.RandomState(seed).permutation(len(x))[:max_px]]
    return np.abs(np.corrcoef(x.T))


def mean_corr(C, mask):
    idx = np.where(mask > 0)[0]
    if len(idx) < 2:
        return 0.0
    sub = C[np.ix_(idx, idx)]
    return float((sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))


def teacher_advice(a_prev, a_cur, score):
    """论文式 (12)：参与波段 B_p = 上一步选中的；高置信 B_h = 两步都选；低置信 B_l = 上一步选、这一步没选。
    k = ⌊(m+n)/2⌋，教师在 B_p 里按得分取前 k 个，落在 B_l 里的波段改为选择。"""
    bp = np.where(a_prev > 0)[0]
    m = int(((a_prev > 0) & (a_cur > 0)).sum())
    low = (a_prev > 0) & (a_cur == 0)
    n = int(low.sum())
    k = (m + n) // 2
    a = a_cur.copy()
    if k == 0 or n == 0:
        return a, 0
    top = bp[np.argsort(-score[bp])[:k]]
    flip = top[low[top]]
    a[flip] = 1
    return a, len(flip)


def run_mhdrl(ev, C, nd, teachers, seed, steps=900, guide=300, alpha=0.1, reward='soft',
              centered=True, eps_start=0.1, eps_end=0.02, lr=1e-3, gamma=0.9, batch=16,
              target_every=10, memory=100000, hidden=64, learn_in_guide=False, act='topk', updates=4,
              device='cuda', log=print):
    """ev: MaskedEvaluator；C: |相关系数| 矩阵；nd: 期望波段数；teachers: [(名字, 得分)]，按顺序各引导 guide/len 步。
    act='argmax'：每个智能体各自取 Q 大的动作；
    act='topk'：按各智能体 Q(选)-Q(不选) 排名取前 nd 个作为联合动作，探索时随机换进换出 ε·nd 个波段。"""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    D = ev.D
    l = guide // max(1, len(teachers))

    def observe(a):
        acc, soft, f = ev.evaluate(a)
        pc = {'soft': soft, 'acc': acc, 'bsoft': ev.last_bsoft}[reward]
        ns = a.sum()
        R = (pc - alpha * mean_corr(C, a)) * (1 - ((ns - nd) / nd) ** 2)   # 式 (5)(8)：β·P_b 化简后的形式
        s = torch.cat([f, torch.as_tensor(a, dtype=torch.float32, device=device)])
        return s, R, acc

    a_prev = (rng.rand(D) < nd / D).astype(np.float32)
    s, R, acc = observe(a_prev)
    q_net = MultiDQN(D, s.numel(), hidden).to(device)
    tgt = copy.deepcopy(q_net)
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    buf_s, buf_a, buf_r, buf_s2 = [], [], [], []
    r_mean, r_var = R, 1e-4
    best = (-1e9, None)
    learn_steps = 0
    hist = []
    for t in range(steps):
        with torch.no_grad():
            q = q_net(s[None])[0]
        eps = eps_start + (eps_end - eps_start) * t / max(1, steps - 1)
        if act == 'topk':
            adv = (q[:, 1] - q[:, 0]).cpu().numpy()
            a = np.zeros(D, np.float32)
            a[np.argsort(-adv)[:nd]] = 1
            n_sw = max(1, int(round(eps * nd)))
            a[rng.choice(np.where(a > 0)[0], n_sw, replace=False)] = 0
            a[rng.choice(np.where(a == 0)[0], n_sw, replace=False)] = 1
        else:
            a = q.argmax(1).cpu().numpy().astype(np.float32)
            rnd = rng.rand(D) < eps
            a[rnd] = rng.randint(2, size=rnd.sum())
        n_adv, who = 0, '-'
        if t < guide and teachers:
            who, sc = teachers[min(t // l, len(teachers) - 1)]
            a_adv, n_adv = teacher_advice(a_prev, a, sc)
            if act == 'topk' and n_adv > 0:
                # 教师改选的波段留下，再从其余已选波段里去掉 Q 优势最低的若干个，保持正好 nd 个
                adv = (q[:, 1] - q[:, 0]).cpu().numpy()
                flipped = (a_adv > 0) & (a == 0)
                keep = np.where((a_adv > 0) & ~flipped)[0]
                n_drop = int(a_adv.sum()) - nd
                if n_drop > 0:
                    a_adv[keep[np.argsort(adv[keep])[:n_drop]]] = 0
            a = a_adv
        if a.sum() == 0:
            a[rng.randint(D)] = 1
        s2, R, acc = observe(a)
        # 式 (9)：奖励只平均分给选择的智能体；centered=True 时先减去奖励滑动均值再除以滑动标准差
        rr = (R - r_mean) / (r_var ** 0.5 + 1e-6) if centered else R
        r_vec = a * rr / a.sum() * nd
        r_mean = 0.95 * r_mean + 0.05 * R
        r_var = 0.95 * r_var + 0.05 * (R - r_mean) ** 2
        buf_s.append(s); buf_a.append(torch.as_tensor(a, device=device).long())
        buf_r.append(torch.as_tensor(r_vec, dtype=torch.float32, device=device)); buf_s2.append(s2)
        if len(buf_s) > memory:
            buf_s.pop(0); buf_a.pop(0); buf_r.pop(0); buf_s2.pop(0)
        # 论文 Stage III：教师引导结束后才更新 DQN；learn_in_guide=True 时引导期间也从教师修正过的经验里学
        for _ in range(updates if ((t >= guide or learn_in_guide) and len(buf_s) >= batch) else 0):
            idx = rng.randint(len(buf_s), size=batch)
            bs_ = torch.stack([buf_s[i] for i in idx]); ba = torch.stack([buf_a[i] for i in idx])
            br = torch.stack([buf_r[i] for i in idx]); bs2 = torch.stack([buf_s2[i] for i in idx])
            qe = q_net(bs_).gather(2, ba[:, :, None]).squeeze(2)
            with torch.no_grad():
                qt = br + gamma * tgt(bs2).max(2)[0]
            loss = F.mse_loss(qe, qt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            learn_steps += 1
            if learn_steps % target_every == 0:
                tgt.load_state_dict(q_net.state_dict())
        if abs(a.sum() - nd) <= 0.1 * nd and R > best[0]:
            best = (R, s2.clone())
        hist.append(dict(t=t, R=float(R), acc=float(acc), ns=int(a.sum()), adv=n_adv, teacher=who))
        if t % 50 == 0 or t == steps - 1:
            log(f'  step {t:4d} [{who}] ns={int(a.sum()):3d} acc={acc:.4f} R={R:.4f} advised={n_adv}')
        s, a_prev = s2, a
    # 输出：按 Q(选)-Q(不选) 取前 nd 个；在最后状态和最佳状态下各取一组，用评估器挑更好的那组
    cands = []
    with torch.no_grad():
        for st in [s] + ([best[1]] if best[1] is not None else []):
            adv = (lambda q: (q[:, 1] - q[:, 0]).cpu().numpy())(q_net(st[None])[0])
            bands = np.sort(np.argsort(-adv)[:nd])
            mask = np.zeros(D, np.float32); mask[bands] = 1
            _, Rc, accc = observe(mask)
            cands.append((Rc, accc, bands))
    Rc, accc, bands = max(cands, key=lambda c: c[0])
    log(f'  final: R={Rc:.4f} OOF acc={accc:.4f} (candidates {[round(c[0], 4) for c in cands]})')
    return bands, hist
