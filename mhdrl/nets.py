"""分类网络：PE-Net（论文 Table 1，HybridSN 式 3D-2D CNN）与 SSFTT。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PENet(nn.Module):
    """论文 Table 1：3 层 3D 卷积 (7×3×3, 5×3×3, 3×3×3; 8/16/32) + 1 层 2D 卷积 (3×3, 64) + FC 256/128 + 分类层。
    输入 [B, D, p, p]，未选波段置零；mask_aware 时把 0/1 波段掩码嵌入加到第一个全连接层上。
    forward 返回 (logits, F_l)，F_l 是 dense_2 的 128 维特征，即 MH-DRL 的状态。"""

    def __init__(self, n_bands, n_classes, patch=25, mask_aware=True, norm='bn'):
        super().__init__()

        def nrm(c, d3):
            if norm == 'gn':
                return nn.GroupNorm(min(8, c), c)
            return nn.BatchNorm3d(c) if d3 else nn.BatchNorm2d(c)

        # 波段数少于 16 时光谱方向做 same 填充（三层 3D 卷积不填充会吃掉 12 个波段）
        spad = n_bands < 16

        def c3(i, o, k):
            pad = (k[0] // 2, 0, 0) if spad else 0
            return nn.Sequential(nn.Conv3d(i, o, k, padding=pad), nrm(o, True), nn.ReLU(inplace=True))

        self.conv3d = nn.Sequential(c3(1, 8, (7, 3, 3)), c3(8, 16, (5, 3, 3)), c3(16, 32, (3, 3, 3)))
        d_out = n_bands if spad else n_bands - 12
        self.conv2d = nn.Sequential(nn.Conv2d(32 * d_out, 64, 3), nrm(64, False), nn.ReLU(inplace=True))
        self.dense1 = nn.Linear(64 * (patch - 8) ** 2, 256)
        self.mask_emb = nn.Linear(n_bands, 256, bias=False) if mask_aware else None
        self.dense2 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.4)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x, mask=None):
        b = x.shape[0]
        h = self.conv3d(x[:, None])
        h = self.conv2d(h.flatten(1, 2)).flatten(1)
        h = self.dense1(h)
        if self.mask_emb is not None:
            m = torch.ones(b, x.shape[1], device=x.device) if mask is None else mask.expand(b, -1)
            h = h + self.mask_emb(m)
        h = self.drop(F.relu(h))
        f = F.relu(self.dense2(h))
        return self.out(self.drop(f)), f


class _Attn(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(drop))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2) * self.scale).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, d))


class SSFTT(nn.Module):
    """Sun et al., Spectral-Spatial Feature Tokenization Transformer, TGRS 2022（按官方实现改写，波段数通用）。
    输入 [B, L, p, p]（L 为 PCA 后通道数）。"""

    def __init__(self, n_bands, n_classes, n_tokens=4, dim=64, depth=1, heads=8, mlp=8, drop=0.1):
        super().__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(1, 8, 3), nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.conv2d = nn.Sequential(nn.Conv2d(8 * (n_bands - 2), 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.wA = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, n_tokens, 64)))
        self.wV = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 64, dim)))
        self.pos = nn.Parameter(torch.randn(1, n_tokens + 1, dim) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(dim), _Attn(dim, heads, drop), nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp, dim), nn.Dropout(drop))]))
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x, mask=None):
        h = self.conv3d(x[:, None]).flatten(1, 2)
        h = self.conv2d(h).flatten(2).transpose(1, 2)            # [B, HW, 64]
        a = (h @ self.wA.transpose(1, 2)).transpose(1, 2).softmax(-1)  # [B, T, HW]
        t = a @ (h @ self.wV)                                      # [B, T, dim]
        z = torch.cat([self.cls.expand(len(x), -1, -1), t], 1) + self.pos
        z = self.drop(z)
        for n1, at, n2, ff in self.blocks:
            z = z + at(n1(z))
            z = z + ff(n2(z))
        f = z[:, 0]
        return self.head(f), f
