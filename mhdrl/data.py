"""数据读取、按类分层划分、GPU 上取邻域块。"""
import glob
import math
import os

import numpy as np
import scipy.io as sio
import torch

DATA_ROOT = os.environ.get('HSI_DATA', 'data')

# ratio / nbands 取自论文 4.1 与 4.3 节；salinas、botswana 为论文外的补充数据集
DATASETS = {
    'indian': dict(file='Indian_pines_corrected.mat', key='indian_pines_corrected',
                   gt='Indian_pines_gt.mat', gtkey='indian_pines_gt', ratio=0.05, nbands=60),
    'paviau': dict(file='PaviaU.mat', key='paviaU',
                   gt='PaviaU_gt.mat', gtkey='paviaU_gt', ratio=0.03, nbands=30),
    'houston': dict(file='Houstondata.mat', key='Houstondata',
                    gt='Houstonlabel.mat', gtkey='Houstonlabel', ratio=0.05, nbands=40),
    'salinas': dict(file='Salinas_corrected.mat', key='salinas_corrected',
                    gt='Salinas_gt.mat', gtkey='salinas_gt', ratio=0.01, nbands=30),
    'botswana': dict(file='Botswana.mat', key='Botswana',
                     gt='Botswana_gt.mat', gtkey='Botswana_gt', ratio=0.05, nbands=30),
}


def find(fname):
    """在 DATA_ROOT 下按文件名递归查找（兼容 Hyperspectral_Image_Datasets_Collection 的目录结构或任意摆放）。"""
    hits = sorted(glob.glob(os.path.join(DATA_ROOT, '**', fname), recursive=True))
    if not hits:
        raise FileNotFoundError(f'{fname} not found under {os.path.abspath(DATA_ROOT)} (set HSI_DATA)')
    return hits[0]


def load(name):
    """返回 (cube[H,W,D] float32 逐波段 z-score, gt[H,W] int, 类别数)。"""
    c = DATASETS[name]
    x = sio.loadmat(find(c['file']))[c['key']].astype(np.float32)
    gt = sio.loadmat(find(c['gt']))[c['gtkey']].astype(np.int64)
    x = (x - x.mean((0, 1), keepdims=True)) / (x.std((0, 1), keepdims=True) + 1e-6)
    return x, gt, int(gt.max())


def split(gt, ratio, seed, min_train=3):
    """每类按比例随机取训练样本（至少 min_train 个），其余为测试。返回坐标与 0 起标签。"""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in range(1, gt.max() + 1):
        loc = np.argwhere(gt == c)
        rng.shuffle(loc)
        n = max(int(math.ceil(ratio * len(loc))), min_train)
        tr.append(loc[:n])
        te.append(loc[n:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    return tr, gt[tr[:, 0], tr[:, 1]] - 1, te, gt[te[:, 0], te[:, 1]] - 1


class PatchSampler:
    """整幅图放 GPU，按坐标批量取 p×p 邻域块，输出 [N, D, p, p]。"""

    def __init__(self, cube, patch, device='cuda'):
        self.r = patch // 2
        self.p = patch
        t = torch.from_numpy(np.ascontiguousarray(cube.transpose(2, 0, 1))).to(device)
        self.img = torch.nn.functional.pad(t[None], (self.r,) * 4, mode='reflect')[0]
        self.off = torch.arange(patch, device=device)
        self.device = device

    def __call__(self, coords):
        c = torch.as_tensor(coords, device=self.device)
        rows = (c[:, 0:1] + self.off)[:, :, None]
        cols = (c[:, 1:2] + self.off)[:, None, :]
        return self.img[:, rows, cols].permute(1, 0, 2, 3).contiguous()


def pca_cube(cube, k):
    """整幅图 PCA 到 k 维（无监督，只用光谱），再逐分量 z-score。"""
    h, w, d = cube.shape
    if k >= d:
        return cube
    x = cube.reshape(-1, d).astype(np.float64)
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x[::max(1, len(x) // 50000)], full_matrices=False)
    z = (x @ vt[:k].T).reshape(h, w, k).astype(np.float32)
    return (z - z.mean((0, 1))) / (z.std((0, 1)) + 1e-6)
