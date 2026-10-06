"""README 用的可视化：OA 对比、MH-DRL 搜索过程、选中波段位置、分类图。

python -m mhdrl.figures --seeds 0,1,2 --pick indian:0,paviau:2,houston:2 --out results/figs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .data import load

RUNS = os.environ.get('HSI_RUNS', 'runs')
BLUE, GRAY, INK, MUTED, GRID = '#2a78d6', '#a3a29b', '#1f1f1d', '#6b6a64', '#e6e5df'
TITLE = {'indian': 'Indian Pines', 'paviau': 'Pavia University', 'houston': 'University of Houston'}
NAMES = [('bsnets', 'BS-Nets'), ('abcnn', 'ABCNN'), ('mrsvm', 'MR-SVM'), ('ddcnn', 'DDCNN'), ('twcnn', 'TWCNN'),
         ('sicnn', 'SICNN'), ('drlbs', 'DRLBS')]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200})


def oa_of(ds, key, seed, mh_tag):
    if key == 'mhdrl':
        return json.load(open(os.path.join(RUNS, ds, 'boost', f'{mh_tag}_seed{seed}', 'ensemble.json')))['oa'] * 100
    d = json.load(open(os.path.join(RUNS, ds, key, f'seed{seed}.json')))
    return d.get('metrics_own', d['metrics'])['oa'] * 100


def fig_oa(datasets, seeds, mh_tag, out):
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.4))
    for ax, ds in zip(axes, datasets):
        rows = []
        for k, n in NAMES + [('mhdrl', 'MH-DRL')]:
            v = np.array([oa_of(ds, k, s, mh_tag) for s in seeds])
            rows.append((n, v.mean(), v.std()))
        rows.sort(key=lambda r: r[1])
        lo = np.floor(min(m - e for _, m, e in rows) / 5) * 5
        for i, (n, m, e) in enumerate(rows):
            c = BLUE if n == 'MH-DRL' else GRAY
            ax.plot([lo, m], [i, i], color=GRID, lw=2, zorder=1)
            ax.plot([m - e, m + e], [i, i], color=c, lw=2, zorder=2)
            ax.scatter(m, i, s=64 if n == 'MH-DRL' else 40, color=c, zorder=3, edgecolor='white', linewidth=2)
            if n == 'MH-DRL' or i == len(rows) - 2:
                ax.text(m - 0.6, i + 0.32, f'{m:.2f}', ha='right', va='bottom', fontsize=9, color=INK)
        ax.set_yticks(np.arange(len(rows)), [n for n, _, _ in rows])
        for t in ax.get_yticklabels():
            t.set_color(INK if t.get_text() == 'MH-DRL' else MUTED)
            t.set_fontweight('bold' if t.get_text() == 'MH-DRL' else 'normal')
        ax.set_xlim(lo, 100.5)
        ax.set_xlabel(f'Overall accuracy (%), mean ± std of {len(seeds)} splits')
        ax.set_title(TITLE[ds], color=INK, fontsize=11, loc='left')
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'oa_comparison.png'))
    plt.close(fig)


def fig_search(ds, seed, mh_tag, out):
    h = json.load(open(os.path.join(RUNS, ds, mh_tag, f'seed{seed}.json')))['hist']
    R = np.array([x['R'] for x in h])
    acc = np.array([x['acc'] for x in h]) * 100
    t = np.arange(len(R))
    k = 25
    smooth = lambda v: np.convolve(v, np.ones(k) / k, mode='valid')
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.6), sharex=True)
    guide = [(x['t'], x['teacher']) for x in h if x['teacher'] != '-']
    spans = {}
    for tt, who in guide:
        spans.setdefault(who, [tt, tt])[1] = tt
    labels = {'bsnets': 'BS-Nets', 'sicnn': 'SICNN', 'twcnn': 'TWCNN'}
    for ax, v, lab in [(axes[0], R, 'Reward'), (axes[1], acc, 'PE-Net accuracy (%)')]:
        for j, (who, (a, b)) in enumerate(spans.items()):
            ax.axvspan(a, b + 1, color=['#eef4fc', '#e3edfa', '#d6e5f8'][j % 3], lw=0, zorder=0)
        ax.plot(t, v, color=BLUE, alpha=0.18, lw=1)
        ax.plot(t[k - 1:], smooth(v), color=BLUE, lw=2)
        ax.set_ylabel(lab)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    top = axes[0].get_ylim()[1]
    for who, (a, b) in spans.items():
        axes[0].text((a + b) / 2, top, labels.get(who, who), ha='center', va='top', fontsize=9, color=MUTED)
    if spans:
        end = max(b for _, b in spans.values())
        axes[0].text((end + len(R)) / 2, top, 'self-exploration (DQN)', ha='center', va='top', fontsize=9, color=MUTED)
    axes[1].set_xlabel('Step')
    axes[0].set_title(f'MH-DRL band search on {TITLE[ds]}, split seed {seed} (teacher-guided stages shaded)',
                      color=INK, fontsize=11, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'search_{ds}.png'))
    plt.close(fig)


def entropy(cube):
    px = cube.reshape(-1, cube.shape[2])
    px = px[np.random.RandomState(0).permutation(len(px))[:50000]]
    ie = []
    for i in range(px.shape[1]):
        hist, _ = np.histogram(px[:, i], bins=256)
        p = hist[hist > 0] / hist.sum()
        ie.append(-(p * np.log2(p)).sum())
    return np.array(ie)


def fig_bands(picks, mh_tag, out):
    fig, axes = plt.subplots(len(picks), 1, figsize=(8.4, 2.1 * len(picks)))
    for ax, (ds, seed) in zip(axes, picks):
        cube, _, _ = load(ds)
        ie = entropy(cube)
        bands = json.load(open(os.path.join(RUNS, ds, 'boost', f'{mh_tag}_seed{seed}', 'ensemble.json')))['bands']
        ax.plot(np.arange(len(ie)), ie, color=GRAY, lw=1.5)
        ax.scatter(bands, ie[bands], s=36, color=BLUE, zorder=3, edgecolor='white', linewidth=1.5)
        ax.set_xlim(-1, len(ie))
        ax.set_ylabel('Entropy')
        ax.set_title(f'{TITLE[ds]}: {len(bands)} of {len(ie)} bands selected by MH-DRL (split seed {seed})', color=INK,
                     fontsize=10, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel('Band index')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'selected_bands.png'))
    plt.close(fig)


def fig_maps(picks, maps_dir, out):
    fig = plt.figure(figsize=(12, 2.9))
    widths = []
    imgs = []
    for ds, _ in picks:
        gt = np.array(Image.open(os.path.join(maps_dir, f'{ds}_gt.png')))
        pr = np.array(Image.open(os.path.join(maps_dir, f'{ds}.png')))
        if gt.shape[0] > 2 * gt.shape[1]:           # Houston 存成竖长条，转成横向展示
            gt, pr = np.rot90(gt), np.rot90(pr)
        imgs.append((ds, gt, pr))
        widths.append(gt.shape[1] / gt.shape[0])
    gs = fig.add_gridspec(2, len(imgs), width_ratios=widths, hspace=0.08, wspace=0.04)
    for j, (ds, gt, pr) in enumerate(imgs):
        for i, (im, lab) in enumerate([(gt, 'Ground truth'), (pr, 'MH-DRL')]):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(im, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title({'indian': 'Indian Pines', 'paviau': 'Pavia U.', 'houston': 'Houston'}[ds], color=INK,
                             fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, color=INK)
    fig.savefig(os.path.join(out, 'classification_maps.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pick', default='indian:0,paviau:2,houston:2', help='波段图、搜索曲线、分类图用的划分')
    ap.add_argument('--seeds', default='0,1,2', help='OA 对比图取均值的划分')
    ap.add_argument('--mh_tag', default='mhdrl')
    ap.add_argument('--maps', default='results/maps')
    ap.add_argument('--out', default='results/figs')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    picks = [(p.split(':')[0], int(p.split(':')[1])) for p in a.pick.split(',')]
    fig_oa([ds for ds, _ in picks], [int(x) for x in a.seeds.split(',')], a.mh_tag, a.out)
    for ds, seed in picks:
        fig_search(ds, seed, a.mh_tag, a.out)
    fig_bands(picks, a.mh_tag, a.out)
    fig_maps(picks, a.maps, a.out)
    print('figures written to', a.out)


if __name__ == '__main__':
    main()
