"""生成结果表：每个方法在多个随机划分上的 OA/AA/Kappa 均值±标准差与逐类精度均值。
对比方法读 runs/<ds>/<method>/seed<k>.json，MH-DRL 读 runs/<ds>/boost/<tag>_seed<k>/ensemble.json。

python -m mhdrl.report --seeds 0,1,2 --out results
"""
import argparse
import json
import os

import numpy as np

RUNS = os.environ.get('HSI_RUNS', 'runs')
NAMES = [('bsnets', 'BS-Nets'), ('abcnn', 'ABCNN'), ('mrsvm', 'MR-SVM'), ('ddcnn', 'DDCNN'), ('twcnn', 'TWCNN'),
         ('sicnn', 'SICNN'), ('drlbs', 'DRLBS'), ('mhdrl', 'MH-DRL')]
TITLE = {'indian': 'Indian Pines (60 bands)', 'paviau': 'Pavia University (30 bands)',
         'houston': 'University of Houston (40 bands)'}


def load_metrics(ds, key, seed, mh_tag):
    if key == 'mhdrl':
        d = json.load(open(os.path.join(RUNS, ds, 'boost', f'{mh_tag}_seed{seed}', 'ensemble.json')))
        return d, d['bands']
    d = json.load(open(os.path.join(RUNS, ds, key, f'seed{seed}.json')))
    return d.get('metrics_own', d['metrics']), d['bands']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='indian,paviau,houston')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--mh_tag', default='mhdrl')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',')]
    md = [f'Mean ± std over {len(seeds)} random splits (seeds {a.seeds}).\n']
    bands_out = {}
    for ds in a.datasets.split(','):
        res = {n: [load_metrics(ds, k, s, a.mh_tag) for s in seeds] for k, n in NAMES}
        names = [n for _, n in NAMES]
        md.append(f'### {TITLE[ds]}\n')
        md.append('| Class | ' + ' | '.join(names) + ' |')
        md.append('|---|' + '---|' * len(names))
        n_cls = len(res[names[0]][0][0]['per_class'])
        for c in range(n_cls):
            md.append(f'| {c + 1} | ' + ' | '.join(
                f'{np.mean([m["per_class"][c] for m, _ in res[n]]) * 100:.1f}' for n in names) + ' |')
        for key, lab in [('oa', 'OA'), ('aa', 'AA'), ('kappa', 'Kappa')]:
            vals = {n: np.array([m[key] for m, _ in res[n]]) * 100 for n in names}
            best = max(names, key=lambda n: vals[n].mean())
            cell = lambda n: f'{vals[n].mean():.2f} ± {vals[n].std():.2f}'
            md.append(f'| **{lab}** | ' + ' | '.join(f'**{cell(n)}**' if n == best else cell(n) for n in names) + ' |')
        md.append('')
        bands_out[ds] = {f'seed{s}': {n: list(map(int, res[n][i][1])) for n in names} for i, s in enumerate(seeds)}
    open(os.path.join(a.out, 'main_table.md'), 'w').write('\n'.join(md))
    json.dump(bands_out, open(os.path.join(a.out, 'selected_bands.json'), 'w'), indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
