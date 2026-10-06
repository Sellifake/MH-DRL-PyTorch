"""用发布的权重复现结果：读入 MH-DRL 选出的波段和集成分类器权重，在同一划分的测试像素上算 OA/AA/Kappa，
可选输出整幅分类图。

python -m mhdrl.infer --ds indian --weights weights/indian --map indian_map.png
weights/<ds>/ 目录内容：ensemble.json（波段、成员列表、划分种子）与每个成员的 .pt（fp16）。
"""
import argparse
import json
import os

import numpy as np
import torch

from .classify import build, metrics, predict
from .data import DATASETS, PatchSampler, load, split

PALETTE = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                    [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212], [218, 112, 214],
                    [160, 82, 45], [127, 255, 0], [216, 191, 216], [238, 0, 0]], np.uint8)


def load_members(wdir, n_bands, n_cls, device='cuda'):
    cfg = json.load(open(os.path.join(wdir, 'ensemble.json')))
    members = []
    for mem in cfg['members']:
        arch, patch = mem.split(':')[:2]
        net = build(arch, n_bands, n_cls, int(patch)).to(device)
        sd = torch.load(os.path.join(wdir, mem.replace(':', '_') + '.pt'), map_location=device, weights_only=True)
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        members.append((net, int(patch)))
    return cfg, members


def ensemble_prob(cube_sel, members, coords, chunk=20000):
    """各成员 8 向 TTA 的 softmax 之和；每个成员的整幅图只上传一次 GPU，坐标分块推理。"""
    prob = 0
    for net, patch in members:
        sampler = PatchSampler(cube_sel, patch)
        prob = prob + np.concatenate([predict(net, sampler, coords[i:i + chunk], True)
                                      for i in range(0, len(coords), chunk)])
        del sampler
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='indian')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--map', default='', help='分类图 PNG 路径（只画有标签像素）；另存 _full（整幅）与 _gt（真值）；空 = 不画')
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    cube, gt, n_cls = load(a.ds)
    bands = np.asarray(json.load(open(os.path.join(a.weights, 'ensemble.json')))['bands'])
    cfg, members = load_members(a.weights, len(bands), n_cls)
    tr, ytr, te, yte = split(gt, DATASETS[a.ds]['ratio'], cfg['seed'])
    sub = cube[:, :, bands]
    m = metrics(yte, ensemble_prob(sub, members, te).argmax(1), n_cls)
    print(f'{a.ds}: {len(bands)} bands, OA {m["oa"] * 100:.2f} AA {m["aa"] * 100:.2f} Kappa {m["kappa"] * 100:.2f}')
    print('per-class:', ' '.join(f'{x * 100:.1f}' for x in m['per_class']))
    if a.map:
        from PIL import Image
        H, W = gt.shape
        coords = np.argwhere(np.ones((H, W), bool))
        pred = ensemble_prob(sub, members, coords).argmax(1).reshape(H, W) + 1
        Image.fromarray(PALETTE[pred % len(PALETTE)]).save(a.map.replace('.png', '_full.png'))
        Image.fromarray(PALETTE[np.where(gt > 0, pred, 0) % len(PALETTE)]).save(a.map)
        Image.fromarray(PALETTE[gt % len(PALETTE)]).save(a.map.replace('.png', '_gt.png'))
        print('map saved to', a.map)


if __name__ == '__main__':
    main()
