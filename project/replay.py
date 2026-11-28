"""Replay one seed's checkpoints on the raw test images and compare with its saved predictions.

Usage: python replay.py --data data --seed-dir results/seed_664 --device cuda
"""
import argparse
import json
from pathlib import Path

import torch

from data import NAMES, read_csv
from train import CONDITIONS, SIZE, deterministic, encode, load_pixels, model, predict, prefix


def replay(data, seed_dir, manifest, out, device):
    torch.set_num_threads(4)
    deterministic()  # TF32 or nondeterministic kernels can flip near-boundary predictions
    tests = {s: [r for r in read_csv(manifest) if r['split'] == s] for s in ['pv_test', 'pd_test']}
    pixels = {s: load_pixels(rows, data) for s, rows in tests.items()}  # also checks every raw-image hash
    saved = {(r['condition'], r['path']): r for r in read_csv(seed_dir / 'predictions.csv')}
    mismatches, max_delta, n = 0, 0., 0
    for condition in CONDITIONS:
        ckpt = torch.load(seed_dir / f'{condition}.pt', map_location=device, weights_only=True)
        if ckpt['labels'] != NAMES or ckpt['size'] != SIZE or ckpt['condition'] != condition:
            raise ValueError('Checkpoint metadata mismatch')
        net = model(False).to(device=device, memory_format=torch.channels_last)
        net.load_state_dict(ckpt['model'])
        net.eval()
        for split, rows in tests.items():
            for r, p in zip(rows, predict(net, encode(prefix(net), pixels[split], device), device).numpy()):
                old = saved[condition, r['path']]
                mismatches += int(p.argmax()) != int(old['y_pred'])
                max_delta = max(max_delta, *(abs(float(p[k]) - float(old[f'p_{k}'])) for k in range(9)))
                n += 1
    report = dict(status='PASS' if mismatches == 0 and max_delta < .005 else 'REVIEW',
                  raw_image_hashes_checked=sum(map(len, tests.values())), prediction_rows=n, device=device,
                  torch=torch.__version__, top1_mismatches=mismatches, max_probability_delta=max_delta)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'replay_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit('Replay differs materially; review hardware/library differences before claiming reproduction.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--seed-dir', type=Path, default=Path('results/seed_664'))
    p.add_argument('--manifest', type=Path, default=Path('results/manifest.csv'))
    p.add_argument('--out', type=Path, default=Path('replay'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    replay(a.data, a.seed_dir, a.manifest, a.out / a.seed_dir.name, a.device)
