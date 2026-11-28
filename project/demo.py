"""Top-3 predictions for one photo from the ordinary and strong checkpoints of a seed (offline, CPU).

Usage: python demo.py examples/field.jpg
"""
import argparse
from pathlib import Path

import torch

from train import CONDITIONS, encode, load_rgb, model, predict, prefix


def classify(photo, seed_dir):
    torch.set_num_threads(2)
    pixels = load_rgb(Path(photo).read_bytes())[None]
    for condition in CONDITIONS:
        ckpt = torch.load(seed_dir / f'{condition}.pt', map_location='cpu', weights_only=True)
        net = model(False)
        net.load_state_dict(ckpt['model'])
        net.eval()
        probs = predict(net, encode(prefix(net), pixels, 'cpu'), 'cpu')[0]
        print(f"{condition} (seed {ckpt['seed']}, epoch {ckpt['epoch']})")
        for i in probs.argsort(descending=True)[:3]:
            print(f"  {ckpt['labels'][i]:22} {float(probs[i]):.1%}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('photo', type=Path)
    p.add_argument('--seed-dir', type=Path, default=Path(__file__).resolve().parent / 'results' / 'seed_664')
    a = p.parse_args()
    classify(a.photo, a.seed_dir)
