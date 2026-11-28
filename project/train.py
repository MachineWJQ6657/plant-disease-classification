"""Ordinary vs strong crop/color augmentation, three paired seeds, fresh augmented views every epoch.

Each seed trains both conditions from the same ImageNet initialization with the same image order.
All checkpoints are chosen on PlantVillage validation and hash-locked before any test image is scored.

Usage: python train.py --data data --out rerun --device cuda
"""
import argparse
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from analysis import analyze, audit
from data import LABELS, NAMES, disk_path, read_csv, write_csv

SIZE, BATCH, EPOCHS, LR, WEIGHT_DECAY = 128, 96, 8, 5e-4, 1e-4
SEEDS, CONDITIONS = [664, 665, 666], ['ordinary', 'strong']
MEAN, STD = [.485, .456, .406], [.229, .224, .225]


def model(pretrained):
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(LABELS))
    return net


def prefix(net):
    """Frozen ImageNet layers. Only the last block, final convolution and classifier are trained."""
    return net.features[:-2]


def head(net, x):
    return net.classifier(net.avgpool(net.features[-2:](x)).flatten(1))


@torch.inference_mode()
def predict(net, feats, device):
    net.eval()
    return torch.cat([head(net, x.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)).softmax(1).cpu()
                      for x in feats.split(BATCH)])


def deterministic():
    """Numerical settings used for training; replay must restore them to reproduce predictions."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest_state(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def event(out, kind, **values):
    row = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=kind, **values)
    with (out / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def crop_parameters(rng, width, height):
    """RandomResizedCrop parameters: area 0.55-1, log-uniform aspect ratio 0.8-1.25."""
    for _ in range(10):
        target = rng.uniform(.55, 1.) * width * height
        aspect = math.exp(rng.uniform(math.log(.8), math.log(1.25)))
        w, h = round(math.sqrt(target * aspect)), round(math.sqrt(target / aspect))
        if 0 < w <= width and 0 < h <= height:
            return rng.randint(0, height - h), rng.randint(0, width - w), h, w
    ratio = width / height  # same central fallback as torchvision
    w, h = (width, round(width / .8)) if ratio < .8 else (round(height * 1.25), height) if ratio > 1.25 else (width, height)
    return (height - h) // 2, (width - w) // 2, h, w


def fresh_transform(rgb, condition, seed, epoch, index):
    """A new view per (seed, epoch, image); a local RNG makes it independent of thread scheduling."""
    rng = random.Random(int(seed) * 1_000_000_000 + int(epoch) * 100_000 + int(index))
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if condition == 'strong':
        top, left, height, width = crop_parameters(rng, *im.size)
        im = F.resized_crop(im, top, left, height, width, [SIZE, SIZE], InterpolationMode.BILINEAR, antialias=True)
        factors = [rng.uniform(.65, 1.35), rng.uniform(.65, 1.35), rng.uniform(.65, 1.35), rng.uniform(-.07, .07)]
        order = list(range(4))
        rng.shuffle(order)
        for k in order:
            im = [F.adjust_brightness, F.adjust_contrast, F.adjust_saturation, F.adjust_hue][k](im, factors[k])
    elif condition != 'ordinary':
        raise ValueError('Unknown condition')
    if rng.random() < .5:
        im = F.hflip(im)
    return F.normalize(F.to_tensor(im), MEAN, STD)


def load_rgb(raw):
    """Deterministic preprocessing shared by training, testing, replay and the demo."""
    with Image.open(io.BytesIO(raw)) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        return np.asarray(F.resize(rgb, [SIZE, SIZE], InterpolationMode.BILINEAR, antialias=True))


def load_pixels(rows, data, cache=None, workers=4):
    """uint8 pixels for a split, checked against the manifest hashes. Random views are never cached."""
    key = sha256(json.dumps(dict(paths=[(r['path'], r['sha256']) for r in rows], size=SIZE), sort_keys=True).encode())
    path = cache / f'{key[:16]}.npy' if cache else None
    if path and path.exists():
        return np.load(path, mmap_mode='r')

    def decode(r):
        raw = disk_path(data, r['path']).read_bytes()
        if sha256(raw) != r['sha256']:
            raise ValueError('Raw image hash mismatch: ' + r['path'])
        return load_rgb(raw)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pixels = np.stack(list(pool.map(decode, rows)))
    if path:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix('.part.npy'), pixels)
        path.with_suffix('.part.npy').replace(path)
    return pixels


@torch.inference_mode()
def encode(frozen, pixels, device):
    """Frozen-prefix activations, stored as float16."""
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    blocks = []
    for start in range(0, len(pixels), BATCH):
        x = torch.from_numpy(np.array(pixels[start:start + BATCH], copy=True)).permute(0, 3, 1, 2)
        x = x.to(device=device, dtype=torch.float32, memory_format=torch.channels_last) / 255.
        blocks.append(frozen((x - mean) / std).cpu().half())
    return torch.cat(blocks)


def train_condition(initial, condition, seed, pixels, y, val_features, y_val, device, pool, out, meta):
    net = model(False).to(device=device, memory_format=torch.channels_last)
    net.load_state_dict(initial)
    prefix(net).requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    torch.manual_seed(seed)  # classifier dropout
    records, best, start = [], -1., time.monotonic()
    for epoch in range(EPOCHS):
        tic = time.monotonic()
        net.train()
        prefix(net).eval()
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + epoch * 1_000_000))
        loss_sum, steps = 0., 0
        for ix in order.split(BATCH):
            views = pool.map(lambda i: fresh_transform(pixels[i], condition, seed, epoch, i), ix.tolist())
            x = torch.stack(list(views)).to(device=device, memory_format=torch.channels_last)
            with torch.no_grad():
                features = prefix(net)(x).half().float()
            loss = nn.functional.cross_entropy(head(net, features), y[ix.to(device)])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum, steps = loss_sum + float(loss.detach()) * len(ix), steps + 1
        pred = predict(net, val_features, device).numpy().argmax(1)
        score = float(f1_score(y_val, pred, labels=np.unique(y_val), average='macro', zero_division=0))
        row = dict(condition=condition, epoch=epoch + 1, train_loss=loss_sum / len(y),
                   val_accuracy=float((pred == y_val).mean()), val_supported_macro_f1=score,
                   seconds=round(time.monotonic() - tic, 3), optimizer_steps=steps,
                   sample_order_sha256=sha256(order.numpy().tobytes()), augmentation_epoch=epoch)
        records.append(row)
        if score > best:  # first epoch with the best supported-class validation macro-F1
            best = score
            torch.save(dict(model={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                            condition=condition, epoch=epoch + 1, size=SIZE, labels=NAMES, seed=seed, **meta),
                       out / f'seed_{seed}' / f'{condition}.pt')
        event(out, 'epoch_complete', seed=seed, **row)
    assert digest_state(prefix(net).state_dict()) == meta['source_prefix_sha256']
    write_json(out / f'seed_{seed}' / f'{condition}_training_complete.json', dict(
        seed=seed, condition=condition, epochs=EPOCHS, optimizer_steps=sum(r['optimizer_steps'] for r in records),
        image_presentations=len(y) * EPOCHS, initial_state_sha256=meta['initial_state_sha256'],
        final_state_sha256=digest_state(net.state_dict()), frozen_prefix_unchanged=True,
        training_seconds=round(time.monotonic() - start, 3), completed_utc=datetime.now(timezone.utc).isoformat()))
    return records


def run(data, manifest, out, cache, device, seeds=SEEDS, workers=4, threads=4):
    if (out / 'protocol.json').exists():
        raise SystemExit(f'{out} already holds a run; choose a new --out so saved evidence is never overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    deterministic()
    if manifest.resolve() != (out / 'manifest.csv').resolve():
        shutil.copy2(manifest, out / 'manifest.csv')
    rows = read_csv(manifest)
    sets = {s: [r for r in rows if r['split'] == s] for s in ['pv_train', 'pv_val', 'pv_test', 'pd_test']}
    script = sha256(Path(__file__).read_bytes())
    write_json(out / 'protocol.json', dict(
        seeds=seeds, conditions=CONDITIONS, epochs=EPOCHS, batch_size=BATCH, input_size=SIZE,
        optimizer='AdamW', learning_rate=LR, weight_decay=WEIGHT_DECAY,
        trainable='last inverted residual block, final convolution, classifier',
        strong='crop area .55-1, log-uniform ratio .8-1.25; brightness/contrast/saturation .35, hue .07',
        view_seed='seed*1000000000 + epoch*100000 + image_index',
        checkpoint_selection='first maximum supported-class source-validation macro-F1',
        manifest_sha256=sha256(manifest.read_bytes()), script_sha256=script,
        torch=torch.__version__, torchvision=torchvision.__version__, python=platform.python_version(),
        hardware=torch.cuda.get_device_name(0) if device.startswith('cuda') else platform.processor(), device=device))
    event(out, 'run_started', seeds=seeds, device=device, script_sha256=script)
    train_pixels = load_pixels(sets['pv_train'], data, cache, workers)
    y = torch.tensor([int(r['label']) for r in sets['pv_train']], device=device)
    y_val = np.array([int(r['label']) for r in sets['pv_val']])
    frozen = prefix(model(True).to(device=device, memory_format=torch.channels_last)).eval().requires_grad_(False)
    prefix_hash = digest_state(frozen.state_dict())
    val_features = encode(frozen, load_pixels(sets['pv_val'], data, cache, workers), device)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for seed in seeds:
            (out / f'seed_{seed}').mkdir(exist_ok=True)
            torch.manual_seed(seed)  # new classifier layer
            initial = {k: v.detach().cpu().clone() for k, v in model(True).state_dict().items()}
            meta = dict(protocol='fresh_views_v1', initial_state_sha256=digest_state(initial),
                        source_prefix_sha256=prefix_hash, script_sha256=script)
            history = []
            for condition in CONDITIONS:
                history += train_condition(initial, condition, seed, train_pixels, y, val_features, y_val,
                                           device, pool, out, meta)
            write_csv(out / f'seed_{seed}' / 'history.csv', history)
    write_json(out / 'selection_locked.json', dict(
        time_utc=datetime.now(timezone.utc).isoformat(), seeds=seeds, all_conditions_complete=True,
        checkpoint_sha256={f'{s}/{c}': sha256((out / f'seed_{s}' / f'{c}.pt').read_bytes()) for s in seeds for c in CONDITIONS}))
    event(out, 'all_checkpoints_locked_before_test_evaluation')
    predictions = {seed: [] for seed in seeds}
    for split in ['pv_test', 'pd_test']:
        feats = encode(frozen, load_pixels(sets[split], data, cache, workers), device)
        for seed in seeds:
            for condition in CONDITIONS:
                net = model(False).to(device=device, memory_format=torch.channels_last)
                net.load_state_dict(torch.load(out / f'seed_{seed}' / f'{condition}.pt', map_location='cpu',
                                               weights_only=True)['model'])
                assert digest_state(prefix(net).state_dict()) == prefix_hash
                for r, p in zip(sets[split], predict(net, feats, device).numpy()):
                    predictions[seed].append(dict(path=r['path'], split=split, condition=condition, y_true=r['label'],
                                                  y_pred=int(p.argmax()), confidence=float(p.max()),
                                                  **{f'p_{k}': float(v) for k, v in enumerate(p)}))
            event(out, 'test_predictions_saved', seed=seed, split=split)
    for seed in seeds:
        write_csv(out / f'seed_{seed}' / 'predictions.csv', predictions[seed])
        analyze(out / f'seed_{seed}')
        audit(out / f'seed_{seed}', out / 'manifest.csv')
    event(out, 'run_complete', seeds=seeds)
    write_json(out / 'run_complete.json', dict(
        status='PASS', completed_utc=datetime.now(timezone.utc).isoformat(), trained_models=2 * len(seeds),
        epochs_per_model=EPOCHS, total_optimizer_steps=2 * len(seeds) * math.ceil(len(y) / BATCH) * EPOCHS,
        total_image_presentations=2 * len(seeds) * len(y) * EPOCHS, all_artifact_audits_passed=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('results/manifest.csv'))
    p.add_argument('--out', type=Path, default=Path('rerun'))
    p.add_argument('--cache', type=Path, default=Path('.build/cache'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    run(a.data, a.manifest, a.out, a.cache, a.device, a.seeds, a.workers, a.threads)
