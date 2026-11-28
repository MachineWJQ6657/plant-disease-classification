"""Class mapping, pinned downloads, portable file names and the audited train/val/test manifest.

Usage: python data.py --data data --out reconstructed
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import unquote
from urllib.request import urlopen
from zipfile import ZipFile

# (category, PlantVillage folder suffix, PlantDoc folder). The order is the class index in every checkpoint.
LABELS = [
    ("Healthy", "healthy", "Tomato leaf"),
    ("Early blight", "Early_blight", "Tomato Early blight leaf"),
    ("Bacterial spot", "Bacterial_spot", "Tomato leaf bacterial spot"),
    ("Late blight", "Late_blight", "Tomato leaf late blight"),
    ("Mosaic virus", "Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
    ("Yellow leaf curl", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    ("Leaf mold", "Leaf_Mold", "Tomato mold leaf"),
    ("Septoria", "Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    ("Spider mites", "Spider_mites Two-spotted_spider_mite", "Tomato two spotted spider mites leaf"),
]
NAMES = [name for name, _, _ in LABELS]
# Pinned upstream revisions: PlantVillage, PlantDoc and the official leaf-grouped split lists.
PV_COMMIT = '7f7ecc7e1eaca78107e3affe7cb5abd9427e139a'
PD_COMMIT = '5467f6012d78d1c446145d5f582da6096f852ae8'
HF_COMMIT = '9e97599868962bd0079b8db4b7f1efa9185fa1e7'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.jfif', '.tif', '.tiff'}


def norm(s):
    return re.sub('[^a-z0-9]+', '', s.casefold())


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# Upstream names contain characters that Windows forbids. Disk names encode them reversibly;
# manifests always store the original upstream path.
def storage_root(root):
    value = os.path.abspath(os.path.normpath(os.fspath(root)))
    if os.name == 'nt' and not value.startswith('\\\\?\\'):  # extended-length paths for long names
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return Path(value)


def portable_component(name):
    value = re.sub(r'[%<>:"\\|?*\x00-\x1f]', lambda m: '%%%02X' % ord(m[0]), name)
    value = re.sub(r'[. ]+$', lambda m: ''.join('%%%02X' % ord(c) for c in m[0]), value)
    if re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', value, re.I):
        value = '%%%02X' % ord(value[0]) + value[1:]
    return value


def disk_path(root, canonical):
    parts = PurePosixPath(canonical).parts
    if not parts or '..' in parts or PurePosixPath(canonical).is_absolute():
        raise ValueError('Expected safe relative upstream path')
    return storage_root(root).joinpath(*(portable_component(p) for p in parts))


def canonical_path(path, root):
    return '/'.join(unquote(p) for p in storage_root(path).relative_to(storage_root(root)).parts)


def fetch(url, path):
    if not path.exists():
        temp = path.with_suffix('.part')
        with urlopen(url, timeout=180) as response, temp.open('wb') as f:
            shutil.copyfileobj(response, f)
        temp.replace(path)


def download(root):
    """Fetch the pinned snapshots (~3.5 GB) and extract only the nine tomato classes."""
    root = storage_root(root)
    root.mkdir(parents=True, exist_ok=True)
    pv_classes = {norm('Tomato___' + pv) for _, pv, _ in LABELS}
    pd_classes = {norm(pd) for _, _, pd in LABELS}
    for name, commit, repo, prefix in [
            ('pv', PV_COMMIT, 'spMohanty/PlantVillage-Dataset', 'PlantVillage-Dataset-master'),
            ('pd', PD_COMMIT, 'pratikkayal/PlantDoc-Dataset', 'PlantDoc-Dataset-master')]:
        archive = root / (name + '.zip')
        fetch(f'https://codeload.github.com/{repo}/zip/{commit}', archive)
        with ZipFile(archive) as z:
            if z.comment.decode().strip() != commit:
                raise ValueError(f'Unexpected snapshot in {archive}. Remove it and retry.')
            for info in z.infolist():
                parts = PurePosixPath(info.filename).parts[1:]
                if info.is_dir() or '..' in parts:
                    continue
                if name == 'pv':
                    keep = len(parts) == 4 and parts[:2] == ('raw', 'color') and norm(parts[2]) in pv_classes
                else:
                    keep = len(parts) == 3 and parts[0] in ('train', 'test') and norm(parts[1]) in pd_classes
                if keep:
                    target = disk_path(root, PurePosixPath(prefix, *parts).as_posix())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, target.open('wb') as dest:
                        shutil.copyfileobj(src, dest)
    fetch(f'https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/{PV_COMMIT}/leaf_grouping/leaf-map.json',
          root / 'leaf-map.json')
    for split in ('train', 'test'):
        fetch(f'https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{HF_COMMIT}/splits/color_{split}.txt',
              root / f'color_{split}.txt')


def prepare(root, out):
    """Write manifest.csv (official leaf-grouped split + leaf-grouped validation), exclusions.csv and class_mapping.csv."""
    root = storage_root(root)
    leafmap = json.loads((root / 'leaf-map.json').read_text(encoding='utf-8'))
    source = {norm('Tomato___' + v): i for i, (_, v, _) in enumerate(LABELS)}
    field = {norm(v): i for i, (_, _, v) in enumerate(LABELS)}
    rows, excluded = [], []

    def add(path, domain, upstream, split, k, group=''):
        rows.append(dict(path=canonical_path(path, root), domain=domain, upstream_split=upstream, split=split,
                         label=k, category=NAMES[k], leaf_group=group,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    for split in ('train', 'test'):
        for line in (root / f'color_{split}.txt').read_text(encoding='utf-8').splitlines():
            path = disk_path(root, 'PlantVillage-Dataset-master/' + line.strip())
            folder = path.parent.name
            k = source.get(norm(folder))
            if k is None:
                continue
            key = path.stem.split('___')[-1].strip().lower()
            ids = [g for g in leafmap.get(key, []) if g.startswith(folder + ':::')]
            if len(ids) > 1:
                raise ValueError(f'Ambiguous leaf id: {path}')
            add(path, 'PlantVillage', split, 'pv_' + split, k, ids[0] if ids else '')
    # 15% of documented leaf groups per class move to validation. Images without a leaf ID stay in training.
    rng = random.Random(664)
    for k in range(len(LABELS)):
        groups = sorted({r['leaf_group'] for r in rows if r['label'] == k and r['split'] == 'pv_train' and r['leaf_group']})
        rng.shuffle(groups)
        holdout = set(groups[:round(0.15 * len(groups))])
        for r in rows:
            if r['split'] == 'pv_train' and r['leaf_group'] in holdout:
                r['split'] = 'pv_val'
    for split in ('train', 'test'):  # PlantDoc train and test folders are pooled as the external test only
        for path in sorted((root / 'PlantDoc-Dataset-master' / split).glob('*/*'), key=lambda p: canonical_path(p, root)):
            k = field.get(norm(path.parent.name))
            if k is not None and path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                add(path, 'PlantDoc', split, 'pd_test', k)
    # Exact-byte audit: copies in a held-out split win; PlantDoc duplicates and label conflicts are removed.
    priority = {'pv_train': 0, 'pv_val': 1, 'pv_test': 2, 'pd_test': 3}
    by_hash, remove = defaultdict(list), set()
    for r in rows:
        by_hash[r['sha256']].append(r)
    for batch in by_hash.values():
        if len({r['split'] for r in batch}) > 1:
            high = max(priority[r['split']] for r in batch)
            for r in batch:
                if priority[r['split']] < high:
                    excluded.append(dict(**r, reason='identical bytes in a held-out split'))
                    remove.add(r['path'])
        if batch[0]['domain'] == 'PlantDoc' and len(batch) > 1:
            conflict = len({r['label'] for r in batch}) > 1
            for r in (batch if conflict else sorted(batch, key=lambda x: x['path'])[1:]):
                excluded.append(dict(**r, reason='conflicting labels' if conflict else 'exact duplicate'))
                remove.add(r['path'])
    rows = sorted((r for r in rows if r['path'] not in remove), key=lambda r: r['path'])
    seen = {}
    for r in rows:
        if r['leaf_group'] and seen.setdefault(r['leaf_group'], r['split']) != r['split']:
            raise ValueError(f"Leaf {r['leaf_group']} crosses a split")
    for split in ('pv_train', 'pv_test', 'pd_test'):
        assert len({r['label'] for r in rows if r['split'] == split}) == 9
    assert len({r['path'] for r in rows}) == len(rows)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'manifest.csv', rows)
    if excluded:
        write_csv(out / 'exclusions.csv', excluded)
    write_csv(out / 'class_mapping.csv', [dict(label=i, category=name, PlantVillage='Tomato___' + pv, PlantDoc=pd)
                                          for i, (name, pv, pd) in enumerate(LABELS)])
    print(Counter(r['split'] for r in rows), 'exclusions', len(excluded))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=Path('reconstructed'))
    p.add_argument('--skip-download', action='store_true')
    a = p.parse_args()
    if not a.skip_download:
        download(a.data)
    prepare(a.data, a.out)
