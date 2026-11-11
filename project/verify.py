"""Audit the complete saved experiment without raw images or a GPU."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from prepare import LABELS


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def require(test, message):
    if not test:
        raise ValueError(message)


def audit(out):
    manifest = read(out / 'manifest.csv')
    paths = {r['path']: r for r in manifest}
    require(len(paths) == len(manifest), 'Duplicate manifest paths')
    require(set(r['split'] for r in manifest) == {'pv_train', 'pv_val', 'pv_test', 'pd_test'}, 'Unexpected splits')
    groups, hashes = defaultdict(set), defaultdict(set)
    for r in manifest:
        k = int(r['label'])
        require(0 <= k < 9 and r['category'] == LABELS[k][0], 'Invalid manifest class')
        require(r['domain'] == ('PlantDoc' if r['split'] == 'pd_test' else 'PlantVillage'), 'Domain/split mismatch')
        if r['domain'] == 'PlantVillage':
            require(r['upstream_split'] == ('test' if r['split'] == 'pv_test' else 'train'), 'Official source split changed')
        if r['leaf_group']:
            groups[r['leaf_group']].add(r['split'])
        hashes[r['sha256']].add(r['split'])
    require(all(len(s) == 1 for s in groups.values()), 'Leaf group crosses splits')
    require(all(len(s) == 1 for s in hashes.values()), 'Identical bytes cross splits')
    field_hashes = [r['sha256'] for r in manifest if r['split'] == 'pd_test']
    require(len(set(field_hashes)) == len(field_hashes), 'External exact duplicates remain')

    pred = read(out / 'predictions.csv')
    batches, seen = defaultdict(list), set()
    for r in pred:
        key = (r['condition'], r['path'])
        require(key not in seen, 'Duplicate prediction')
        seen.add(key)
        require(r['condition'] in {'ordinary', 'strong'}, 'Unexpected condition')
        require(r['split'] in {'pv_test', 'pd_test'}, 'Predictions must use test images')
        ref = paths.get(r['path'])
        require(ref is not None and ref['split'] == r['split'] and ref['label'] == r['y_true'], 'Prediction/manifest mismatch')
        p = [float(r[f'p_{k}']) for k in range(9)]
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in p), 'Invalid probability')
        require(abs(sum(p)-1) < 2e-6, 'Probabilities do not sum to one')
        require(int(r['y_pred']) == max(range(9), key=lambda k: p[k]), 'Prediction is not argmax')
        require(abs(float(r['confidence'])-max(p)) < 1e-7, 'Confidence mismatch')
        batches[r['condition'], r['split']].append(r)
    require(set(batches) == {(c,s) for c in ['ordinary','strong'] for s in ['pv_test','pd_test']}, 'Missing experimental condition or test set')
    summary = {(r['condition'],r['split']):r for r in read(out/'overall.csv')}
    metrics = []
    for key, batch in sorted(batches.items()):
        expected = {r['path'] for r in manifest if r['split'] == key[1]}
        require({r['path'] for r in batch} == expected, f'Incomplete test coverage: {key}')
        cm = [[0]*9 for _ in range(9)]
        for r in batch:
            cm[int(r['y_true'])][int(r['y_pred'])] += 1
        correct = sum(cm[k][k] for k in range(9))
        f1 = []
        for k in range(9):
            den = sum(cm[k])+sum(row[k] for row in cm)
            f1.append(2*cm[k][k]/den if den else 0.)
        accuracy, macro = correct/len(batch), sum(f1)/9
        saved = summary.get(key)
        require(saved is not None and int(saved['n']) == len(batch) and int(saved['correct']) == correct, 'Saved count mismatch')
        require(abs(float(saved['accuracy'])-accuracy) < 1e-12 and abs(float(saved['macro_f1'])-macro) < 1e-12, 'Saved metric mismatch')
        saved_cm = read(out/f'confusion_{key[0]}_{key[1]}.csv')
        require([[int(r[name]) for name,_,_ in LABELS] for r in saved_cm] == cm, 'Saved confusion mismatch')
        metrics.append(dict(condition=key[0], split=key[1], n=len(batch), correct=correct, accuracy=accuracy, macro_f1=macro))
    history = read(out/'history.csv')
    selected = {c:int(max((r for r in history if r['condition']==c), key=lambda r:float(r['val_supported_macro_f1']))['epoch']) for c in ['ordinary','strong']}
    return dict(status='PASS', manifest_images=len(manifest), prediction_rows=len(pred),
                split_counts=dict(Counter(r['split'] for r in manifest)), documented_leaf_groups=len(groups),
                selected_epochs=selected, metrics=metrics,
                scope='Saved-artifact integrity and independently recomputed metrics; raw-image replay is a separate check.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--json',type=Path)
    args = p.parse_args()
    report = audit(args.out)
    result = json.dumps(report,indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(result+'\n',encoding='utf-8')
    print(result)
