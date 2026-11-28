"""Metrics from saved per-image predictions, an independent audit, and the three-seed summary.

Usage: python analysis.py --results results   (recomputes every table; needs no raw images or GPU)
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from data import NAMES, read_csv

CONDITIONS, TESTS = ['ordinary', 'strong'], ['pv_test', 'pd_test']


def analyze(seed_dir):
    """Per-seed tables: overall.csv, per_class.csv, confusion matrices and a paired field bootstrap."""
    p = pd.read_csv(seed_dir / 'predictions.csv')
    ids = list(range(len(NAMES)))
    overall, classes = [], []
    for (condition, split), g in p.groupby(['condition', 'split']):
        truth, pred = g.y_true, g.y_pred
        f = f1_score(truth, pred, labels=ids, average=None, zero_division=0)
        rec = recall_score(truth, pred, labels=ids, average=None, zero_division=0)
        overall.append(dict(condition=condition, split=split, n=len(g), correct=int((truth == pred).sum()),
                            accuracy=accuracy_score(truth, pred), macro_f1=f.mean()))
        classes += [dict(condition=condition, split=split, category=name, support=int((truth == k).sum()),
                         recall=rec[k], f1=f[k]) for k, name in enumerate(NAMES)]
        pd.DataFrame(confusion_matrix(truth, pred, labels=ids), index=NAMES, columns=NAMES).rename_axis(
            'true_label').to_csv(seed_dir / f'confusion_{condition}_{split}.csv', lineterminator='\n')
    pd.DataFrame(overall).to_csv(seed_dir / 'overall.csv', index=False, lineterminator='\n')
    pd.DataFrame(classes).to_csv(seed_dir / 'per_class.csv', index=False, lineterminator='\n')
    # Strong-minus-ordinary field differences, resampling images within each true class.
    g = p[p.split == 'pd_test'].pivot(index=['path', 'y_true'], columns='condition', values='y_pred').reset_index()
    truth, a, b = g.y_true.to_numpy(), g.ordinary.to_numpy(), g.strong.to_numpy()
    rng = np.random.default_rng(664)
    groups = [np.flatnonzero(truth == k) for k in ids]
    boot = []
    for _ in range(2000):
        ix = np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in groups])
        y = truth[ix]

        def scores(pred):
            cm = np.bincount(9 * y + pred[ix], minlength=81).reshape(9, 9)
            return np.array([np.trace(cm) / len(y), np.mean(2 * cm.diagonal() / (cm.sum(0) + cm.sum(1)))])
        boot.append(scores(b) - scores(a))
    pd.DataFrame(np.quantile(boot, [.025, .975], axis=0), index=['p025', 'p975'],
                 columns=['accuracy_delta', 'macro_f1_delta']).to_csv(seed_dir / 'paired_field_bootstrap.csv', lineterminator='\n')


def require(test, message):
    if not test:
        raise ValueError(message)


def audit(seed_dir, manifest_path):
    """Check split integrity and recompute every saved metric in plain Python, independently of analyze()."""
    manifest = read_csv(manifest_path)
    paths = {r['path']: r for r in manifest}
    require(len(paths) == len(manifest), 'Duplicate manifest paths')
    require({r['split'] for r in manifest} == {'pv_train', 'pv_val', 'pv_test', 'pd_test'}, 'Unexpected splits')
    groups, hashes = defaultdict(set), defaultdict(set)
    for r in manifest:
        k = int(r['label'])
        require(0 <= k < 9 and r['category'] == NAMES[k], 'Invalid manifest class')
        require(r['domain'] == ('PlantDoc' if r['split'] == 'pd_test' else 'PlantVillage'), 'Domain/split mismatch')
        if r['domain'] == 'PlantVillage':
            require(r['upstream_split'] == ('test' if r['split'] == 'pv_test' else 'train'), 'Official split changed')
        if r['leaf_group']:
            groups[r['leaf_group']].add(r['split'])
        hashes[r['sha256']].add(r['split'])
    require(all(len(s) == 1 for s in groups.values()), 'Leaf group crosses splits')
    require(all(len(s) == 1 for s in hashes.values()), 'Identical bytes cross splits')
    field = [r['sha256'] for r in manifest if r['split'] == 'pd_test']
    require(len(set(field)) == len(field), 'External exact duplicates remain')

    pred = read_csv(seed_dir / 'predictions.csv')
    batches, seen = defaultdict(list), set()
    for r in pred:
        key = (r['condition'], r['path'])
        require(key not in seen, 'Duplicate prediction')
        seen.add(key)
        ref = paths.get(r['path'])
        require(r['condition'] in CONDITIONS and r['split'] in TESTS, 'Unexpected condition or split')
        require(ref is not None and ref['split'] == r['split'] and ref['label'] == r['y_true'], 'Prediction/manifest mismatch')
        p = [float(r[f'p_{k}']) for k in range(9)]
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in p), 'Invalid probability')
        require(abs(sum(p) - 1) < 2e-6, 'Probabilities do not sum to one')
        require(int(r['y_pred']) == max(range(9), key=lambda k: p[k]), 'Prediction is not argmax')
        require(abs(float(r['confidence']) - max(p)) < 1e-7, 'Confidence mismatch')
        batches[r['condition'], r['split']].append(r)
    require(set(batches) == {(c, s) for c in CONDITIONS for s in TESTS}, 'Missing condition or test set')
    saved = {(r['condition'], r['split']): r for r in read_csv(seed_dir / 'overall.csv')}
    for (condition, split), batch in sorted(batches.items()):
        require({r['path'] for r in batch} == {r['path'] for r in manifest if r['split'] == split}, 'Incomplete test coverage')
        cm = [[0] * 9 for _ in range(9)]
        for r in batch:
            cm[int(r['y_true'])][int(r['y_pred'])] += 1
        correct = sum(cm[k][k] for k in range(9))
        f1 = [2 * cm[k][k] / d if (d := sum(cm[k]) + sum(row[k] for row in cm)) else 0. for k in range(9)]
        s = saved.get((condition, split))
        require(s is not None and int(s['n']) == len(batch) and int(s['correct']) == correct, 'Saved count mismatch')
        require(abs(float(s['accuracy']) - correct / len(batch)) < 1e-12
                and abs(float(s['macro_f1']) - sum(f1) / 9) < 1e-12, 'Saved metric mismatch')
        saved_cm = read_csv(seed_dir / f'confusion_{condition}_{split}.csv')
        require([[int(r[name]) for name in NAMES] for r in saved_cm] == cm, 'Saved confusion mismatch')
    history = read_csv(seed_dir / 'history.csv')
    selected = {c: int(max((r for r in history if r['condition'] == c),
                           key=lambda r: float(r['val_supported_macro_f1']))['epoch']) for c in CONDITIONS}
    return dict(status='PASS', selected_epochs=selected)


def summarize(root):
    """Check the paired training budgets and the pre-test checkpoint lock, then aggregate the seeds."""
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    lock = json.loads((root / 'selection_locked.json').read_text(encoding='utf-8'))
    assert json.loads((root / 'run_complete.json').read_text(encoding='utf-8'))['status'] == 'PASS'
    manifest = pd.read_csv(root / 'manifest.csv')
    n_train = int((manifest.split == 'pv_train').sum())
    metrics, classes, histories, checks, predictions = [], [], [], [], []
    for seed in protocol['seeds']:
        folder = root / f'seed_{seed}'
        result = audit(folder, root / 'manifest.csv')
        history = pd.read_csv(folder / 'history.csv')
        histories.append(history.assign(seed=seed))
        a, b = (history[history.condition == c].sort_values('epoch') for c in CONDITIONS)
        assert a.epoch.tolist() == b.epoch.tolist() == list(range(1, protocol['epochs'] + 1))
        assert a.sample_order_sha256.tolist() == b.sample_order_sha256.tolist()
        assert a.optimizer_steps.tolist() == b.optimizer_steps.tolist()
        records = []
        for condition in CONDITIONS:
            done = json.loads((folder / f'{condition}_training_complete.json').read_text(encoding='utf-8'))
            assert done['optimizer_steps'] == a.optimizer_steps.sum() and done['frozen_prefix_unchanged']
            assert done['image_presentations'] == n_train * protocol['epochs']
            digest = hashlib.sha256((folder / f'{condition}.pt').read_bytes()).hexdigest()
            assert digest == lock['checkpoint_sha256'][f'{seed}/{condition}'], 'Checkpoint changed after the lock'
            records.append(done)
            checks.append(dict(seed=seed, condition=condition, optimizer_steps=done['optimizer_steps'],
                               image_presentations=done['image_presentations'],
                               selected_epoch=result['selected_epochs'][condition],
                               frozen_prefix_unchanged=True, checkpoint_sha256=digest))
        assert records[0]['initial_state_sha256'] == records[1]['initial_state_sha256']
        metrics.append(pd.read_csv(folder / 'overall.csv').assign(seed=seed))
        classes.append(pd.read_csv(folder / 'per_class.csv').assign(seed=seed))
        predictions.append(pd.read_csv(folder / 'predictions.csv').assign(seed=seed))
    m, c, h = (pd.concat(x, ignore_index=True) for x in (metrics, classes, histories))
    m.to_csv(root / 'per_seed_metrics.csv', index=False, lineterminator='\n')
    c.to_csv(root / 'per_seed_class_metrics.csv', index=False, lineterminator='\n')
    h.to_csv(root / 'all_history.csv', index=False, lineterminator='\n')
    summary = m.groupby(['condition', 'split']).agg(
        seeds=('seed', 'nunique'), accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std')).reset_index()
    summary.to_csv(root / 'summary.csv', index=False, lineterminator='\n')
    paired = []
    for seed in protocol['seeds']:
        for split in TESTS:
            s = m[(m.seed == seed) & (m.split == split)].set_index('condition')
            paired.append(dict(seed=seed, split=split,
                               accuracy_delta_pp=100 * (s.loc['strong', 'accuracy'] - s.loc['ordinary', 'accuracy']),
                               macro_f1_delta_pp=100 * (s.loc['strong', 'macro_f1'] - s.loc['ordinary', 'macro_f1'])))
    paired = pd.DataFrame(paired)
    paired.to_csv(root / 'paired_seed_deltas.csv', index=False, lineterminator='\n')
    effects = paired.groupby('split').agg(
        seeds=('seed', 'nunique'), accuracy_delta_mean_pp=('accuracy_delta_pp', 'mean'),
        accuracy_delta_std_pp=('accuracy_delta_pp', 'std'), macro_f1_delta_mean_pp=('macro_f1_delta_pp', 'mean'),
        macro_f1_delta_std_pp=('macro_f1_delta_pp', 'std')).reset_index()
    effects.to_csv(root / 'paired_seed_summary.csv', index=False, lineterminator='\n')
    wide = c.pivot(index=['seed', 'condition', 'category'], columns='split', values=['recall', 'f1'])
    for metric in ['recall', 'f1']:
        wide[(metric, 'gap_pp')] = 100 * (wide[(metric, 'pv_test')] - wide[(metric, 'pd_test')])
    wide.columns = ['_'.join(x) for x in wide.columns]
    wide.reset_index().to_csv(root / 'per_seed_class_gaps.csv', index=False, lineterminator='\n')
    class_summary = c.groupby(['condition', 'split', 'category']).agg(
        seeds=('seed', 'nunique'), support=('support', 'first'), recall_mean=('recall', 'mean'),
        recall_std=('recall', 'std'), f1_mean=('f1', 'mean'), f1_std=('f1', 'std')).reset_index()
    class_summary.to_csv(root / 'class_summary.csv', index=False, lineterminator='\n')
    pd.DataFrame(checks).to_csv(root / 'training_audit.csv', index=False, lineterminator='\n')

    # Reference floor: always predict the most common source-training label (no field label is used).
    majority = int(manifest.loc[manifest.split == 'pv_train', 'label'].value_counts().idxmax())
    baseline = []
    for split in TESTS:
        y = manifest.loc[manifest.split == split, 'label'].to_numpy()
        z = np.full(len(y), majority)
        baseline.append(dict(method='source_training_majority', class_index=majority, category=NAMES[majority],
                             split=split, n=len(y), accuracy=accuracy_score(y, z),
                             macro_f1=f1_score(y, z, labels=range(9), average='macro', zero_division=0)))
    pd.DataFrame(baseline).to_csv(root / 'majority_baseline.csv', index=False, lineterminator='\n')

    # Field confidence and prediction concentration.
    confidence, predicted = [], []
    for p in predictions:
        seed = int(p.seed.iloc[0])
        for condition in CONDITIONS:
            f = p[(p.condition == condition) & (p.split == 'pd_test')]
            high = f[f.confidence >= .9]
            confidence.append(dict(seed=seed, condition=condition, n=len(f), high_confidence_n=len(high),
                                   high_confidence_correct=int((high.y_true == high.y_pred).sum()),
                                   high_confidence_accuracy=float((high.y_true == high.y_pred).mean())))
            predicted += [dict(seed=seed, condition=condition, category=name, true_count=int((f.y_true == k).sum()),
                               predicted_count=int((f.y_pred == k).sum())) for k, name in enumerate(NAMES)]
    pd.DataFrame(confidence).to_csv(root / 'confidence_diagnostics.csv', index=False, lineterminator='\n')
    predicted = pd.DataFrame(predicted)
    predicted.to_csv(root / 'predicted_class_counts.csv', index=False, lineterminator='\n')

    n_field = int((manifest.split == 'pd_test').sum())
    errors = []
    for condition, label in [('ordinary', 'Ordinary'), ('strong', 'Strong')]:
        totals = predicted[predicted.condition == condition].groupby('category').agg(
            mean=('predicted_count', 'mean'), sd=('predicted_count', 'std'), true=('true_count', 'first'))
        name = totals['mean'].idxmax()
        r = totals.loc[name]
        errors.append(f"{label} training most often predicted {name.lower()}, assigning that label to {r['mean']:.1f} "
                      f"of {n_field} field images on average (SD {r['sd']:.1f}); its true support was {int(r['true'])}.")
    field_recall = class_summary[class_summary.split == 'pd_test'].pivot(index='category', columns='condition',
                                                                         values='recall_mean')
    change = 100 * (field_recall['strong'] - field_recall['ordinary'])
    errors.append(f"The largest signed mean class-recall change was {change.max():+.1f} pp for {change.idxmax().lower()}, "
                  f"and the smallest was {change.min():+.1f} pp for {change.idxmin().lower()}.")
    field_delta, e = paired[paired.split == 'pd_test'], effects.set_index('split').loc['pd_test']
    if (field_delta.accuracy_delta_pp > 0).all() and (field_delta.macro_f1_delta_pp > 0).all():
        conclusion = ('Strong augmentation improved both field metrics in all three fitted seed pairs, '
                      'while a substantial clean-to-field gap remained.')
    elif e.accuracy_delta_mean_pp > 0 and e.macro_f1_delta_mean_pp > 0:
        conclusion = ('Strong augmentation improved both field metrics on average, '
                      'but the paired improvements were not consistent across all three seeds.')
    else:
        conclusion = ('Strong augmentation did not improve both average field metrics under the fresh-view protocol, '
                      'and the seed-wise results limit any general claim of benefit.')
    (root / 'interpretation.json').write_text(json.dumps(dict(conclusion=conclusion, field_error_summary=errors),
                                                         indent=2) + '\n', encoding='utf-8', newline='\n')

    events = [json.loads(line) for line in (root / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    epochs = {(x['seed'], x['condition'], x['epoch']) for x in events if x['event'] == 'epoch_complete'}
    assert len(epochs) == len(protocol['seeds']) * len(CONDITIONS) * protocol['epochs']
    locked = next(i for i, x in enumerate(events) if x['event'] == 'all_checkpoints_locked_before_test_evaluation')
    assert all(i > locked for i, x in enumerate(events) if x['event'] == 'test_predictions_saved')
    record = dict(status='PASS', seeds=protocol['seeds'], trained_models=len(checks), unique_completed_epochs=len(epochs),
                  optimizer_steps=sum(x['optimizer_steps'] for x in checks),
                  total_image_presentations=sum(x['image_presentations'] for x in checks),
                  identical_initialization_within_seed=True, identical_sample_order_within_seed=True,
                  all_six_checkpoints_locked_before_test_predictions=True, original_split_preserved=True,
                  all_per_seed_artifact_audits_passed=True,
                  interpretation='Mean and sample SD describe three fitted seeds on the same test images, '
                                 'not three independent test datasets.')
    (root / 'aggregate_verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(conclusion)
    return summary, effects


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=Path('results'))
    root = p.parse_args().results
    for seed in json.loads((root / 'protocol.json').read_text(encoding='utf-8'))['seeds']:
        analyze(root / f'seed_{seed}')
    summarize(root)
