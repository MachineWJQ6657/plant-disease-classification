"""Report figures (PDF) and LaTeX tables/macros, generated from the saved results.

Usage: python figures.py --results results --dest ../report/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from data import NAMES

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SHORT = ['Healthy', 'Early blight', 'Bacterial spot', 'Late blight', 'Mosaic', 'Yellow curl', 'Leaf mold', 'Septoria',
         'Spider mites']
LABEL = {'ordinary': 'Ordinary', 'strong': 'Crop + color'}
CELLS = [('pv_test', 'accuracy'), ('pd_test', 'accuracy'), ('pv_test', 'macro_f1'), ('pd_test', 'macro_f1')]


def row(values):
    return ' & '.join(str(v) for v in values) + r' \\'


def write(path, lines):
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def build(root, dest):
    dest.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / 'summary.csv').set_index(['condition', 'split'])
    per_seed = pd.read_csv(root / 'per_seed_metrics.csv')
    classes = pd.read_csv(root / 'class_summary.csv').set_index(['condition', 'split', 'category'])
    history = pd.read_csv(root / 'all_history.csv')
    effects = pd.read_csv(root / 'paired_seed_summary.csv').set_index('split')
    deltas = pd.read_csv(root / 'paired_seed_deltas.csv')
    audit = pd.read_csv(root / 'training_audit.csv').set_index(['seed', 'condition'])
    baseline = pd.read_csv(root / 'majority_baseline.csv').set_index('split')
    confidence = pd.read_csv(root / 'confidence_diagnostics.csv')
    manifest = pd.read_csv(root / 'manifest.csv')
    interpretation = json.loads((root / 'interpretation.json').read_text(encoding='utf-8'))
    seeds = sorted(per_seed.seed.unique())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), layout='constrained')
    for ax, metric, title in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
        for j, (condition, color) in enumerate([('ordinary', '#555555'), ('strong', '#BBBBBB')]):
            means = [100 * summary.loc[(condition, s), metric + '_mean'] for s in ['pv_test', 'pd_test']]
            stds = [100 * summary.loc[(condition, s), metric + '_std'] for s in ['pv_test', 'pd_test']]
            positions = np.arange(2) + (j - .5) * .32
            bars = ax.bar(positions, means, .32, color=color, edgecolor='black', linewidth=.5, yerr=stds, capsize=4,
                          label=LABEL[condition])
            ax.bar_label(bars, fmt='%.1f', padding=7, fontsize=8)
            for pos, split in zip(positions, ['pv_test', 'pd_test']):
                values = per_seed[(per_seed.condition == condition) & (per_seed.split == split)].sort_values('seed')
                ax.scatter(pos + np.array([-.045, 0, .045]), 100 * values[metric].to_numpy(), s=15, c='black', zorder=3)
        ax.set(xticks=[0, 1], xticklabels=['PlantVillage', 'PlantDoc'], ylim=(0, 111), ylabel=title + ' (%)')
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig(dest / 'performance.pdf', dpi=180, metadata={'CreationDate': None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3), layout='constrained')
    for condition, color in [('ordinary', '#333333'), ('strong', '#999999')]:
        h = history[history.condition == condition]
        for ax, metric, label in zip(axes, ['train_loss', 'val_supported_macro_f1'],
                                     ['Training cross-entropy', 'Validation macro-F1 (%)']):
            scale = 100 if metric == 'val_supported_macro_f1' else 1
            g = h.groupby('epoch')[metric].agg(['mean', 'std'])
            x, mean, std = g.index.to_numpy(), scale * g['mean'].to_numpy(), scale * g['std'].to_numpy()
            ax.plot(x, mean, 'o-' if condition == 'ordinary' else 's--', color=color, label=condition)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=.12)
            ax.set(xlabel='Epoch', ylabel=label, xticks=range(1, 9))
    axes[0].legend(frameon=False)
    fig.savefig(dest / 'history.pdf', dpi=180, metadata={'CreationDate': None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
    for ax, condition in zip(axes, ['ordinary', 'strong']):
        cm = pd.read_csv(root / 'seed_664' / f'confusion_{condition}_pd_test.csv', index_col=0).to_numpy()
        share = cm / cm.sum(1, keepdims=True)
        ax.imshow(share, vmin=0, vmax=1, cmap='Greys')
        for i in range(9):
            for j in range(9):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                        color='white' if share[i, j] > .5 else 'black')
        ax.set(xticks=range(9), yticks=range(9), xticklabels=SHORT, yticklabels=SHORT, title=condition,
               xlabel='Predicted class', ylabel='True class')
        plt.setp(ax.get_xticklabels(), rotation=55, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.savefig(dest / 'confusions.pdf', dpi=180, metadata={'CreationDate': None})
    plt.close(fig)

    counts = pd.crosstab(manifest.category, manifest.split)
    write(dest / 'counts.tex', [row([name] + [f'{counts.loc[name, s]:,}' for s in ['pv_train', 'pv_val', 'pv_test', 'pd_test']])
                                for name in NAMES])
    write(dest / 'summary.tex', [row([LABEL[c]] + [f"${100 * summary.loc[(c, s), m + '_mean']:.2f} \\pm "
                                                   f"{100 * summary.loc[(c, s), m + '_std']:.2f}$" for s, m in CELLS])
                                 for c in ['ordinary', 'strong']])
    by_seed = per_seed.set_index(['seed', 'condition', 'split'])
    write(dest / 'seeds.tex', [row([seed, LABEL[c], int(audit.loc[(seed, c), 'selected_epoch'])]
                                   + [f'{100 * by_seed.loc[(seed, c, s), m]:.2f}' for s, m in CELLS])
                               for seed in seeds for c in ['ordinary', 'strong']])
    write(dest / 'deltas.tex', [row([int(r.seed), f'{r.accuracy_delta_pp:+.2f}', f'{r.macro_f1_delta_pp:+.2f}'])
                                for r in deltas[deltas.split == 'pd_test'].sort_values('seed').itertuples()])
    recall = []
    for name in NAMES:
        values = [name, int(classes.loc[('ordinary', 'pd_test', name), 'support'])]
        for c in ['ordinary', 'strong']:
            clean, field = (100 * classes.loc[(c, s, name), 'recall_mean'] for s in ['pv_test', 'pd_test'])
            values += [f'{clean:.1f}', f'{field:.1f}', f'{clean - field:.1f}']
        recall.append(row(values))
    write(dest / 'recall.tex', recall)
    write(dest / 'confidence.tex', [row([r.seed, LABEL[r.condition], r.high_confidence_n, r.high_confidence_correct,
                                         f'{100 * r.high_confidence_accuracy:.2f}']) for r in confidence.itertuples()])
    write(dest / 'errors.tex', [' '.join(interpretation['field_error_summary'])])

    macros = {}
    for c, prefix in [('ordinary', 'Ord'), ('strong', 'Aug')]:
        for s, domain in [('pv_test', 'Clean'), ('pd_test', 'Field')]:
            for m, label in [('accuracy', 'Acc'), ('macro_f1', 'Fone')]:
                macros[f'{prefix}{domain}{label}'] = f"{100 * summary.loc[(c, s), m + '_mean']:.2f}"
                macros[f'{prefix}{domain}{label}SD'] = f"{100 * summary.loc[(c, s), m + '_std']:.2f}"
    for m, label in [('accuracy', 'Acc'), ('macro_f1', 'Fone')]:
        macros[f'Delta{label}'] = f"{effects.loc['pd_test', m + '_delta_mean_pp']:+.2f}"
        macros[f'Delta{label}SD'] = f"{effects.loc['pd_test', m + '_delta_std_pp']:.2f}"
    macros['BaseClass'] = baseline.loc['pv_test', 'category'].lower()
    for s, domain in [('pv_test', 'Clean'), ('pd_test', 'Field')]:
        macros[f'Base{domain}Acc'] = f"{100 * baseline.loc[s, 'accuracy']:.2f}"
        macros[f'Base{domain}Fone'] = f"{100 * baseline.loc[s, 'macro_f1']:.2f}"
    macros['Conclusion'] = interpretation['conclusion']
    demo = pd.read_csv(root / 'seed_664' / 'predictions.csv')
    credits = pd.read_csv(Path(__file__).resolve().parent / 'examples' / 'credits.csv')
    for filename, suffix in [('field.jpg', 'One'), ('field_second.jpg', 'Two')]:
        path = credits.loc[credits['use'] == filename, 'path'].iloc[0]
        for c, prefix in [('ordinary', 'Ord'), ('strong', 'Aug')]:
            pred = demo[(demo.path == path) & (demo.condition == c)].y_pred.iloc[0]
            macros[f'Demo{prefix}{suffix}'] = NAMES[int(pred)].lower()
    write(dest / 'numbers.tex', ['\\newcommand{\\' + k + '}{' + v + '}' for k, v in macros.items()])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=Path('results'))
    p.add_argument('--dest', type=Path, default=Path('../report/figures'))
    a = p.parse_args()
    build(a.results, a.dest)
