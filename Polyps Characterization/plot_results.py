"""Create publication-resolution PNG and PDF charts from saved run metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ['1', '2A', '2B', '3']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parent / 'outputs/ercpmp_jnet')
    args = parser.parse_args()
    results = json.loads((args.run / 'test_results.json').read_text())
    history = json.loads((args.run / 'history.json').read_text())
    out = args.run / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(out / f'{name}.png')

    cm = np.asarray(results['confusion_matrix'], dtype=int)
    report = results['patient_report']
    assert cm.shape == (4, 4) and cm.sum() == results['patients']
    assert np.array_equal(cm.sum(1), [report[c]['support'] for c in CLASSES])
    normalized = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) != 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
    for ax, values, title, percentage in zip(axes, [cm, normalized], ['Patient counts', 'Row-normalized (% of true class)'], [False, True]):
        plot = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if percentage else max(1, cm.max()))
        for i in range(4):
            for j in range(4):
                label = f'{values[i,j]:.1%}' if percentage else str(values[i,j])
                ax.text(j, i, label, ha='center', va='center', fontsize=14,
                        color='white' if values[i,j] > (0.5 if percentage else cm.max()/2) else '#172b4d')
        ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted JNET class', ylabel='True JNET class', title=title)
        fig.colorbar(plot, ax=ax, shrink=.75)
    fig.suptitle(f'ERCPMP — held-out patient confusion matrix\n{results["patients"]} patients / {results["images"]} images · Accuracy {report["accuracy"]:.1%} · Macro F1 {report["macro avg"]["f1-score"]:.3f}', fontsize=15)
    save(fig, 'confusion_matrix')

    epochs = [r['epoch'] for r in history]
    scores = [r['val_patient_macro_f1'] for r in history]
    best_index = int(np.argmax(scores))
    best = epochs[best_index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axes[0].plot(epochs, [r['loss'] for r in history], color='#147d92', linewidth=2, marker='.', markersize=7)
    axes[0].set(title='Training loss', xlabel='Epoch', ylabel='Cross-entropy loss', ylim=(0, None))
    axes[1].plot(epochs, scores, color='#7552a3', linewidth=2, marker='.', markersize=7)
    axes[1].scatter([best], [scores[best_index]], color='#db7b24', s=90, zorder=4, label=f'Selected epoch {best}: {scores[best_index]:.3f}')
    axes[1].set(title='Validation patient macro F1', xlabel='Epoch', ylabel='Macro F1', ylim=(0, 1))
    axes[1].legend(loc='upper left', frameon=False)
    for ax in axes:
        ax.axvline(best, color='#db7b24', linestyle='--', alpha=.7)
        ax.grid(alpha=.2)
    fig.suptitle('ERCPMP — training and checkpoint selection', fontsize=15)
    save(fig, 'training_curves')

    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    x = np.arange(4)
    for offset, metric, color in zip([-.25, 0, .25], ['precision', 'recall', 'f1-score'], ['#147d92', '#7552a3', '#db7b24']):
        bars = ax.bar(x+offset, [report[c][metric] for c in CLASSES], width=.24, label=metric.replace('f1-score', 'F1').capitalize(), color=color)
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=[f'JNET {c}\n(n={int(report[c]["support"])})' for c in CLASSES],
           ylim=(0, 1.2), yticks=np.arange(0, 1.01, .2), ylabel='Score',
           title='ERCPMP — held-out per-class patient performance')
    ax.legend(loc='upper right', ncols=3, frameon=False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    save(fig, 'per_class_metrics')

    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    labels = ['Model accuracy', 'Majority baseline\naccuracy', 'Model macro F1', 'Model balanced\naccuracy']
    vals = [report['accuracy'], results['majority_baseline_patient_accuracy'], report['macro avg']['f1-score'], report['macro avg']['recall']]
    bars = ax.bar(labels, vals, color=['#147d92', '#8c96a5', '#7552a3', '#db7b24'], width=.6)
    ax.bar_label(bars, labels=[f'{vals[0]:.1%}', f'{vals[1]:.1%}', f'{vals[2]:.3f}', f'{vals[3]:.1%}'], padding=5)
    ax.set(ylim=(0, 1), ylabel='Score (0–1)', title='ERCPMP — held-out patient results (n=13)')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    save(fig, 'test_summary')


if __name__ == '__main__':
    main()
