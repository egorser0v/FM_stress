#!/usr/bin/env python3
"""Visual inspection of archived main-study GP paths, with explicit selection.

No smoothing, interpolation or alignment is applied. Straight line segments
connect the saved samples. A generated path and an observed future are different
conditional realizations; their pointwise discrepancy is not a forecast score.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/roughness-posthoc'


def rough(x, order):
    return np.abs(np.diff(x, n=order, axis=-1)).mean(axis=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / 'results/main/samples/gp_s4_seed0.npz') as arrays:
        target, generated = arrays['target'], arrays['generated'].astype(float)
    with np.load(ROOT / 'results/main/samples/gp_mlp_seed0.npz') as arrays:
        assert np.array_equal(target, arrays['target'])
        mlp = arrays['generated'].astype(float)
    r1, r2 = rough(generated, 1) / rough(target, 1), rough(generated, 2) / rough(target, 2)
    eligible = np.flatnonzero((np.abs(r1 - 1) <= .05) & (r2 > 1.3)
                             & (rough(target, 2) >= np.quantile(rough(target, 2), .1)))
    ordered = eligible[np.argsort(r2[eligible])]
    # First stored path, plus moderate/strong examples selected by a disclosed rule.
    selected = [0] + [int(ordered[round(q * (len(ordered) - 1))]) for q in (.35, .65)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': .18,
                         'figure.facecolor': 'white'})
    colors = {'target': '#233f58', 's4': '#d66030', 'mlp': '#168878'}
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    for row, index in enumerate(selected):
        for col, order in enumerate((0, 1, 2)):
            ax = axes[row, col]
            for values, color, label in [(target[index], colors['target'], 'Наблюдаемый ряд'),
                                          (generated[index], colors['s4'], 'Генерация GP + S4')]:
                values = np.diff(values, n=order)
                ax.plot(np.arange(1, len(values) + 1), values, '-o', ms=3, lw=1.7,
                        color=color, label=label)
            if order:
                ax.axhline(0, color='#8a8f96', lw=.7)
            title = ['Траектория', 'Первая разность Δx', 'Вторая разность Δ²x'][col]
            ax.set_title(title)
            ax.set_xlabel('Шаг' if not order else 'Номер разности')
            if col == 0:
                ax.set_ylabel(f'Окно {index}\nR1: {r1[index]:.3f} · R2: {r2[index]:.3f}\nНормированные значения')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
    fig.suptitle('Почти одинаковые jumps, разная регулярность наклона', y=.99, fontsize=17, weight='bold')
    fig.text(.5, .025, 'Main, seed 0. R1/R2 в подписях — отношения к наблюдаемому ряду для данного окна.\n'
             'Окно 0 — первое в архиве; два других — выбранные иллюстрации, не случайная выборка.',
             ha='center', fontsize=9, color='#505660')
    fig.tight_layout(rect=(0, .07, 1, .91))
    fig.savefig(OUT / 'path_differences.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for index, ax in enumerate(axes.flat):
        for values, key, label in [(target, 'target', 'Наблюдаемый ряд'),
                                   (generated, 's4', 'GP + S4'), (mlp, 'mlp', 'GP + MLP')]:
            ax.plot(np.arange(1, 17), values[index], '-o', color=colors[key], lw=1.6, ms=2.5, label=label)
        ax.set_title(f'Окно {index}: S4 R1={r1[index]:.2f}, R2={r2[index]:.2f}')
        ax.set_xlabel('Шаг прогноза')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    fig.suptitle('Первые 6 окон без отбора по метрикам · main, seed 0', fontsize=16, weight='bold')
    fig.text(.5, .02, 'Показаны отдельные реализации; совпадение с конкретным будущим не требуется.',
             ha='center', fontsize=10, color='#505660')
    fig.tight_layout(rect=(0, .05, 1, .90))
    fig.savefig(OUT / 'first_six_paths.png', dpi=160)
    plt.close(fig)
    metadata = {'suite': 'main', 'source': 'gp', 'seed': 0, 'selected_indices': selected,
                'selection': 'First stored path plus 35%/65% R2 order statistics among eligible paths',
                'eligible_rule': 'abs(R1_ratio-1)<=0.05, R2_ratio>1.3, target_R2>=10th percentile',
                'eligible_count': len(eligible), 'total_paths': len(target),
                'examples': [{'index': i, 'r1_ratio': float(r1[i]), 'r2_ratio': float(r2[i]),
                              'target_R2': float(rough(target[i], 2)),
                              'generated_R2': float(rough(generated[i], 2))} for i in selected],
                'overview': 'First six saved paths without metric filtering',
                'plotting': 'Straight segments through saved samples; no smoothing or alignment'}
    (OUT / 'visual_selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
