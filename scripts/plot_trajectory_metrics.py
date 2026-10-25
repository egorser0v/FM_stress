#!/usr/bin/env python3
"""Plot canonical archived ensemble metrics; ranges describe saved runs, not CIs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    'main': [('mlp', 'MLP'), ('s4', 'S4'), ('whitened_mlp', 'Whitened MLP')],
    'followup-capacity': [('mlp_original', 'Original MLP'),
                          ('mlp_rank_clear', 'Rank-clear MLP'),
                          ('mlp_wide', 'Wide MLP'), ('s4', 'S4')],
}
COLORS = ['#446B9E', '#D87532', '#25847A', '#8B669C']
METRICS = [('R1_ratio', 'R1: mean absolute first difference', 'Generated / observed target'),
           ('R2_ratio', 'R2: mean absolute second difference', 'Generated / observed target'),
           ('variogram_fair', 'Fair Variogram Score', 'VS (p = 0.5); lower is better')]


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(summary, suite, output):
    ordered = STUDIES[suite]
    groups = {(g['suite'], g['group']): g for g in summary['groups']}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    for ax, (metric, title, xlabel) in zip(axes, METRICS):
        for index, (group, label) in enumerate(ordered):
            record = groups[(suite, group)]
            values = [r['means'][metric] for r in summary['runs'] if r['suite'] == suite and r['group'] == group]
            if len(values) != record['n_runs']:
                raise ValueError('Canonical run count differs from group count')
            mean, lo, hi = record['means'][metric], min(values), max(values)
            ax.plot([lo, hi], [index, index], color=COLORS[index], lw=2, alpha=.72, zorder=2)
            ax.plot([lo, lo], [index-.07, index+.07], color=COLORS[index], lw=1.3)
            ax.plot([hi, hi], [index-.07, index+.07], color=COLORS[index], lw=1.3)
            ax.scatter(values, index+np.linspace(-.11, .11, len(values)), s=22,
                       color=COLORS[index], alpha=.6, edgecolors='none', zorder=3)
            ax.scatter([mean], [index], s=64, marker='D', color=COLORS[index],
                       edgecolors='white', linewidths=.9, zorder=4)
            ax.annotate(f'{mean:.3f}' if metric != 'variogram_fair' else f'{mean:.5f}',
                        (mean, index), xytext=(0, 12), textcoords='offset points',
                        ha='center', fontsize=9, color='#263445', weight='bold')
        if metric.endswith('_ratio'):
            ax.axvline(1., color='#65707C', lw=1, ls=(0, (3, 3)), zorder=1)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_xlabel(xlabel, fontsize=10, labelpad=9)
        ax.grid(axis='x', color='#DEE3E8', lw=.8)
        ax.set_axisbelow(True)
        ax.set_ylim(len(ordered)-.6, -.6)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.tick_params(axis='x', labelsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#AAB3BC')
        ax.margins(x=.18)
    axes[0].set_yticks(range(len(ordered)), [label for group, label in ordered], fontsize=10)
    if suite == 'followup-capacity':
        axes[1].set_xscale('log')
        axes[1].set_xlim(.8, 30)
        axes[1].xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
        axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        axes[1].xaxis.set_minor_locator(FixedLocator([]))
        axes[1].set_xlabel('Generated / observed target (log scale)', fontsize=10, labelpad=9)
        title = 'Capacity follow-up: large changes in trajectory regularity'
        subtitle = 'RBF F64, white source · 256 contexts × 32 members per run · 3 independent paired data-draw / training-seed replicates'
        stem = 'capacity_trajectory_metrics'
    else:
        axes[0].set_xlim(.94, 1.12)
        axes[1].set_xlim(.85, 1.85)
        title = 'Main experiment: first differences conceal second-difference changes'
        subtitle = 'GP F16 · 256 shared test contexts × 64 members · 5 training seeds on the same test draw'
        stem = 'main_trajectory_metrics'
    fig.suptitle(title, x=.16, y=.975, ha='left', fontsize=15, weight='bold', color='#24364A')
    fig.text(.16, .88, subtitle, fontsize=9, ha='left', color='#526272')
    legend = [Line2D([], [], marker='D', color='none', markerfacecolor='#446B9E', markeredgecolor='white', markersize=7, label='Group mean'),
              Line2D([], [], marker='o', color='none', markerfacecolor='#8A9CAD', markersize=5, label='Individual saved runs'),
              Line2D([], [], color='#7E8C98', lw=2, marker='|', markersize=7, label='Min–max across runs (descriptive; not a CI)')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.56, .025), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.16, right=.98, top=.73, bottom=.25, wspace=.30)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'{stem}.{extension}', dpi=200, facecolor='white',
                    metadata={'Creator': 'scripts/plot_trajectory_metrics.py'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT/'results/trajectory-metrics/summary.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/trajectory-metrics')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 200})
    for suite in STUDIES:
        draw(summary, suite, args.output)
    rows = []
    table = ['| Suite / model | R1 ratio | R2 ratio | Fair VS | Fair CRPS | Fair energy |',
             '|---|---:|---:|---:|---:|---:|']
    for suite, ordered in STUDIES.items():
        for key, label in ordered:
            group = next(g for g in summary['groups'] if g['suite'] == suite and g['group'] == key)
            row = {'suite': suite, 'model': key, 'runs': group['n_runs']}
            for metric in ('R1_ratio', 'R2_ratio', 'variogram_fair', 'fair_crps', 'fair_energy_score'):
                row[metric] = group['means'][metric]
                row[metric+'_run_min'] = group['run_min'][metric]
                row[metric+'_run_max'] = group['run_max'][metric]
            rows.append(row)
            table.append(f"| {suite} / {label} | "+' | '.join(f"{row[k]:.6f}" for k in
                         ('R1_ratio', 'R2_ratio', 'variogram_fair', 'fair_crps', 'fair_energy_score'))+' |')
    with (args.output/'metrics_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'metrics_table.md').write_text('\n'.join(table)+'\n\nGroup ratios are ratios of aggregated generated and target means. Ranges in CSV and figures describe runs; they are not confidence intervals.\n')
    groups = {(g['suite'], g['group']): g['means'] for g in summary['groups']}
    main_mlp, main_s4 = groups[('main', 'mlp')], groups[('main', 's4')]
    original, wide = groups[('followup-capacity', 'mlp_original')], groups[('followup-capacity', 'mlp_wide')]
    readme = f'''# Постфактум: регулярность траекторий и Variogram Score

Два рисунка построены только из `results/trajectory-metrics/summary.json`, без
новой генерации. PNG — 200 dpi; PDF — векторный, с внедрёнными шрифтами.
`metrics_table.csv` сохраняет полную точность чисел и диапазоны; Markdown-таблица
даёт удобное представление средних. Скрипт: `scripts/plot_trajectory_metrics.py`.

Каждая панель использует ансамблевую когорту из 256 контекстов: main — 64 члена,
capacity — 32. Отдельные 1024 одиночные траектории follow-up сюда не входят.
Ромбы — средние группы; точки — отдельные запуски; отрезки — min–max запусков,
**не доверительные интервалы**. Main: пять обучающих seed на одном тестовом
наборе. Capacity: три независимых парных реплики «набор данных + seed обучения».
R1/R2 — отношения агрегированной сгенерированной и наблюдаемой шероховатости.
Вертикаль 1 означает совпадение одного момента, а не качество всего прогноза.
На capacity-панели R2 ось логарифмическая, чтобы сохранить видимость всех моделей.

В main R2 у S4 равен {main_s4['R2_ratio']:.3f}, у MLP — {main_mlp['R2_ratio']:.3f},
хотя R1 близки к 1. Это свидетельствует о различии вторых разностей, но **не
устанавливает, что S4 хуже по Variogram Score**: fair VS у S4
{main_s4['variogram_fair']:.6f} против {main_mlp['variogram_fair']:.6f} у MLP
(меньше — лучше). Условный bootstrap контрастов и его ограничения приведены
в каноническом JSON и `docs/trajectory_metrics.md`; диапазоны на рисунке его не заменяют.

В F64 widening снижает R2 ratio с {original['R2_ratio']:.3f} до {wide['R2_ratio']:.3f}
({100*(1-wide['R2_ratio']/original['R2_ratio']):.1f}%); fair VS — с {original['variogram_fair']:.6f}
до {wide['variogram_fair']:.6f} ({100*(1-wide['variogram_fair']/original['variogram_fair']):.1f}%).
При этом fair CRPS меняется с {original['fair_crps']:.6f} до {wide['fair_crps']:.6f}
(лишь {100*(1-wide['fair_crps']/original['fair_crps']):.2f}%). Большое изменение геометрии траекторий
не означает столь же большого улучшения маргинального вероятностного прогноза.
Абсолютные VS между main и capacity не сравниваются: горизонты и ансамбли различаются.
'''
    (args.output/'README.md').write_text(readme)
    manifest = {'source_summary': str(args.summary.relative_to(ROOT)), 'source_sha256': hash_file(args.summary),
                'script_sha256': hash_file(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
                'numpy_version': np.__version__, 'python_version': sys.version.split()[0],
                'cohort': '256-context ensembles; main64 members, capacity32 members',
                'ranges': 'Saved-run min/max, descriptive; not confidence intervals',
                'outputs_sha256': {p.name: hash_file(p) for p in sorted(args.output.iterdir())
                                   if p.is_file() and p.name != 'manifest.json'}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(f'Wrote two figures (PNG + PDF), native tables and Russian interpretation to {args.output}')


if __name__ == '__main__':
    main()
