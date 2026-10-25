#!/usr/bin/env python3
"""Build the defense's two-panel paired-score contrast plot from frozen JSON."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/defense-score-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/trajectory-metrics/summary.json'
OUT = ROOT / 'output/final-presentation/figures'


def main():
    source_bytes = SOURCE.read_bytes()
    data = json.loads(source_bytes)
    contrasts = {r['a']: r for r in data['contrasts']
                 if r['suite'] == 'main' and r['b'] == 'mlp'}
    assert set(contrasts) == {'s4', 'whitened_mlp'}
    models = [('s4', 'S4', '#166a84'),
              ('whitened_mlp', 'Whitened MLP', '#b86525')]
    fields = [('fair_crps', 'Fair CRPS', (-0.020, 0.009), [-0.02, -0.01, 0]),
              ('variogram_fair', 'Fair variogram score', (-0.0062, 0.0016), [-0.006, -0.004, -0.002, 0])]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'axes.titleweight': 'bold', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), dpi=100)
    fig.patch.set_facecolor('#ffffff')
    fig.subplots_adjust(left=0.172, right=0.982, top=0.83, bottom=0.36, wspace=0.30)
    records = []
    for ax, (field, title, limits, ticks) in zip(axes, fields):
        ax.set_title(title, fontsize=19, pad=19, loc='left', color='#183348')
        ax.axvline(0, color='#506575', linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
        for row, (key, label, color) in enumerate(models):
            result = contrasts[key]['conditional_context_bootstrap'][field]
            mean = result['difference']; low, high = result['context_ci95']
            assert low <= mean <= high
            y = 1-row
            ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt='o',
                        markersize=10, color=color, elinewidth=3.2,
                        capsize=6, capthick=2, zorder=3)
            digits=4 if field=='fair_crps' else 5
            interval=f'{mean:+.{digits}f}  [{low:+.{digits}f}, {high:+.{digits}f}]'
            ax.text(0.03, y-0.245, interval, transform=ax.get_yaxis_transform(),
                    fontsize=11.5, color='#536573', va='center')
            records.append({'comparison': key+' minus mlp', 'metric': field,
                            'difference': mean, 'context_ci95': [low, high]})
        ax.set_xlim(*limits); ax.set_ylim(-0.45, 1.45)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: '0' if v == 0 else f'{v:.3f}'.rstrip('0').rstrip('.')))
        ax.set_yticks([1, 0], [x[1] for x in models] if ax is axes[0] else ['', ''])
        ax.tick_params(axis='y', length=0, pad=14, labelsize=16)
        ax.tick_params(axis='x', labelsize=13, pad=7, color='#a7b5bd')
        ax.spines['bottom'].set_color('#acbac2')
        ax.grid(axis='x', color='#e8edf0', linewidth=.8, zorder=0)
        ax.set_xlabel('Difference from MLP (lower is better)', fontsize=13, labelpad=11)
    fig.text(.172, .12, 'Negative values favor the comparison model.  Dashed zero = no change.',
             fontsize=13, color='#183348')
    fig.text(.172, .045,
             '95% paired-history bootstrap intervals; 256 histories, five fitted models per head held fixed. Exploratory analysis.',
             fontsize=11, color='#536573')
    OUT.mkdir(parents=True, exist_ok=True)
    output=OUT/'score_differences.png'
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    provenance={'scope':'Main GP-source frozen ensembles only; no training, new inference, checkpoint selection, or new success gate.',
                'source':str(SOURCE.relative_to(ROOT)),
                'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
                'script':str(Path(__file__).relative_to(ROOT)),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'output':str(output.relative_to(ROOT)),
                'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                'size_pixels':[1150,370],
                'direction':'comparison model minus MLP; negative favors the comparison model',
                'uncertainty':'Paired-context percentile bootstrap of 256 histories after averaging five fixed models per head; conditional on fitted models and saved ensemble draws; no multiplicity adjustment.',
                'warning':'Intervals spanning zero do not prove equivalence. S4 fair-VS upper endpoint remains positive; printed +0.00002, never rounded to zero.',
                'records':records}
    (OUT/'score_differences.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
