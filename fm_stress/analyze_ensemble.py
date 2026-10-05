"""Read-only tables and figures for the complete prospective ensemble evaluation.

No outputs are authored until all 15 declared checkpoints and their matching
fingerprints/sample archives are available. Does not alter evaluation records.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analyze import setup_style

LABELS = {'mlp': 'MLP', 's4': 'S4', 'whitened_mlp': 'Whitened MLP'}
COLORS = {'mlp': '#186c9f', 's4': '#c6643d', 'whitened_mlp': '#6b8138'}
METRIC_LABELS = {'fair_crps': 'Fair marginal CRPS', 'fair_energy_score': 'Fair path energy score'}


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, stem):
    for ext in ('png', 'svg'):
        fig.savefig(out/f'{stem}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_complete(results):
    root=Path(results)
    cfg, manifest, summary = (read(root/name) for name in ('config.json','manifest.json','summary.json'))
    expected={(h,s) for h in cfg['heads'] for s in cfg['seeds']}
    if len(expected)!=15 or cfg['n_histories']!=256 or cfg['ensemble_size']!=64:
        raise ValueError('This final analysis requires the declared 15-run H256/S64 evaluation')
    records=[read(p) for p in sorted((root/'runs').glob('*.json'))]
    observed={(r['head'],r['seed']) for r in records}
    fingerprint=manifest['fingerprint']
    if not summary['complete'] or observed!=expected or len(records)!=len(expected):
        raise ValueError('All 15 distinct declared evaluations are required')
    if summary['fingerprint']!=fingerprint or any(r['fingerprint']!=fingerprint for r in records):
        raise ValueError('Mixed evaluation fingerprints')
    for r in records:
        sample=root/'samples'/f"{r['name']}.npz"
        if not sample.exists():
            raise ValueError(f'Missing ensemble {sample}')
        with np.load(sample) as archive:
            if archive['generated'].shape!=(cfg['n_histories'],cfg['ensemble_size'],16):
                raise ValueError(f'Unexpected ensemble shape: {sample}')
        if any(len(v)!=cfg['n_histories'] for v in r['per_context'].values()):
            raise ValueError('Incorrect context score count')
    reference=read(root/'privileged_reference.json') if (root/'privileged_reference.json').exists() else None
    if reference is not None and reference['fingerprint']!=fingerprint:
        raise ValueError('Reference fingerprint mismatch')
    return cfg, manifest, summary, records, reference


def tables(out, summary, records, reference):
    csv_write(out/'per_seed_metrics.csv', [dict(head=r['head'],seed=r['seed'],device=r['device'],
        checkpoint_step=r['checkpoint_step'],seconds=r['seconds'],**r['means']) for r in records])
    rows=[]
    for head,entry in summary['heads'].items():
        for metric,values in entry['metrics'].items():
            rows.append(dict(head=head,label=LABELS[head],metric=metric,n_seeds=entry['n_seeds'],**values))
    csv_write(out/'summary_metrics.csv',rows)
    csv_write(out/'paired_context_contrasts.csv',summary['contrasts'])
    if reference is not None:
        csv_write(out/'privileged_reference_metrics.csv',[dict(metric=k,value=v,
            information='Exact conditional GP draws given RAW history; different information') for k,v in reference['means'].items()])


def proper_scores(out, cfg, summary, records, reference):
    fig, axes=plt.subplots(1,2,figsize=(11.4,4.2))
    for ax,metric in zip(axes,('fair_crps','fair_energy_score')):
        all_values=[]
        for i,head in enumerate(cfg['heads']):
            values=np.array([r['means'][metric] for r in sorted(records,key=lambda r:r['seed']) if r['head']==head])
            all_values.extend(values)
            offsets=np.linspace(-.12,.12,len(values))
            ax.scatter(i+offsets,values,s=32,c=COLORS[head],alpha=.75,zorder=3)
            ax.plot([i-.23,i+.23],[values.mean()]*2,c=COLORS[head],lw=3,zorder=4)
        if reference is not None:
            v=reference['means'][metric]
            all_values.append(v)
            ax.axhline(v,color='#64747C',ls='--',lw=1.3,label='Privileged GP reference (raw history)')
            ax.legend(frameon=False,fontsize=8,loc='upper left')
        ax.set_xticks(range(3),[LABELS[h] for h in cfg['heads']])
        ax.set_ylabel(METRIC_LABELS[metric]+' ↓')
        ax.set_ylim(0,max(all_values)*1.25)
        ax.grid(axis='y',alpha=.7)
        ax.set_title(METRIC_LABELS[metric])
    fig.suptitle('Conditional forecast scores on 256 fresh histories',y=1.03)
    fig.text(.5,-.025,'Dots: five trained seeds; thick line: their mean. 64 samples per history, Euler-128. Lower is better.',ha='center',fontsize=9)
    fig.tight_layout()
    save(fig,out,'proper_scores')


def coverage_table(out, cfg, summary, reference):
    columns=['Head / reference','80% coverage','80% width','90% coverage','90% width','Ensemble variance']
    metrics=['coverage_80','width_80','coverage_90','width_90','ensemble_variance']
    rows=[]
    for head in cfg['heads']:
        entry=summary['heads'][head]['metrics']
        vals=[]
        for metric in metrics:
            mean,sd=entry[metric]['mean'],entry[metric]['seed_sd']
            vals.append(f'{100*mean:.1f}% ± {100*sd:.1f}' if metric.startswith('coverage') else f'{mean:.3f} ± {sd:.3f}')
        rows.append([LABELS[head],*vals])
    if reference is not None:
        rows.append(['Privileged GP*',*[f"{100*reference['means'][m]:.1f}%" if m.startswith('coverage') else f"{reference['means'][m]:.3f}" for m in metrics]])
    fig,ax=plt.subplots(figsize=(12.7,3.3));ax.axis('off')
    table=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center',colWidths=[.19,.165,.145,.165,.145,.19])
    table.auto_set_font_size(False);table.set_fontsize(9.4);table.scale(1,2.0)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D4DEE2')
        if row==0:
            cell.set_facecolor('#E6EFF1');cell.set_text_props(weight='bold',color='#20333F')
        elif row%2==0: cell.set_facecolor('#F6F8F9')
    ax.set_title('Marginal interval calibration and spread',weight='bold',pad=13)
    fig.text(.5,.045,'Mean ± SD across five trained seeds. Coverage: averaged over histories and forecast coordinates; width in normalized units.',ha='center',fontsize=8.5)
    fig.text(.5,-.005,'* Exact conditional reference has raw-history information unavailable to the learned heads. It is not a finite-sample score bound.',ha='center',fontsize=8.5)
    save(fig,out,'coverage_width_table')


def paths(out, root, cfg, reference=False):
    inputs=np.load(root/'common_inputs.npz'); target=inputs['target']
    heads=['privileged_reference'] if reference else cfg['heads']
    ensembles={head:np.load(root/'samples'/(f'{head}.npz' if reference else f'gp_{head}_seed0.npz'))['generated'] for head in heads}
    fig,axes=plt.subplots(3,len(heads),figsize=(11.8 if not reference else 5.7,8.1),squeeze=False)
    t=np.arange(1,target.shape[1]+1)
    for i in range(3):
        quantiles={h:np.quantile(x[i],[.05,.1,.5,.9,.95],axis=0,method='linear') for h,x in ensembles.items()}
        ymin=min(float(target[i].min()),min(float(q[0].min()) for q in quantiles.values()))
        ymax=max(float(target[i].max()),max(float(q[-1].max()) for q in quantiles.values()))
        margin=.08*max(ymax-ymin,.1)
        for j,head in enumerate(heads):
            ax=axes[i,j];q=quantiles[head];c='#64747C' if reference else COLORS[head]
            ax.fill_between(t,q[0],q[4],color=c,alpha=.13,label='90% interval')
            ax.fill_between(t,q[1],q[3],color=c,alpha=.25,label='80% interval')
            ax.plot(t,q[2],color=c,lw=1.8,label='Predictive median')
            ax.plot(t,target[i],color='#172D37',lw=1.5,label='Observed future')
            ax.set_ylim(ymin-margin,ymax+margin);ax.set_xlim(1,len(t));ax.grid(alpha=.4)
            if i==0: ax.set_title('Privileged exact GP' if reference else LABELS[head])
            if j==0: ax.set_ylabel(f'History {i}\nNormalized value')
            if i==2: ax.set_xlabel('Forecast step')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2 if reference else 4,frameon=False,fontsize=9)
    title='Privileged raw-history reference: first three fresh contexts' if reference else 'First three fresh contexts: seed 0, fixed before evaluation'
    fig.suptitle(title,fontsize=13,weight='bold',y=.995)
    fig.tight_layout(rect=(0,.065,1,.975))
    save(fig,out,'privileged_reference_paths' if reference else 'fixed_context_forecasts')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/ensemble-evaluation')
    parser.add_argument('--output',default='output/ensemble-analysis')
    args=parser.parse_args();root=Path(args.results);out=Path(args.output)
    cfg,manifest,summary,records,reference=load_complete(root)
    out.mkdir(parents=True,exist_ok=True);setup_style()
    tables(out,summary,records,reference)
    proper_scores(out,cfg,summary,records,reference)
    coverage_table(out,cfg,summary,reference)
    paths(out,root,cfg)
    if reference is not None: paths(out,root,cfg,reference=True)
    (out/'analysis_status.json').write_text(json.dumps(dict(complete=True,observed_runs=len(records),
        fingerprint=manifest['fingerprint'],results=str(root),n_histories=cfg['n_histories'],
        ensemble_size=cfg['ensemble_size'],euler_steps=cfg['euler_steps']),indent=2)+'\n')
    (out/'README.md').write_text('''# Conditional ensemble evaluation figures

All 15 original GP checkpoints are included. Each shares 256 new histories and 64
source draws per history; Euler-128 is fixed. Scores and intervals are supplemental.
The main experiment and its success rule are unchanged.

- `per_seed_metrics.csv`: every checkpoint's mean metrics and runtime.
- `summary_metrics.csv`: metric means and SD across the five fixed trained seeds.
- `paired_context_contrasts.csv`: S4−MLP and whitened MLP−MLP, paired-history
  bootstrap after averaging context scores across training seeds. Intervals are
  conditional on these trained models and ensemble draws, with no multiplicity
  correction. Histories—not coordinates or ensemble members—are sampling units.
- `proper_scores`: CRPS and energy score; individual seed values and head means.
- `coverage_width_table`: marginal interval calibration and spread.
- `fixed_context_forecasts`: first three fresh contexts, original seed 0, with
  consistent y axes across heads within each row. Bands are marginal intervals;
  the single observed future need not equal the predictive median.
- `privileged_reference_metrics.csv` and `privileged_reference_paths`: exact
  conditional GP given RAW history, information unavailable to the networks.
  This reference is not an attainable baseline or finite-sample lower bound.

Figures are exported as PNG and editable vector SVG. `analysis_status.json`
records the exact input fingerprint. Raw ensembles remain in the results folder.
''')
    print('Complete ensemble analysis:',out)


if __name__=='__main__':
    main()
