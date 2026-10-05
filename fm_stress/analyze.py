"""Reproducible scientific tables/figures from immutable experiment run records.

Usage: MPLCONFIGDIR=/tmp/fmstress-mpl .venv/bin/python -m fm_stress.analyze \
       --results results/main --output output/analysis --require-complete
Partial results are rendered honestly; --require-complete additionally fails if
any configured cell/seed or its sample archive is absent.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from .metrics import paired_bootstrap_difference

CELLS = [('white', 'mlp'), ('white', 's4'), ('gp', 'mlp'), ('gp', 's4'), ('gp', 'whitened_mlp')]
LABELS = {('white','mlp'): 'White noise + MLP', ('white','s4'): 'White noise + S4',
          ('gp','mlp'): 'GP + MLP', ('gp','s4'): 'GP + S4', ('gp','whitened_mlp'): 'GP + whitened MLP'}
COLORS = {('white','mlp'): '#526DAB', ('white','s4'): '#132F59',
          ('gp','mlp'): '#DB8642', ('gp','s4'): '#13837E', ('gp','whitened_mlp'): '#945698'}
SCALARS = ['pc1_mse', 'pc2_mse', 'residual_mse', 'velocity_mse',
           'pc1_normalized_mse', 'pc2_normalized_mse', 'residual_normalized_mse',
           'generated_roughness', 'target_roughness', 'roughness_ratio',
           'roughness_relative_deviation', 'residual_to_pc1', 'normalized_residual_to_pc1']


def read(path):
    return json.loads(Path(path).read_text())


def save_json(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                        'axes.titlesize':12, 'axes.titleweight':'bold',
                        'axes.labelsize':10, 'figure.titlesize':16,
                        'axes.spines.top':False, 'axes.spines.right':False,
                        'axes.edgecolor':'#BDCACF', 'text.color':'#20333F',
                        'axes.labelcolor':'#20333F', 'xtick.color':'#52646E',
                        'ytick.color':'#52646E', 'grid.color':'#DFE5E8',
                        'grid.linewidth':.6, 'axes.axisbelow':True,
                        'savefig.facecolor':'white', 'figure.facecolor':'white',
                        'pdf.fonttype':42, 'ps.fonttype':42})


def save_figure(fig, out, stem):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('png','pdf'):
        fig.savefig(out / f'{stem}.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def values(record):
    m = record['metrics']
    v = {key: m.get(key) for key in SCALARS}
    v['roughness_ratio'] = m['success']['roughness_ratio']
    v['roughness_relative_deviation'] = None if v['roughness_ratio'] is None else abs(v['roughness_ratio'] - 1)
    v['residual_to_pc1'] = m['success']['residual_to_pc1']
    v['normalized_residual_to_pc1'] = m['normalized_success_diagnostic']['residual_to_pc1']
    return v


def collect(results):
    results = Path(results)
    config = read(results / 'config.json') if (results/'config.json').exists() else {}
    records = [read(p) for p in sorted((results/'runs').glob('*.json'))]
    expected_seeds = config.get('seeds', [0,1,2,3,4])
    groups = {cell: [] for cell in CELLS}
    seen = set()
    hashes = set()
    for record in records:
        cell = (record['source'], record['head'])
        identity = (*cell, record['seed'])
        if identity in seen:
            raise ValueError(f'Duplicate cell/seed: {identity}')
        seen.add(identity)
        hashes.add(record.get('config_sha256'))
        if cell not in groups:
            warnings.warn(f'Ignoring unrecognized cell {cell}')
        else:
            groups[cell].append(record)
    if len(hashes) > 1:
        raise ValueError('Run files contain multiple configuration hashes; refusing to pool')
    for group in groups.values():
        group.sort(key=lambda x: x['seed'])
    missing = [f'{s}_{h}_seed{seed}' for s,h in CELLS for seed in expected_seeds if (s,h,seed) not in seen]
    sample_missing = [r['run'] for r in records if not (results/'samples'/f"{r['run']}.npz").exists()]
    status = {'complete': not missing and not sample_missing, 'expected_seeds': expected_seeds,
              'expected_runs': len(CELLS)*len(expected_seeds), 'observed_runs':len(records),
              'missing_runs':missing, 'missing_sample_archives':sample_missing,
              'seed_counts':{LABELS[cell]:len(group) for cell,group in groups.items()},
              'config_sha256': next(iter(hashes), None)}
    return config, records, groups, status


def tables(records, groups, out):
    raw = []
    summary = []
    for record in records:
        row = {k:record[k] for k in ('run','source','head','seed','parameters','device','best_step','seconds')}
        row.update(values(record))
        row['primary_works'] = record['metrics']['success']['works']
        row['normalized_diagnostic_works'] = record['metrics']['normalized_success_diagnostic']['works']
        raw.append(row)
    for cell, group in groups.items():
        if not group:
            continue
        row = {'source':cell[0], 'head':cell[1], 'label':LABELS[cell], 'n_seeds':len(group),
               'parameters':group[0]['parameters'],
               'primary_works_count':sum(r['metrics']['success']['works'] for r in group),
               'normalized_diagnostic_works_count':sum(r['metrics']['normalized_success_diagnostic']['works'] for r in group)}
        for metric in SCALARS + ['seconds']:
            x = [r['seconds'] if metric == 'seconds' else values(r)[metric] for r in group]
            x = np.asarray([v for v in x if v is not None], float)
            row[f'{metric}_mean'] = float(x.mean()) if len(x) else None
            row[f'{metric}_sd'] = float(x.std(ddof=1)) if len(x)>1 else None
        summary.append(row)
    contrasts = []
    comparisons = [('Head effect, white source', ('white','s4'),('white','mlp')),
                   ('Head effect, GP source', ('gp','s4'),('gp','mlp')),
                   ('Source effect, MLP', ('gp','mlp'),('white','mlp')),
                   ('Source effect, S4', ('gp','s4'),('white','s4')),
                   ('Whitening control', ('gp','whitened_mlp'),('gp','mlp')),
                   ('Whitened MLP versus S4', ('gp','whitened_mlp'),('gp','s4'))]
    for title,a,b in comparisons:
        ga={r['seed']:r for r in groups[a]}; gb={r['seed']:r for r in groups[b]}
        common=sorted(set(ga)&set(gb))
        if len(common)<2:
            continue
        for metric in SCALARS:
            av=[values(ga[s])[metric] for s in common]; bv=[values(gb[s])[metric] for s in common]
            if any(v is None for v in av+bv):
                continue
            result=paired_bootstrap_difference(av,bv,seed=917,repetitions=10000)
            contrasts.append({'contrast':title,'a':LABELS[a],'b':LABELS[b],'metric':metric,
                              'direction':'a minus b','seeds':','.join(map(str,common)),**result})
    write_csv(out/'per_seed_metrics.csv',raw)
    write_csv(out/'summary_metrics.csv',summary)
    write_csv(out/'paired_seed_contrasts.csv',contrasts)
    save_json(out/'summary_metrics.json',summary)
    save_json(out/'paired_seed_contrasts.json',contrasts)
    lines=['# Required metrics', '', 'Mean ± sample SD across optimization seeds; evaluation windows are shared.', '',
           '| Cell | Seeds | PC1 velocity MSE | PCs 3–F velocity MSE | Generated roughness | Target roughness | Literal rule passes |',
           '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(row,key):
        mean,sd=row.get(key+'_mean'),row.get(key+'_sd')
        if key == 'target_roughness':
            return f'{mean:.4g}'  # The same held-out target set is reused across seeds.
        return f'{mean:.4g}' + (f' ± {sd:.2g}' if sd is not None else ' (one seed)')
    for row in summary:
        lines.append('| '+ ' | '.join([row['label'],str(row['n_seeds']),fmt(row,'pc1_mse'),fmt(row,'residual_mse'),fmt(row,'generated_roughness'),fmt(row,'target_roughness'),f"{row['primary_works_count']}/{row['n_seeds']}"])+' |')
    lines.extend(['', 'The literal rule is two-sided: residual/PC1 MSE and generated/target roughness must each lie in [0.8,1.2].',
                  'A smaller residual MSE can fail this rule. Variance-normalized diagnostics are secondary, not substituted successes.',
                  '', 'Paired bootstrap intervals resample optimization seeds (10,000 resamples, percentile 95% CI).',
                  'They describe seed variability conditional on this fixed data draw and hyperparameters; five seeds provide limited interval resolution.',
                  'Source contrasts use source-specific velocity PCA bases and therefore also reflect different regression-label geometry.'])
    (out/'metrics_table.md').write_text('\n'.join(lines)+'\n')
    return summary


def geometry_figure(results,out):
    path=results/'geometry.json'
    if not path.exists(): return
    g=read(path)
    styles={'target':('#20333F','Target patches'),'white':('#526DAB','White-source velocity'),'gp':('#13837E','GP-source velocity')}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    rows=[]
    for name,(color,label) in styles.items():
        split='test' if 'test' in g['spectra'][name] else 'val'
        s=g['spectra'][name][split]; energy=np.asarray(s['energy']); pc=np.arange(1,len(energy)+1)
        axes[0].semilogy(pc,energy,'o-',ms=4,lw=1.8,color=color,label=label)
        axes[1].plot(pc,np.cumsum(energy),'o-',ms=4,lw=1.8,color=color,label=f'{label} (E₂={s["top2_energy"]:.3f})')
        for splitname,stat in g['spectra'][name].items():
            rows.append({'quantity':name,'split':splitname,**{k:v for k,v in stat.items() if not isinstance(v,list)}})
    axes[1].axhline(.9,color='#B85348',ls='--',lw=1,label='90% geometry threshold')
    axes[1].axvline(2,color='#A9B5BC',ls=':',lw=1)
    axes[0].set(title='Variance spectrum',xlabel='Principal component (train-fitted axis)',ylabel='Held-out variance share')
    axes[1].set(title='Cumulative energy',xlabel='Number of principal components',ylabel='Held-out cumulative variance share',ylim=(0,1.03))
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.6)
    axes[1].legend(fontsize=8.2,loc='lower right')
    fig.suptitle('Task 1 · held-out geometry in training PCA axes',fontweight='bold')
    save_figure(fig,out,'geometry_spectra');write_csv(out/'geometry_table.csv',rows)


def paths_figure(results,groups,out,seed=0):
    indices=list(range(6))  # Fixed before inspecting paths; never outcome-selected.
    active=[]
    for cell in CELLS:
        name=f'{cell[0]}_{cell[1]}_seed{seed}'
        p=results/'samples'/f'{name}.npz'
        if p.exists(): active.append((cell,p))
    if not active:return
    data={cell:dict(np.load(p)) for cell,p in active}
    reference=next(iter(data.values()))
    for cell,d in data.items():
        if not np.allclose(d['history'][:6],reference['history'][:6]) or not np.allclose(d['target'][:6],reference['target'][:6]):
            raise ValueError(f'Path comparison must use identical histories/targets: {cell}')
    fig,axes=plt.subplots(len(active),6,figsize=(15,2.0*len(active)+.7),squeeze=False,sharex=True,layout='constrained')
    # Same y-limits down each context column enables fair visual comparison.
    for col,idx in enumerate(indices):
        all_values=np.concatenate([np.r_[d['generated'][idx],d['target'][idx],d['history'][idx,-8:]] for d in data.values()])
        low,high=np.min(all_values),np.max(all_values);pad=max((high-low)*.1,.2)
        for row,(cell,_) in enumerate(active):
            d=data[cell];ax=axes[row,col];f=d['generated'].shape[-1]
            ax.plot(np.arange(-7,1),d['history'][idx,-8:],color='#AAB5BB',lw=1.2)
            ax.plot(np.arange(1,f+1),d['target'][idx],color='#253944',ls='--',lw=1.5)
            ax.plot(np.arange(1,f+1),d['generated'][idx],color=COLORS[cell],lw=1.7)
            ax.axvline(.5,color='#CCD5DA',lw=.7);ax.set_ylim(low-pad,high+pad);ax.grid(axis='y',alpha=.5)
            if row==0:ax.set_title(f'Context {idx}',fontsize=10)
            if col==0:ax.set_ylabel(LABELS[cell],fontsize=9)
            if row==len(active)-1:ax.set_xlabel('Relative time')
            ax.tick_params(labelsize=8);ax.xaxis.set_major_locator(MaxNLocator(4,integer=True));ax.yaxis.set_major_locator(MaxNLocator(4))
    legend=[Line2D([0],[0],color='#AAB5BB',lw=1.5,label='Last 8 history points'),
            Line2D([0],[0],color='#253944',ls='--',lw=1.5,label='Observed future draw'),
            Line2D([0],[0],color='#13837E',lw=1.7,label='Generated draw (row color)')]
    fig.suptitle(f'Generated paths · fixed first six held-out contexts · seed {seed}',fontweight='bold')
    fig.legend(handles=legend,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    save_figure(fig,out,'generated_paths_all_cells')
    save_json(out/'path_figure_manifest.json',{'seed':seed,'indices':indices,'selection':'first six test contexts, no outcome selection',
        'normalization':'history-normalized observation coordinates',
        'interpretation':'Generated and observed futures are distinct stochastic draws; their pointwise difference is not the forecast error metric.',
        'files':[str(p) for _,p in active]})
    # A4 report view: fixed first THREE contexts. Reduced column count improves
    # embedded text size; it is a layout choice made without selecting outcomes.
    report_indices=indices[:3]
    fig,axes=plt.subplots(len(active),3,figsize=(8,8.5),squeeze=False,sharex=True,layout='constrained')
    for col,idx in enumerate(report_indices):
        all_values=np.concatenate([np.r_[d['generated'][idx],d['target'][idx],d['history'][idx,-8:]] for d in data.values()])
        low,high=np.min(all_values),np.max(all_values);pad=max((high-low)*.1,.2)
        for row,(cell,_) in enumerate(active):
            d=data[cell];ax=axes[row,col];f=d['generated'].shape[-1]
            ax.plot(np.arange(-7,1),d['history'][idx,-8:],color='#AAB5BB',lw=1.2)
            ax.plot(np.arange(1,f+1),d['target'][idx],color='#253944',ls='--',lw=1.4)
            ax.plot(np.arange(1,f+1),d['generated'][idx],color=COLORS[cell],lw=1.8)
            ax.axvline(.5,color='#CCD5DA',lw=.7);ax.set_ylim(low-pad,high+pad);ax.grid(axis='y',alpha=.5)
            if row==0:ax.set_title(f'Context {idx}',fontsize=11)
            if col==0:ax.set_ylabel(LABELS[cell].replace(' + ',' +\n'),fontsize=10)
            if row==len(active)-1:ax.set_xlabel('Relative time',fontsize=10)
            ax.tick_params(labelsize=10);ax.xaxis.set_major_locator(MaxNLocator(3,integer=True));ax.yaxis.set_major_locator(MaxNLocator(3))
    fig.legend(handles=legend,loc='outside lower center',ncol=1,frameon=False,fontsize=10)
    save_figure(fig,out,'generated_paths_report')
    save_json(out/'report_path_figure_manifest.json',{'seed':seed,'indices':report_indices,
        'selection':'fixed first three test contexts; fewer columns solely for A4 legibility',
        'full_supplement':'generated_paths_all_cells.pdf contains fixed first six contexts',
        'same_y_limits_across_models_per_context':True})
    # Conventional 2x2 overview plus separate control, same six color-coded contexts.
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,layout='constrained')
    palette=['#156A78','#C07932','#86558A','#365EA1','#638348','#BC545A']
    for cell,ax in zip(CELLS[:4],axes.flat):
        if cell not in data:
            ax.text(.5,.5,'Run not available',ha='center',transform=ax.transAxes);continue
        d=data[cell];f=d['generated'].shape[-1]
        for idx,color in zip(indices,palette):
            ax.plot(np.arange(1,f+1),d['target'][idx],color=color,ls='--',lw=1.0,alpha=.75)
            ax.plot(np.arange(1,f+1),d['generated'][idx],color=color,lw=1.5)
        ax.set_title(LABELS[cell]);ax.set_xlabel('Future time step');ax.set_ylabel('Normalized value');ax.grid(alpha=.5)
    fig.suptitle('Task 2 · same six test contexts in every cell',fontweight='bold')
    fig.legend(handles=[Line2D([0],[0],color='#20333F',lw=1.5,label='Generated'),Line2D([0],[0],color='#20333F',ls='--',lw=1.2,label='Observed target draw')],loc='outside lower center',ncol=2,frameon=False)
    save_figure(fig,out,'generated_paths_2x2')


def pc_errors_figure(groups,oracle,out):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,source in enumerate(('white','gp')):
        for cell,group in groups.items():
            if cell[0]!=source or not group:continue
            for row,key in enumerate(('pc_mse','pc_normalized_mse')):
                v=np.asarray([r['metrics'][key] for r in group]);pc=np.arange(1,v.shape[1]+1);mean=v.mean(0)
                ax=axes[row,col];ax.semilogy(pc,mean,'o-',ms=3,color=COLORS[cell],lw=1.7,label=LABELS[cell])
                if len(v)>1:
                    sd=v.std(0,ddof=1);ax.fill_between(pc,np.maximum(mean-sd,1e-15),mean+sd,color=COLORS[cell],alpha=.12)
        if oracle and source in oracle.get('sources',{}):
            for row,key in enumerate(('pc_mse','pc_normalized_mse')):
                v=oracle['sources'][source][key]
                axes[row,col].semilogy(np.arange(1,len(v)+1),v,'--',color='#7B848B',lw=1.4,label='Privileged GP oracle')
        for row in range(2):
            axes[row,col].set(title=f'{"White noise" if source=="white" else "GP"} source · {"raw" if row==0 else "variance-normalized"}',xlabel='Velocity principal component',ylabel='MSE' if row==0 else 'MSE / floored train PC variance')
            axes[row,col].axvspan(2.5,16.5,color='#EBF1F2',alpha=.4,zorder=-2)
            axes[row,col].xaxis.set_major_locator(MaxNLocator(integer=True));axes[row,col].grid(alpha=.5);axes[row,col].legend(fontsize=8)
    fig.suptitle('Velocity errors · mean ± SD across seeds',fontweight='bold')
    save_figure(fig,out,'velocity_errors_per_pc')


def metric_figure(groups,out):
    active=[cell for cell in CELLS if groups[cell]]
    if not active:return
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    specs=[('pc1_mse','PC1 velocity error','MSE',False),('residual_mse','Residual velocity error','Mean MSE, PCs 3–F',False),('roughness_ratio','Generated roughness','Generated / target',False)]
    for ax,(key,title,ylabel,log) in zip(axes,specs):
        for i,cell in enumerate(active):
            v=np.asarray([values(r)[key] for r in groups[cell]])
            ax.bar(i,v.mean(),color=COLORS[cell],width=.66,alpha=.85)
            if len(v)>1:ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='none',ecolor='#20333F',capsize=3,lw=1)
            ax.scatter(i+np.linspace(-.13,.13,len(v)),v,s=17,color='white',edgecolor='#20333F',linewidth=.6,zorder=3)
        ax.set_xticks(range(len(active)),[LABELS[c].replace(' + ','\n+ ') for c in active],rotation=25,ha='right',fontsize=8)
        ax.set(title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.5)
        if log:ax.set_yscale('log')
        elif key == 'roughness_ratio':
            ax.axhspan(.8,1.2,color='#CFE5DD',alpha=.55,zorder=-1);ax.axhline(1,color='#60766D',ls='--',lw=1)
    fig.suptitle('Task 2 + control · required evaluation metrics',fontweight='bold')
    save_figure(fig,out,'required_metrics')


def learning_curves(results,groups,out):
    # Reads live training records too, allowing honest partial-progress previews.
    curves={cell:[] for cell in CELLS}
    for p in sorted((results/'training').glob('*.json')):
        for cell in CELLS:
            if p.stem.startswith(f'{cell[0]}_{cell[1]}_seed'):
                curves[cell].append(read(p));break
    if not any(curves.values()):return
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for cell,series in curves.items():
        if not series:continue
        for curve in series:
            steps=[p['step'] for p in curve]
            for ax,key in zip(axes,('validation_raw_mse','validation_objective')):
                ax.plot(steps,[p[key] for p in curve],color=COLORS[cell],lw=1,alpha=.25)
        # Summarize only steps present for every seed; avoid misrepresenting tails.
        common=sorted(set.intersection(*(set(p['step'] for p in s) for s in series)))
        for ax,key in zip(axes,('validation_raw_mse','validation_objective')):
            means=[np.mean([next(p[key] for p in s if p['step']==step) for s in series]) for step in common]
            ax.plot(common,means,color=COLORS[cell],lw=2,label=LABELS[cell])
    for ax in axes:
        ax.set_yscale('log');ax.set_xlabel('Optimizer steps');ax.grid(alpha=.5);ax.legend(fontsize=8)
    axes[0].set(title='Validation error in observation coordinates',ylabel='Raw velocity MSE')
    axes[1].set(title='Validation objective used for checkpoint selection',ylabel='Training-coordinate MSE')
    fig.suptitle('Learning curves · individual seeds and mean',fontweight='bold')
    save_figure(fig,out,'learning_curves')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/main',type=Path)
    parser.add_argument('--output',default='output/analysis',type=Path)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--path-seed',default=0,type=int)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);setup_style()
    config,records,groups,status=collect(args.results)
    save_json(args.output/'analysis_status.json',status)
    tables(records,groups,args.output)
    geometry_figure(args.results,args.output)
    oracle=read(args.results/'oracle.json') if (args.results/'oracle.json').exists() else None
    if records:
        paths_figure(args.results,groups,args.output,args.path_seed)
        pc_errors_figure(groups,oracle,args.output)
        metric_figure(groups,args.output)
    learning_curves(args.results,groups,args.output)
    notes=['# Analysis provenance', '',f'Results directory: `{args.results}`',f'Complete: **{status["complete"]}**',
           f'Observed/expected runs: {status["observed_runs"]}/{status["expected_runs"]}', '',
           '- Summary error bars are sample SD across optimization seeds, not standard errors.',
           '- Paired 95% bootstrap intervals resample matched seeds, 10,000 replicates, fixed seed 917.',
           '- Paths use seed 0 and the first six test contexts, never visually selected examples.',
           '- Solid and dashed paths are independent stochastic draws conditional on the same history; pointwise discrepancy is not a forecast metric.',
           '- The GP oracle has access to raw history statistics unavailable to the learned heads.',
           '- Learning-curve objective differs for the whitened control; raw validation MSE is shown separately.',
           '- Per-PC axes are fitted to training velocity draws separately for each source.',
           '- PNG and vector PDF versions of every figure are exported.']
    if status['missing_runs']:notes.extend(['','Missing runs: '+', '.join(status['missing_runs'])])
    (args.output/'README.md').write_text('\n'.join(notes)+'\n')
    print(json.dumps(status,indent=2))
    if args.require_complete and not status['complete']:
        raise SystemExit('Final completeness check failed; partial outputs were saved and labeled.')


if __name__=='__main__':main()
