"""Read-only analysis of the frozen multivariate/real extension.

Strict by default: all 96 declared fits and sample archives are required.
--allow-partial is for development and labels every output as incomplete.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .extended_data import build_data
from .extended_metrics import forecast_scores, means
from .experiment import config_hash

LABELS={'mlp':'Dense MLP','sparse_mlp':'Sparse MLP','s4':'S4','unet':'U-Net'}
DATA_LABELS={'gp_multi':'Synthetic GP · 3 channels','etth1_uni':'ETTh1 · oil temperature',
             'etth1_multi':'ETTh1 · 7 channels','exchange_multi':'Exchange · 8 channels'}
COLORS={'white':'#367A98','gp':'#BE713B'}
SCORES=('fair_crps','joint_energy_scaled')


def read(path):return json.loads(Path(path).read_text())


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def mean_sd(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None}


def percentile_interval(values,block=1,seed=9141,repetitions=10000):
    """Circular moving-block mean bootstrap; block=1 is iid resampling."""
    a=np.asarray(values,dtype=float);n=len(a)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(repetitions,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(repetitions,-1)[:,:n]
    draws=a[indices].mean(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return {'mean_difference':float(a.mean()),'ci_low':float(lo),'ci_high':float(hi)}


def metric_row(r):
    metrics=dict(r['forecast_means'])
    metrics.update({'velocity_'+k:float(v) for k,v in r['velocity_metrics'].items()
                    if isinstance(v,(int,float))})
    return metrics


def load_results(root,allow_partial):
    manifest=read(root/'manifest.json');cfg=manifest['config'];fingerprint=manifest['fingerprint']
    unsigned={k:v for k,v in manifest.items() if k!='fingerprint'}
    if config_hash(unsigned)!=fingerprint:raise ValueError('Invalid experiment fingerprint')
    for filename,digest in manifest['source_sha256'].items():
        path=Path(__file__).parent/filename
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed numerical source '+filename)
    for filename,digest in manifest['dataset_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:raise ValueError('Changed real data '+filename)
    expected={(d,h,s,k) for d in cfg['datasets'] for h in cfg['heads'] for s in cfg['sources'] for k in cfg['seeds']}
    records=[];datasets={};bundles={}
    for dataset,dc in cfg['datasets'].items():
        directory=root/dataset
        required=('data_metadata.json','geometry.json','baselines.json','capacities.json')
        if not all((directory/name).exists() for name in required):
            if allow_partial:continue
            raise ValueError('Incomplete dataset setup: '+dataset)
        meta=read(directory/'data_metadata.json');geo=read(directory/'geometry.json')
        datasets[dataset]={'channels':meta['channels'],'lookback':meta['lookback'],'horizon':meta['horizon'],
            'kind':meta['kind'],'split_sizes':meta['split_sizes'],
            'geometry':{key:{k:v for k,v in row.items() if k!='pca'} for key,row in geo.items()},
            'baselines':read(directory/'baselines.json'),'capacities':read(directory/'capacities.json')}
        bundles[dataset]=build_data(dc)
        for path in sorted((directory/'runs').glob('*.json')):
            r=read(path);identity=(dataset,r['head'],r['source'],r['seed'])
            if identity not in expected or r['fingerprint']!=fingerprint or r['status']!='complete':raise ValueError('Unexpected or invalid run '+str(path))
            if r['steps']!=cfg['steps']:raise ValueError('Mismatched training budget')
            if r['best_validation_mse']!=min(x['validation_mse'] for x in r['training']):raise ValueError('Checkpoint not validation-selected')
            sample=directory/'samples'/(r['name']+'.npz')
            if not sample.exists():raise ValueError('Missing sample archive '+str(sample))
            with np.load(sample) as a:
                shape=(min(cfg['n_ensemble_contexts'],meta['split_sizes']['test']),cfg['ensemble_size'],meta['horizon'],meta['channels'])
                if a['generated'].shape!=shape:raise ValueError('Unexpected ensemble shape')
                ix=np.asarray(a['indices'])
                expected_indices=np.linspace(0,meta['split_sizes']['test']-1,shape[0],dtype=int)
                np.testing.assert_array_equal(ix,expected_indices)
            for key,values in r['per_context'].items():
                if len(values)!=shape[0]:raise ValueError('Incorrect per-context score count')
                np.testing.assert_allclose(np.mean(values),r['forecast_means'][key],rtol=1e-12,atol=1e-12)
            records.append({'dataset':dataset,**r})
    observed={(r['dataset'],r['head'],r['source'],r['seed']) for r in records}
    if len(observed)!=len(records):raise ValueError('Duplicated run identities')
    complete=observed==expected
    if not complete and not allow_partial:raise ValueError(f'Incomplete extension: {len(observed)}/{len(expected)} runs')
    return cfg,fingerprint,records,datasets,bundles,complete,sorted(expected-observed)


def aggregates(records,cfg):
    rows=[]
    for dataset in cfg['datasets']:
        for head in cfg['heads']:
            for source in cfg['sources']:
                cell=sorted([r for r in records if (r['dataset'],r['head'],r['source'])==(dataset,head,source)],key=lambda r:r['seed'])
                if not cell:continue
                rows.append({'dataset':dataset,'head':head,'source':source,'n':len(cell),
                    'parameters':{'active':cell[0]['parameters']['active'],'allocated':cell[0]['parameters']['allocated'],
                                  'width':cell[0]['model_options']['width']},
                    'metrics':{k:mean_sd([metric_row(r)[k] for r in cell]) for k in metric_row(cell[0])},
                    'seconds':mean_sd([r['seconds'] for r in cell]),'best_steps':[r['best_step'] for r in cell],
                    'per_pc_velocity_mse':np.mean([r['velocity_metrics']['pc_mse'] for r in cell],axis=0).tolist()})
    return rows


def contrasts(records,cfg,complete):
    seed_rows=[];context_rows=[]
    if not complete:return seed_rows,context_rows
    for dataset,dc in cfg['datasets'].items():
        comparisons=[]
        for source in cfg['sources']:
            comparisons.extend(('head_vs_mlp',(h,source),('mlp',source)) for h in cfg['heads'] if h!='mlp')
        comparisons.extend(('gp_vs_white',(h,'gp'),(h,'white')) for h in cfg['heads'])
        for kind,a,b in comparisons:
            left=sorted([r for r in records if (r['dataset'],r['head'],r['source'])==(dataset,*a)],key=lambda r:r['seed'])
            right=sorted([r for r in records if (r['dataset'],r['head'],r['source'])==(dataset,*b)],key=lambda r:r['seed'])
            if [r['seed'] for r in left]!=cfg['seeds'] or [r['seed'] for r in right]!=cfg['seeds']:raise ValueError('Unpaired comparison')
            for metric in SCORES:
                differences=np.array([r['forecast_means'][metric] for r in left])-np.array([r['forecast_means'][metric] for r in right])
                identity={'dataset':dataset,'type':kind,'a':a[0]+'/'+a[1],'b':b[0]+'/'+b[1],'metric':metric}
                ci=percentile_interval(differences)
                seed_rows.append({**identity,'paired_seed_differences':differences.tolist(),**ci,
                    'seed_range':[float(differences.min()),float(differences.max())],
                    'scope':'Three paired optimization/mask seeds on one fixed dataset/test period; descriptive, not population evidence'})
                delta=np.mean([r['per_context'][metric] for r in left],axis=0)-np.mean([r['per_context'][metric] for r in right],axis=0)
                for block in ([1] if dc['kind']=='synthetic' else [4,8]):
                    context_rows.append({**identity,'bootstrap':'iid' if block==1 else 'circular_moving_block',
                        'block_length':block,'contexts':len(delta),**percentile_interval(delta,block),
                        'scope':'Fixed trained-model seed average; ordered selected contexts, no training-dataset uncertainty; block lengths are a sensitivity choice'})
        for head in cfg['heads']:
            if head=='mlp':continue
            terms=((head,'gp',1),('mlp','gp',-1),(head,'white',-1),('mlp','white',1))
            cells=[]
            for h,s,weight in terms:
                cell=sorted([r for r in records if (r['dataset'],r['head'],r['source'])==(dataset,h,s)],key=lambda r:r['seed'])
                if [r['seed'] for r in cell]!=cfg['seeds']:raise ValueError('Unpaired interaction')
                cells.append((weight,cell))
            for metric in SCORES:
                differences=sum(weight*np.array([r['forecast_means'][metric] for r in cell]) for weight,cell in cells)
                delta=sum(weight*np.mean([r['per_context'][metric] for r in cell],axis=0) for weight,cell in cells)
                identity={'dataset':dataset,'type':'interaction','head':head,'a':f'{head}/gp - mlp/gp',
                    'b':f'{head}/white - mlp/white','metric':metric,
                    'definition':f'({head} GP - MLP GP) - ({head} white - MLP white)',
                    'direction':'Negative means the head-versus-MLP score difference is more favorable under GP than under temporally white source'}
                seed_rows.append({**identity,'paired_seed_differences':differences.tolist(),**percentile_interval(differences),
                    'seed_range':[float(differences.min()),float(differences.max())],
                    'scope':'Three paired optimization/mask seeds on one fixed dataset/test period; exploratory difference of differences, not population evidence'})
                for block in ([1] if dc['kind']=='synthetic' else [4,8]):
                    context_rows.append({**identity,'bootstrap':'iid' if block==1 else 'circular_moving_block',
                        'block_length':block,'contexts':len(delta),**percentile_interval(delta,block),
                        'scope':'Four cells paired before resampling; fixed trained-model seed average; ordered selected contexts; no training-dataset uncertainty'})
    return seed_rows,context_rows


def ot_comparison(root,cfg,records,bundles,complete):
    names=('etth1_uni','etth1_multi')
    if not complete or not all(n in bundles for n in names):return {'status':'pending_complete_suite'}
    uni,multi=[bundles[n] for n in names]
    channel=multi.metadata['channel_names'].index('OT')
    for field in ('scaler_mean','scaler_scale'):
        np.testing.assert_allclose(uni.metadata[field][0],multi.metadata[field][channel],atol=1e-12,rtol=1e-12)
    ix=np.linspace(0,len(uni.splits['test'])-1,min(cfg['n_ensemble_contexts'],len(uni.splits['test'])),dtype=int)
    np.testing.assert_array_equal(uni.splits['test'].starts[ix],multi.splits['test'].starts[ix])
    target=uni.splits['test'].target[ix]
    np.testing.assert_allclose(target,multi.splits['test'].target[ix,:,channel:channel+1],atol=1e-12,rtol=1e-12)
    rows=[]
    for r in records:
        if r['dataset'] not in names:continue
        with np.load(root/r['dataset']/'samples'/(r['name']+'.npz')) as a:
            x=a['generated'] if r['dataset']==names[0] else a['generated'][:,:,:,channel:channel+1]
        scores=forecast_scores(x,target)
        rows.append({'dataset':r['dataset'],'head':r['head'],'source':r['source'],'seed':r['seed'],
                     'means':means(scores),'per_context':{k:v.tolist() for k,v in scores.items()}})
    agg=[];pairs=[]
    for head in cfg['heads']:
        for source in cfg['sources']:
            for name in names:
                group=[r for r in rows if (r['dataset'],r['head'],r['source'])==(name,head,source)]
                agg.append({'dataset':name,'head':head,'source':source,'n':len(group),
                            'metrics':{m:mean_sd([r['means'][m] for r in group]) for m in group[0]['means']}})
            for metric in SCORES:
                groups=[sorted([r for r in rows if (r['dataset'],r['head'],r['source'])==(name,head,source)],key=lambda r:r['seed']) for name in names]
                diff=np.array([r['means'][metric] for r in groups[1]])-np.array([r['means'][metric] for r in groups[0]])
                pairs.append({'head':head,'source':source,'metric':metric,'contrast':'joint-model OT minus univariate OT',
                    'paired_seed_differences':diff.tolist(),**percentile_interval(diff),
                    'scope':'Aligned forecast origins, exact same OT target/scaler; different model input/output dimensionality and width at matched active capacity'})
    return {'status':'complete','aligned_contexts':len(ix),'multivariate_ot_channel_index':channel,
            'records':rows,'aggregate':agg,'contrasts':pairs,
            'interpretation':'OT-only forecast score comparison; never compares seven-channel average with univariate OT. Different source channel covariance and capacity allocation also change.'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'savefig.bbox':'tight'})


def save_figure(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180)
    fig.savefig(out/(name+'.pdf'))
    plt.close(fig)


def score_figure(out,cfg,aggregate,datasets,metric,stem,complete):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,dataset in zip(axes.flat,cfg['datasets']):
        for j,source in enumerate(cfg['sources']):
            for i,head in enumerate(cfg['heads']):
                row=next((r for r in aggregate if (r['dataset'],r['head'],r['source'])==(dataset,head,source)),None)
                if row is None:continue
                y=row['metrics'][metric]
                ax.errorbar(i+(j-.5)*.22,y['mean'],yerr=y['sd'] or 0,fmt='o',capsize=3,color=COLORS[source],
                            label=('Temporally white' if source=='white' else 'Smooth GP') if i==0 else None)
        if dataset in datasets:
            baseline=datasets[dataset]['baselines']['records']
            for name,linestyle in [('persistence',':'),('ridge_residual_bootstrap','--')]:
                ax.axhline(baseline[name]['means'][metric],color='#65747B',linestyle=linestyle,lw=1.2,
                           label='Persistence' if name=='persistence' else 'Ridge + train residuals')
        ax.set_xticks(range(len(cfg['heads'])),[LABELS[h] for h in cfg['heads']],fontsize=12)
        ax.tick_params(axis='y',labelsize=12)
        ax.set_title(DATA_LABELS[dataset],fontsize=14);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Fair marginal CRPS ↓' if metric=='fair_crps' else 'Joint energy / √(F×C) ↓',fontsize=14)
        ax.set_ylim(bottom=0)
    axes.flat[0].legend(frameon=False,fontsize=11)
    fig.suptitle(('COMPLETE' if complete else 'PARTIAL — NOT FINAL')+' · mean ± SD across optimization seeds · fixed test contexts',fontsize=13)
    save_figure(fig,out,stem)


def geometry_figure(out,cfg,datasets):
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,dataset in zip(axes.flat,cfg['datasets']):
        if dataset not in datasets:ax.set_axis_off();continue
        geometry=datasets[dataset]['geometry']
        keys=['target','white','gp'];x=np.arange(3)
        ax.bar(x-.16,[geometry[k]['validation_top2_energy'] for k in keys],width=.32,color='#377E91',label='First 2 PCs')
        ax.bar(x+.16,[geometry[k]['validation_top2C_energy'] for k in keys],width=.32,color='#CC985A',label='First 2C PCs')
        ax.set_xticks(x,['Target','White velocity','GP velocity'],fontsize=12);ax.set_ylim(0,1.04)
        ax.tick_params(axis='y',labelsize=12)
        ax.set_title(DATA_LABELS[dataset],fontsize=14);ax.set_ylabel('Held-out variance share',fontsize=14);ax.grid(axis='y',alpha=.2)
    axes.flat[0].legend(frameon=False,fontsize=11)
    fig.suptitle('Training-fitted PCA · validation spectra · diagnostic, not a forced real-data gate',fontsize=13)
    save_figure(fig,out,'geometry')


def learning_figure(out,cfg,records):
    fig,axes=plt.subplots(len(cfg['datasets']),2,figsize=(12,12),constrained_layout=True)
    palette={'mlp':'#247C92','sparse_mlp':'#8865A1','s4':'#BF733E','unet':'#73854A'}
    for row,dataset in enumerate(cfg['datasets']):
        for col,source in enumerate(cfg['sources']):
            ax=axes[row,col]
            for head in cfg['heads']:
                cell=[r for r in records if (r['dataset'],r['source'],r['head'])==(dataset,source,head)]
                if not cell:continue
                curves=np.array([[p['validation_mse'] for p in r['training']] for r in cell]);steps=[p['step'] for p in cell[0]['training']]
                ax.plot(steps,curves.mean(0),label=LABELS[head],color=palette[head])
                ax.fill_between(steps,curves.min(0),curves.max(0),alpha=.1,color=palette[head])
            ax.set_title(DATA_LABELS[dataset]+' · '+source);ax.set_xlabel('Training updates');ax.set_ylabel('Validation velocity MSE')
            ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed-budget learning curves · mean and seed range · velocity losses differ across sources',fontsize=13)
    save_figure(fig,out,'learning_curves')


def path_figures(out,root,cfg,records,bundles,complete):
    if not complete:return
    for dataset,bundle in bundles.items():
        channel=bundle.metadata['channels']-1
        columns=[(source,head) for source in cfg['sources'] for head in cfg['heads']]
        ensembles={}
        for source,head in columns:
            name=f'{source}_{head}_seed{cfg["seeds"][0]}'
            with np.load(root/dataset/'samples'/(name+'.npz')) as a:
                ensembles[(source,head)]=a['generated'][:3,:,:,channel].copy();indices=a['indices'][:3].copy()
        target=bundle.splits['test'].target[indices,:,channel]
        fig,axes=plt.subplots(3,len(columns),figsize=(22,8),squeeze=False,constrained_layout=True)
        x=np.arange(1,target.shape[1]+1)
        for i in range(3):
            qs={key:np.quantile(a[i],[.05,.5,.95],axis=0) for key,a in ensembles.items()}
            lo=min(target[i].min(),min(q[0].min() for q in qs.values()));hi=max(target[i].max(),max(q[2].max() for q in qs.values()));pad=.08*max(hi-lo,.1)
            for j,key in enumerate(columns):
                ax=axes[i,j];q=qs[key];color=COLORS[key[0]]
                ax.fill_between(x,q[0],q[2],color=color,alpha=.18);ax.plot(x,q[1],color=color,lw=1.3);ax.plot(x,target[i],color='#172D37',lw=1.3)
                ax.set_ylim(lo-pad,hi+pad);ax.grid(alpha=.15)
                if i==0:ax.set_title(LABELS[key[1]]+' · '+key[0],fontsize=10)
                if j==0:ax.set_ylabel(f'Fixed context {i+1}\nTrain-standardized value')
                if i==2:ax.set_xlabel('Forecast step')
        name=bundle.metadata['channel_names'][channel]
        fig.suptitle(DATA_LABELS[dataset]+f' · {name} · first three selected contexts, seed {cfg["seeds"][0]}\nBlack: observed future; color: median and marginal 90% band; shared row axes',fontsize=14)
        save_figure(fig,out,'paths_'+dataset)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/extension');parser.add_argument('--output',default='output/extension-analysis')
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    root,out=Path(args.results),Path(args.output)
    cfg,fingerprint,records,datasets,bundles,complete,missing=load_results(root,args.allow_partial)
    aggregate=aggregates(records,cfg);seed_contrasts,context_contrasts=contrasts(records,cfg,complete)
    ot=ot_comparison(root,cfg,records,bundles,complete)
    summary={'status':'complete' if complete else 'partial','expected_runs':len(cfg['datasets'])*len(cfg['heads'])*len(cfg['sources'])*len(cfg['seeds']),
        'observed_runs':len(records),'missing':missing,'fingerprint':fingerprint,'config':cfg,'datasets':datasets,
        'aggregate':aggregate,'contrasts':seed_contrasts,'context_contrasts':context_contrasts,'ot_only':ot,
        'uncertainty':'Exploratory paired seed summaries; fixed data/test period. Real context bootstrap uses circular blocks of 4/8 ordered selected origins; sensitivity choice, not iid or population certainty.',
        'verification':'Actual source/data hashes, run completeness, archive shapes, selected indices and stored per-context means checked. Full independent array/checkpoint audit is a separate step.'}
    out.mkdir(parents=True,exist_ok=True);save_json(out/'summary.json',summary)
    csv_write(out/'run_metrics.csv',[{'dataset':r['dataset'],'head':r['head'],'source':r['source'],'seed':r['seed'],
        'active_parameters':r['parameters']['active'],'allocated_parameters':r['parameters']['allocated'],
        'width':r['model_options']['width'],'seconds':r['seconds'],'best_step':r['best_step'],**metric_row(r)} for r in records])
    csv_write(out/'aggregate.csv',[{'dataset':r['dataset'],'head':r['head'],'source':r['source'],'n':r['n'],**r['parameters'],
        **{k+'_'+stat:value for k,v in r['metrics'].items() for stat,value in v.items()},'mean_seconds':r['seconds']['mean']} for r in aggregate])
    csv_write(out/'paired_seed_contrasts.csv',seed_contrasts);csv_write(out/'paired_context_contrasts.csv',context_contrasts)
    if ot['status']=='complete':
        csv_write(out/'ot_only_run_metrics.csv',[{k:r[k] for k in ('dataset','head','source','seed')}|r['means'] for r in ot['records']])
        csv_write(out/'ot_only_contrasts.csv',ot['contrasts'])
    style();score_figure(out,cfg,aggregate,datasets,'fair_crps','score_crps',complete)
    score_figure(out,cfg,aggregate,datasets,'joint_energy_scaled','score_energy',complete)
    geometry_figure(out,cfg,datasets);learning_figure(out,cfg,records);path_figures(out,root,cfg,records,bundles,complete)
    (out/'README.md').write_text(f'''# Extension analysis: {summary['status']}

{len(records)} / {summary['expected_runs']} declared fits. Original experiments remain unchanged.
Scores use global training-standardized channel units. Sources share channel correlation;
white means temporally white. Joint energy is divided by sqrt(F*C). Baselines use the
same evaluation origins. Error bars are optimization-seed SD, not confidence intervals.

`summary.json` contains all scalar metrics, per-PC velocity means, baseline values,
paired contrasts, geometry, capacity and OT-only comparison. Real context intervals
use circular moving blocks of 4 and 8 selected chronological origins; their agreement
does not prove independent observations or validity under arbitrary nonstationarity.
Many exploratory comparisons are unadjusted for multiplicity. Three seeds do not
establish population-level significance or model equivalence. Source comparisons
use proper forecast scores, never raw velocity errors with different labels.
Source-by-head interactions are (head GP - MLP GP) - (head white - MLP white),
paired before resampling. Negative values favor the head relative to MLP more
under GP; an interaction alone does not establish superiority in either cell.

`paths_<dataset>` shows the first three fixed evaluation contexts, first declared
seed, and last channel for all eight source/head cells with shared row axes.
Plots show marginal 90% intervals, not simultaneous path bands. ETTh1 OT-only
comparisons align forecast origins, target/scaler and channel; they do not compare
the seven-channel average with a univariate target. The joint-model conditioning,
output dimensionality, capacity allocation and source covariance still differ.

PNG and vector PDF figures are provided. This analyzer verifies provenance and
basic aggregation; final delivery additionally requires independent raw-array and
checkpoint recomputation. Partial output is never a final comparison.
''')
    print(json.dumps({'status':summary['status'],'runs':len(records),'output':str(out)}))


if __name__=='__main__':main()
