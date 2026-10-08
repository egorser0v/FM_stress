"""Strict 64-vs-192 velocity dictionary comparison; no model/source mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analyze_extension import (read, save_json, csv_write, mean_sd, metric_row,
                               percentile_interval, style, DATA_LABELS, COLORS)
from .experiment import config_hash

HEADS=('dictionary_dense','dictionary_topk')
DATASETS=('etth1_multi','exchange_multi')
PROPER=('fair_crps','joint_energy_scaled')
METRICS=(*PROPER,'roughness_ratio','velocity_pc1_mse',
         'velocity_residual_mse','velocity_residual_normalized_mse')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_manifest(root,config_path):
    manifest=read(root/'manifest.json')
    if config_hash({k:v for k,v in manifest.items() if k!='fingerprint'})!=manifest['fingerprint']:
        raise ValueError('Invalid manifest fingerprint: '+str(root))
    if read(config_path)!=manifest['config']:
        raise ValueError('Configuration differs from manifest: '+str(config_path))
    for filename,digest in manifest['source_sha256'].items():
        if sha(Path(__file__).parent/filename)!=digest:
            raise ValueError('Changed numerical source: '+filename)
    for filename,digest in manifest['dataset_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Changed dataset: '+filename)
    ref=Path(manifest['config']['reference_results'])
    if sha(ref/'manifest.json')!=manifest['reference_manifest_sha256']:
        raise ValueError('Changed reference manifest')
    for filename,digest in manifest['reference_run_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Changed reference run: '+filename)
    return manifest


def load_comparison(main_root,supplement_root,main_config,supplement_config):
    roots={64:Path(main_root),192:Path(supplement_root)}
    manifests={64:checked_manifest(roots[64],main_config),192:checked_manifest(roots[192],supplement_config)}
    a,b=[manifests[k]['config'] for k in (64,192)]
    expected_shared=('sources','seeds','parameter_target','steps','batch_size','learning_rate',
        'weight_decay','clip_grad','eval_every','evaluation_seed','n_ensemble_contexts',
        'ensemble_size','euler_steps','generation_batch_size','reference_results')
    for key in expected_shared:
        if a[key]!=b[key]:raise ValueError('Incomparable configurations: '+key)
    if b['heads']!=list(HEADS) or set(b['datasets'])!=set(DATASETS):
        raise ValueError('Supplement is not the declared two-dataset/two-head control')
    if a['seeds']!=[31,32,33] or a['sources']!=['white','gp']:
        raise ValueError('Unexpected seed/source grid')
    for atoms,cfg in ((64,a),(192,b)):
        if cfg['sparse_model']['latent_dim']!=atoms or cfg['sparse_model']['topk']!=8:
            raise ValueError('Wrong dictionary size or TopK budget')
    if {k:v for k,v in a['sparse_model'].items() if k!='latent_dim'}!={k:v for k,v in b['sparse_model'].items() if k!='latent_dim'}:
        raise ValueError('Other sparse model settings changed')
    for key in ('source_sha256','reference_manifest_sha256','reference_run_sha256'):
        if manifests[64][key]!=manifests[192][key]:raise ValueError('Different source/reference provenance: '+key)
    records=[];files={};metadata={};observed={64:set(),192:set()}
    expected={(d,h,s,k) for d in DATASETS for h in HEADS for s in a['sources'] for k in a['seeds']}
    for dataset in DATASETS:
        if a['datasets'][dataset]!=b['datasets'][dataset]:raise ValueError('Different data configuration')
        meta=[read(roots[z]/dataset/'data_metadata.json') for z in (64,192)]
        if meta[0]!=meta[1]:raise ValueError('Different dataset metadata')
        metadata[dataset]=meta[0]
        if read(roots[64]/dataset/'geometry.json')!=read(roots[192]/dataset/'geometry.json'):
            raise ValueError('Different PCA/source geometry')
        for atoms,root in roots.items():
            dest=root/dataset;capacities=read(dest/'capacities.json')
            if atoms==192:
                allowed={f'{s}_{h}_seed{k}.json' for h in HEADS for s in a['sources'] for k in a['seeds']}
                if any(p.name not in allowed for p in (dest/'runs').glob('*.json')):
                    raise ValueError('Unexpected supplement run')
            if capacities[HEADS[0]]!=capacities[HEADS[1]]:
                raise ValueError('Dense/TopK pair has different capacities')
            for head in HEADS:
                for source in a['sources']:
                    for seed in a['seeds']:
                        name=f'{source}_{head}_seed{seed}';path=dest/'runs'/f'{name}.json'
                        if not path.exists():continue
                        r=read(path);identity=(dataset,head,source,seed)
                        if (r['name'],r['head'],r['source'],r['seed'])!=(name,head,source,seed):
                            raise ValueError('Run identity mismatch')
                        if r['status']!='complete' or r['fingerprint']!=manifests[atoms]['fingerprint'] or r['device']!='mps':
                            raise ValueError('Invalid completed MPS record')
                        if r['steps']!=a['steps'] or r['coordinate_input'] or r['variance_weighted_objective']:
                            raise ValueError('Different training budget/objective')
                        opts=r['model_options']
                        if opts['latent_dim']!=atoms or opts['topk']!=8 or opts['width']!=capacities[head]['width']:
                            raise ValueError('Wrong model options/capacity')
                        log=read(dest/'training'/f'{name}.json')
                        if log!=r['training'] or [z['step'] for z in log]!=list(range(a['eval_every'],a['steps']+1,a['eval_every'])):
                            raise ValueError('Incomplete training history')
                        best=min(log,key=lambda z:z['validation_objective'])
                        if best['step']!=r['best_step'] or best['validation_objective']!=r['best_validation_objective']:
                            raise ValueError('Checkpoint not validation selected')
                        cp=dest/'checkpoints'/f'{name}.pt';npz=dest/'samples'/f'{name}.npz'
                        for key,file in (('checkpoint_sha256',cp),('samples_sha256',npz)):
                            if sha(file)!=r[key]:raise ValueError('Corrupted '+str(file))
                            files[str(file)]=r[key]
                        checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
                        if checkpoint['options']!=opts or checkpoint['seed']!=seed or checkpoint['step']!=r['best_step'] or checkpoint['fingerprint']!=r['fingerprint']:
                            raise ValueError('Checkpoint metadata mismatch')
                        n=min(a['n_ensemble_contexts'],meta[0]['split_sizes']['test'])
                        shape=(n,a['ensemble_size'],meta[0]['horizon'],meta[0]['channels'])
                        with np.load(npz) as samples:
                            if samples['generated'].shape!=shape or not np.isfinite(samples['generated']).all():
                                raise ValueError('Invalid generated paths')
                            np.testing.assert_array_equal(samples['indices'],np.linspace(0,meta[0]['split_sizes']['test']-1,n,dtype=int))
                        for metric,values in r['per_context'].items():
                            if len(values)!=n or not np.isfinite(values).all():raise ValueError('Invalid per-context scores')
                            np.testing.assert_allclose(np.mean(values),r['forecast_means'][metric],rtol=1e-12,atol=1e-12)
                        files[str(path)]=sha(path);observed[atoms].add(identity)
                        records.append({'dataset':dataset,'atoms':atoms,'result_directory':str(dest),**r})
    missing={str(z):sorted(expected-observed[z]) for z in roots}
    if any(missing.values()):
        raise ValueError(f'Incomplete dictionary comparison: 64={len(observed[64])}/24; 192={len(observed[192])}/24')
    if len(records)!=48:raise ValueError('Wrong total record count')
    for d in DATASETS:
        for source in a['sources']:
            for seed in a['seeds']:
                group=[r for r in records if (r['dataset'],r['source'],r['seed'])==(d,source,seed)]
                for key in ('first_batch_sha256','last_batch_sha256'):
                    if len({r[key] for r in group})!=1:raise ValueError('Unpaired training random draws')
    return b,manifests,records,metadata,files


def aggregates(records):
    result=[]
    for dataset in DATASETS:
        for atoms in (64,192):
            for head in HEADS:
                for source in ('white','gp'):
                    cell=sorted([r for r in records if (r['dataset'],r['atoms'],r['head'],r['source'])==(dataset,atoms,head,source)],key=lambda z:z['seed'])
                    result.append({'dataset':dataset,'atoms':atoms,'head':head,'source':source,
                        'n':len(cell),'seeds':[r['seed'] for r in cell],
                        'width':cell[0]['model_options']['width'],'parameters':cell[0]['parameters'],
                        'metrics':{m:mean_sd([metric_row(r)[m] for r in cell]) for m in METRICS},
                        'activity':{m:mean_sd([r['validation_activity'][m] for r in cell]) for m in cell[0]['validation_activity']},
                        'best_steps':[r['best_step'] for r in cell],
                        'selected_final_step_count':sum(r['best_step']==r['steps'] for r in cell)})
    return result


def definitions():
    for source in ('white','gp'):
        for head in HEADS:
            yield '192_minus_64',source,[(192,head,1),(64,head,-1)]
        for atoms in (64,192):
            yield 'topk_minus_dense',source,[(atoms,'dictionary_topk',1),(atoms,'dictionary_dense',-1)]
        yield 'sparsity_by_dictionary_size',source,[(192,'dictionary_topk',1),(192,'dictionary_dense',-1),
                                                   (64,'dictionary_topk',-1),(64,'dictionary_dense',1)]


def contrasts(records):
    index={(r['dataset'],r['atoms'],r['head'],r['source'],r['seed']):r for r in records}
    seeds=[31,32,33];seed_rows=[];context_rows=[]
    for dataset in DATASETS:
        for kind,source,terms in definitions():
            identity={'dataset':dataset,'type':kind,'source':source,
                'terms':[{'atoms':z,'head':h,'weight':w} for z,h,w in terms],
                'definition':' '.join(('+' if w>0 and i else '-' if w<0 else '')+f'{h}/{z}' for i,(z,h,w) in enumerate(terms))}
            for metric in METRICS:
                differences=np.array([sum(w*metric_row(index[(dataset,z,h,source,seed)])[metric] for z,h,w in terms) for seed in seeds])
                seed_rows.append({**identity,'metric':metric,'seeds':seeds,
                    'paired_seed_differences':differences.tolist(),
                    'mean_difference':float(differences.mean()),'sd_difference':float(differences.std(ddof=1)),
                    'seed_range':[float(differences.min()),float(differences.max())],
                    'scope':'Three paired optimization seeds on one fixed dataset; descriptive, no multiple-comparison correction'})
                if metric in PROPER:
                    delta=sum(w*np.mean([index[(dataset,z,h,source,seed)]['per_context'][metric] for seed in seeds],axis=0) for z,h,w in terms)
                    for block in (4,8):
                        context_rows.append({**identity,'metric':metric,'contexts':len(delta),
                            'bootstrap':'circular_moving_block','block_length':block,'repetitions':10000,'bootstrap_seed':9141,
                            **percentile_interval(delta,block),
                            'scope':'Fixed trained-model seed average; paired ordered selected contexts; excludes training-data and model-selection uncertainty; exploratory supplement'})
    return seed_rows,context_rows


def score_figure(out,rows):
    style();fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    index={(r['dataset'],r['atoms'],r['head'],r['source']):r for r in rows}
    choices=[(64,HEADS[0]),(64,HEADS[1]),(192,HEADS[0]),(192,HEADS[1])]
    labels=['64 templates · dense','64 templates · TopK','192 templates · dense','192 templates · TopK']
    for i,dataset in enumerate(DATASETS):
        for j,metric in enumerate(PROPER):
            ax=axes[i,j]
            for offset,source in enumerate(('white','gp')):
                for k,(atoms,head) in enumerate(choices):
                    stat=index[(dataset,atoms,head,source)]['metrics'][metric]
                    ax.errorbar(stat['mean'],k+(offset-.5)*.18,xerr=stat['sd'],fmt='o',capsize=3,
                        color=COLORS[source],label=('Temporally white source' if source=='white' else 'Smooth GP source') if k==0 else None)
            ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(left=0);ax.grid(axis='x',alpha=.18)
            ax.set_title(DATA_LABELS[dataset]);ax.set_xlabel('Fair marginal CRPS ↓' if metric=='fair_crps' else 'Joint energy / √(F×C) ↓')
    axes[0,0].legend(frameon=False,fontsize=8,loc='best')
    fig.suptitle('Learned velocity templates · mean ± SD across three seeds\n192 templates remove the size-based rank limit; predictor width changes under the fixed parameter budget',fontsize=11)
    fig.savefig(out/'dictionary_scores.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'dictionary_scores.pdf',bbox_inches='tight');plt.close(fig)


def analyze(main_root,supplement_root,output,main_config,supplement_config,rank_diagnostics=None):
    cfg,manifests,records,metadata,files=load_comparison(main_root,supplement_root,main_config,supplement_config)
    rows=aggregates(records);seed_rows,context_rows=contrasts(records)
    summary={'status':'complete','expected_counts':{'main_64':24,'supplement_192':24},
        'observed_counts':{'main_64':24,'supplement_192':24},
        'comparison':'64-vs-192 learned velocity templates (dictionary atoms); TopK k=8; two multivariate datasets, both sources, three paired seeds',
        'manifest_fingerprints':{str(k):v['fingerprint'] for k,v in manifests.items()},
        'manifest_sha256':{'main_64':sha(Path(main_root)/'manifest.json'),'supplement_192':sha(Path(supplement_root)/'manifest.json')},
        'analyzer_source_sha256':sha(__file__),'analysis_helper_sha256':sha(Path(__file__).parent/'analyze_extension.py'),
        'verified_run_checkpoint_sample_sha256':files,'metadata':metadata,
        'aggregate':rows,'seed_contrasts':seed_rows,'context_contrasts':context_rows,
        'limitations':['This supplement was designed after inspecting the main-suite architecture and results; it is an exploratory follow-up.',
            'A dictionary with 192 atoms has enough columns to span 168-dimensional ETTh1 and 128-dimensional Exchange futures; learned numerical rank must be checked independently.',
            'The 64k parameter constraint changes predictor width when the dictionary grows. A 192-minus-64 comparison is not a pure causal rank intervention.',
            'TopK retains k=8 at both sizes, so its active fraction changes from 12.5% to about 4.2%; the available atom vocabulary also changes.',
            'Within each dictionary size, TopK and dense use the same predictor width and stored parameter count.',
            'Context block-bootstrap intervals condition on these fitted models and this test period; multiple comparisons are not adjusted.'],
        'rank_diagnostics':{'status':'not_attached','note':'Atom count is a capacity bound; no assertion of learned full rank without independent diagnostics.'}}
    if rank_diagnostics:
        paths=[rank_diagnostics] if isinstance(rank_diagnostics,(str,Path)) else rank_diagnostics
        lookup={(r['atoms'],r['dataset'],r['name']):r for r in records}
        rank_files={};rank_runs=[];rank_observed=set()
        for path in paths:
            data=read(path)
            if not data['complete'] or data['missing']:
                raise ValueError('Incomplete independent dictionary rank audit')
            match=[z for z,m in manifests.items() if m['fingerprint']==data['fingerprint']]
            if len(match)!=1:raise ValueError('Rank audit fingerprint does not match experiment')
            atoms=match[0]
            for record in data['runs']:
                key=(atoms,record['dataset'],record['name'])
                if key not in lookup:continue
                if key in rank_observed:raise ValueError('Duplicate rank audit record')
                expected=lookup[key]
                for field in ('checkpoint_sha256','samples_sha256'):
                    if record[field]!=expected[field]:raise ValueError('Rank audit uses different fitted artifacts')
                if record['dictionary_atoms']!=atoms:raise ValueError('Rank audit atom count mismatch')
                rank_runs.append({'atoms':atoms,**record});rank_observed.add(key)
            rank_files[str(path)]=sha(path)
        if rank_observed!=set(lookup):raise ValueError('Provide both complete main and supplement rank audit files')
        summary['rank_diagnostics']={'status':'verified','files_sha256':rank_files,'runs':rank_runs,
            'scope':'Independent audits matched to all 48 compared checkpoints and sample archives'}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    save_json(out/'summary.json',summary)
    csv_write(out/'aggregate.csv',[{'dataset':r['dataset'],'atoms':r['atoms'],'head':r['head'],'source':r['source'],
        'n':r['n'],'width':r['width'],'allocated_trainable':r['parameters']['allocated_trainable'],
        **{m+'_'+s:v for m,stat in r['metrics'].items() for s,v in stat.items()}} for r in rows])
    csv_write(out/'paired_seed_contrasts.csv',seed_rows);csv_write(out/'paired_context_contrasts.csv',context_rows)
    score_figure(out,rows)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--main',default='results/sparse-extension')
    p.add_argument('--supplement',default='results/sparse-fullrank-dictionary')
    p.add_argument('--output',default='output/dictionary-supplement')
    p.add_argument('--main-config',default='configs/sparse_extension.json')
    p.add_argument('--supplement-config',default='configs/sparse_fullrank_dictionary.json')
    p.add_argument('--rank-diagnostics',action='append')
    args=p.parse_args()
    summary=analyze(args.main,args.supplement,args.output,args.main_config,args.supplement_config,args.rank_diagnostics)
    print('COMPLETE dictionary supplement: 24 controls + 24 new fits;',len(summary['seed_contrasts']),
          'paired-seed contrasts;',len(summary['context_contrasts']),'context-block contrasts')


if __name__=='__main__':main()
