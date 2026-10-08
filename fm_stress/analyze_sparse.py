"""Read-only synthesis of 264 sparse/whitening fits and 96 frozen controls.

Strict completion is the default. Partial output is a development artifact and
does not compute comparative intervals or select conclusions from early cells.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze_extension import (read,save_json,csv_write,mean_sd,percentile_interval,
                               metric_row,DATA_LABELS,COLORS,style)
from .experiment import config_hash
from .extended_data import build_data

LABELS={'mlp':'MLP','sparse_mlp':'Fixed-mask MLP','s4':'S4','unet':'U-Net',
 'mlp_whitened':'MLP · PCA whitening','mlp_balanced':'MLP · balanced loss',
 'history_dense_ae':'History AE · dense','history_topk_ae':'History AE · TopK',
 'history_conv_dense':'Conv. history · dense','history_conv_sparse':'Conv. history · sparse',
 'dictionary_dense':'Velocity dictionary · dense','dictionary_topk':'Velocity dictionary · TopK',
 'unet_l0':'U-Net · L₀ gates','moe_dense':'Experts · dense','moe_topk':'Experts · Top-2'}
ORDER=['mlp','mlp_balanced','mlp_whitened','sparse_mlp','s4','unet','unet_l0',
       'history_dense_ae','history_topk_ae','history_conv_dense','history_conv_sparse',
       'dictionary_dense','dictionary_topk','moe_dense','moe_topk']
SPARSE_PAIRS=[('history_topk_ae','history_dense_ae'),('history_conv_sparse','history_conv_dense'),
              ('dictionary_topk','dictionary_dense'),('moe_topk','moe_dense')]
METRICS=('fair_crps','joint_energy_scaled','roughness_ratio','velocity_pc1_mse',
         'velocity_residual_mse','velocity_residual_normalized_mse')
PROPER=('fair_crps','joint_energy_scaled')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_results(root,allow_partial=False):
    manifest=read(root/'manifest.json');cfg=manifest['config'];fingerprint=manifest['fingerprint']
    if config_hash({k:v for k,v in manifest.items() if k!='fingerprint'})!=fingerprint:
        raise ValueError('Invalid manifest fingerprint')
    for filename,digest in manifest['source_sha256'].items():
        if sha(Path(__file__).parent/filename)!=digest:raise ValueError('Changed numerical source '+filename)
    for filename,digest in manifest['dataset_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Changed input data '+filename)
    reference=Path(cfg['reference_results'])
    if sha(reference/'manifest.json')!=manifest['reference_manifest_sha256']:
        raise ValueError('Reference manifest changed')
    for filename,digest in manifest['reference_run_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Reference run changed '+filename)
    expected={(d,h,s,k) for d in cfg['datasets'] for h in cfg['heads'] for s in cfg['sources'] for k in cfg['seeds']}
    records=[];datasets={};observed=set()
    for dataset,dc in cfg['datasets'].items():
        dest=root/dataset;ref=reference/dataset
        common=read(ref/'data_metadata.json')
        datasets[dataset]={'metadata':common,'geometry':read(ref/'geometry.json'),
                           'baselines':read(ref/'baselines.json')['records'],
                           'reference_capacities':read(ref/'capacities.json')}
        if (dest/'sparse_baselines.json').exists():
            baseline=read(dest/'sparse_baselines.json')
            if baseline['status']!='complete' or config_hash(baseline['provenance'])!=baseline['fingerprint']:
                raise ValueError('Invalid Elastic Net provenance '+dataset)
            if sha(dest/'sparse_baseline_samples.npz')!=baseline['samples_sha256']:
                raise ValueError('Changed Elastic Net samples '+dataset)
            for filename,digest in baseline['provenance']['source_sha256'].items():
                if sha(Path(__file__).parent/filename)!=digest:raise ValueError('Changed baseline source '+filename)
            datasets[dataset]['sparse_baseline']=baseline
            datasets[dataset]['baselines'].update(baseline['records'])
        elif not allow_partial:raise ValueError('Missing Elastic Net baseline '+dataset)
        if (dest/'capacities.json').exists():datasets[dataset]['new_capacities']=read(dest/'capacities.json')
        if (dest/'data_metadata.json').exists() and read(dest/'data_metadata.json')!=common:
            raise ValueError('Dataset differs from reference '+dataset)
        for folder,is_new in ((ref,False),(dest,True)):
            for path in sorted((folder/'runs').glob('*.json')):
                r=read(path);identity=(dataset,r['head'],r['source'],r['seed'])
                if r['status']!='complete' or r['device']!='mps' or r['steps']!=cfg['steps']:
                    raise ValueError('Invalid run '+str(path))
                if is_new:
                    if identity not in expected or identity in observed or r['fingerprint']!=fingerprint:
                        raise ValueError('Unexpected/duplicate new run '+str(path))
                    observed.add(identity)
                    if sha(folder/'samples'/f"{r['name']}.npz")!=r['samples_sha256']:
                        raise ValueError('New sample hash changed '+str(path))
                checkpoint=folder/'checkpoints'/f"{r['name']}.pt"
                if sha(checkpoint)!=r['checkpoint_sha256']:raise ValueError('Checkpoint hash changed '+str(path))
                objective='validation_objective' if is_new else 'validation_mse'
                best=min(r['training'],key=lambda z:z[objective])
                if best['step']!=r['best_step']:raise ValueError('Checkpoint selection mismatch '+str(path))
                n=min(cfg['n_ensemble_contexts'],common['split_sizes']['test'])
                with np.load(folder/'samples'/f"{r['name']}.npz") as a:
                    if a['generated'].shape!=(n,cfg['ensemble_size'],common['horizon'],common['channels']):
                        raise ValueError('Forecast shape mismatch '+str(path))
                    np.testing.assert_array_equal(a['indices'],np.linspace(0,common['split_sizes']['test']-1,n,dtype=int))
                for key,values in r['per_context'].items():
                    if len(values)!=n:raise ValueError('Context count mismatch')
                    np.testing.assert_allclose(np.mean(values),r['forecast_means'][key],rtol=1e-12,atol=1e-12)
                records.append({'dataset':dataset,'origin':'new' if is_new else 'frozen_control','result_directory':str(folder),**r})
    controls=[r for r in records if r['origin']=='frozen_control']
    if len(controls)!=96:raise ValueError('Expected all 96 frozen controls')
    missing=sorted(expected-observed)
    complete=not missing and all('sparse_baseline' in d for d in datasets.values())
    if not complete and not allow_partial:raise ValueError(f'Incomplete new suite: {len(observed)}/{len(expected)}')
    return cfg,fingerprint,records,datasets,complete,missing


def aggregate(records):
    cells={}
    for r in records:cells.setdefault((r['dataset'],r['head'],r['source']),[]).append(r)
    output=[]
    for (dataset,head,source),cell in sorted(cells.items()):
        cell.sort(key=lambda r:r['seed']);first=cell[0]
        activity=set.intersection(*(set(r.get('validation_activity',{})) for r in cell))
        output.append({'dataset':dataset,'head':head,'source':source,'n':len(cell),'seeds':[r['seed'] for r in cell],
            'origin':first['origin'],'parameters':first['parameters'],'width':first['model_options']['width'],
            'metrics':{m:mean_sd([metric_row(r)[m] for r in cell]) for m in metric_row(first)},
            'activity':{k:mean_sd([r['validation_activity'][k] for r in cell]) for k in sorted(activity)},
            'seconds':mean_sd([r['seconds'] for r in cell]),'best_steps':[r['best_step'] for r in cell],
            'selected_final_step_count':sum(r['best_step']==r['steps'] for r in cell),
            'pca_floored_components':first.get('pca_floored_components'),
            'per_pc_velocity_mse':np.mean([r['velocity_metrics']['pc_mse'] for r in cell],axis=0).tolist()})
    return output


def definitions(cfg):
    for source in cfg['sources']:
        for head in cfg['heads']:
            yield 'new_vs_mlp',[(head,source,1),('mlp',source,-1)]
        for sparse,dense in SPARSE_PAIRS:
            yield 'sparse_vs_dense',[(sparse,source,1),(dense,source,-1)]
        yield 'whitened_vs_balanced',[('mlp_whitened',source,1),('mlp_balanced',source,-1)]
        yield 'l0_vs_unet',[('unet_l0',source,1),('unet',source,-1)]
    for head in ORDER:
        yield 'gp_vs_white',[(head,'gp',1),(head,'white',-1)]
        if head!='mlp':
            yield 'source_head_interaction',[(head,'gp',1),('mlp','gp',-1),(head,'white',-1),('mlp','white',1)]


def contrasts(records,cfg,complete):
    if not complete:return [],[]
    index={(r['dataset'],r['head'],r['source'],r['seed']):r for r in records}
    seed_rows=[];context_rows=[]
    for dataset,dc in cfg['datasets'].items():
        for kind,terms in definitions(cfg):
            identity={'dataset':dataset,'type':kind,'terms':[{'head':h,'source':s,'weight':w} for h,s,w in terms],
                      'definition':' '.join(('+' if w>0 and i else '-' if w<0 else '')+h+'/'+s for i,(h,s,w) in enumerate(terms))}
            for metric in METRICS:
                diffs=np.array([sum(w*metric_row(index[(dataset,h,s,seed)])[metric] for h,s,w in terms) for seed in cfg['seeds']])
                record={**identity,'metric':metric,'seeds':cfg['seeds'],'paired_seed_differences':diffs.tolist(),
                        'mean_difference':float(diffs.mean()),'sd_difference':float(diffs.std(ddof=1)),
                        'seed_range':[float(diffs.min()),float(diffs.max())],
                        'scope':'Three paired optimization seeds on the fixed dataset; exploratory, no multiple-comparison correction'}
                # Raw source-specific velocity errors are not comparable labels
                # across sources. Keep only proper forecasts/R for source effects.
                if kind in ('gp_vs_white','source_head_interaction') and metric.startswith('velocity_'):
                    continue
                if metric in PROPER:
                    record.update(percentile_interval(diffs))
                    delta=sum(w*np.mean([index[(dataset,h,s,seed)]['per_context'][metric] for seed in cfg['seeds']],axis=0) for h,s,w in terms)
                    for block in ([1] if dc['kind']=='synthetic' else [4,8]):
                        context_rows.append({**identity,'metric':metric,'contexts':len(delta),'block_length':block,
                            'bootstrap':'iid' if block==1 else 'circular_moving_block',**percentile_interval(delta,block),
                            'scope':'Fixed trained-model seed average; paired ordered selected contexts; excludes training-data/model-selection uncertainty'})
                seed_rows.append(record)
    return seed_rows,context_rows


def plot_save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)


def score_figure(out,cfg,rows,datasets,metric,complete):
    fig,axes=plt.subplots(2,2,figsize=(15,13),constrained_layout=True)
    index={(r['dataset'],r['head'],r['source']):r for r in rows}
    for ax,dataset in zip(axes.flat,cfg['datasets']):
        for j,source in enumerate(cfg['sources']):
            for i,head in enumerate(ORDER):
                row=index.get((dataset,head,source))
                if row is None:continue
                stat=row['metrics'][metric]
                ax.errorbar(stat['mean'],i+(j-.5)*.25,xerr=stat['sd'] or 0,fmt='o',markersize=5,
                            capsize=2,color=COLORS[source],label=('Temporally white' if source=='white' else 'Smooth GP') if i==0 else None)
        for name,color,linestyle,label in [('ridge_residual_bootstrap','#777777','--','Ridge + train residuals'),
                ('multitask_elastic_net_residual_bootstrap','#722F5C',':','Elastic Net + train residuals')]:
            if name in datasets[dataset]['baselines']:
                ax.axvline(datasets[dataset]['baselines'][name]['means'][metric],color=color,ls=linestyle,lw=1.4,label=label)
        ax.set_yticks(range(len(ORDER)),[LABELS[h] for h in ORDER],fontsize=10)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_xlim(left=0)
        ax.set_title(DATA_LABELS[dataset],fontsize=13)
        ax.set_xlabel('Fair marginal CRPS ↓' if metric=='fair_crps' else 'Joint energy / √(F×C) ↓')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=10,loc='outside lower center',ncol=4)
    fig.suptitle(('Complete' if complete else 'PARTIAL — not final')+' · mean ± seed SD · identical forecast contexts',fontsize=15)
    plot_save(fig,out,'score_crps' if metric=='fair_crps' else 'score_energy')


def paired_figure(out,cfg,seed_rows,complete):
    if not complete:return
    comparisons=[('mlp_whitened','mlp'),('mlp_balanced','mlp'),('mlp_whitened','mlp_balanced'),
                 *SPARSE_PAIRS,('unet_l0','unet')]
    titles=['Whitening − MLP','Balanced loss − MLP','Whitening − balanced loss',
            'TopK history − dense AE','Sparse conv. − dense conv.','TopK dictionary − dense dictionary',
            'Top-2 experts − dense experts','L₀ U-Net − U-Net']
    fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    for ax,dataset in zip(axes.flat,cfg['datasets']):
        for j,source in enumerate(cfg['sources']):
            for i,(a,b) in enumerate(comparisons):
                row=next(r for r in seed_rows if r['dataset']==dataset and r['metric']=='fair_crps' and
                    r['terms']==[{'head':a,'source':source,'weight':1},{'head':b,'source':source,'weight':-1}])
                ax.errorbar(row['mean_difference'],i+(j-.5)*.25,xerr=row['sd_difference'],fmt='o',color=COLORS[source],capsize=3,
                            label=('Temporally white' if source=='white' else 'Smooth GP') if i==0 else None)
        ax.axvline(0,color='#555555',lw=1);ax.set_yticks(range(len(titles)),titles,fontsize=10);ax.invert_yaxis()
        ax.set_xlabel('Paired CRPS difference · negative favors first method');ax.set_title(DATA_LABELS[dataset]);ax.grid(axis='x',alpha=.15)
    axes.flat[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Controlled comparisons · mean ± SD of three paired seed differences',fontsize=15)
    plot_save(fig,out,'paired_crps')


def activity_figure(out,cfg,rows):
    heads=['history_topk_ae','history_conv_sparse','dictionary_topk','unet_l0','moe_topk']
    keys=['code_active_fraction','code_active_fraction','code_active_fraction','gate_eval_active_fraction','routing_fraction']
    labels=['TopK history code','Sparse conv. code','TopK velocity code','L₀ eval gates','Top-2 expert routing']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,dataset in zip(axes.flat,cfg['datasets']):
        for j,source in enumerate(cfg['sources']):
            for i,(head,key) in enumerate(zip(heads,keys)):
                row=next((r for r in rows if (r['dataset'],r['head'],r['source'])==(dataset,head,source)),None)
                if row is None:continue
                value=row['activity'][key]
                ax.barh(i+(j-.5)*.3,value['mean'],height=.28,color=COLORS[source],
                        label=('Temporally white' if source=='white' else 'Smooth GP') if i==0 else None)
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,1.04)
        ax.set_xlabel('Fraction actually nonzero on validation');ax.set_title(DATA_LABELS[dataset]);ax.grid(axis='x',alpha=.15)
    axes.flat[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Measured sparsity · activation, channel gating and routing are different quantities',fontsize=13)
    plot_save(fig,out,'activity')


def convergence_figure(out,cfg,rows):
    fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for ax,source in zip(axes,cfg['sources']):
        values=np.full((len(cfg['datasets']),len(ORDER)),np.nan)
        for i,dataset in enumerate(cfg['datasets']):
            for j,head in enumerate(ORDER):
                row=next((r for r in rows if (r['dataset'],r['head'],r['source'])==(dataset,head,source)),None)
                if row:values[i,j]=row['selected_final_step_count']/row['n']
        im=ax.imshow(values,vmin=0,vmax=1,cmap='YlOrBr',aspect='auto')
        ax.set_xticks(range(len(ORDER)),[LABELS[h] for h in ORDER],rotation=65,ha='right',fontsize=8)
        ax.set_yticks(range(len(cfg['datasets'])),[DATA_LABELS[d] for d in cfg['datasets']],fontsize=10)
        ax.set_title('Temporally white' if source=='white' else 'Smooth GP')
    fig.colorbar(im,ax=axes,label='Fraction of seeds selecting final training step',shrink=.7)
    fig.suptitle('Budget sensitivity diagnostic · final-step selection does not establish convergence',fontsize=13)
    plot_save(fig,out,'selected_final_step')


def path_figures(out,cfg,records,complete):
    if not complete:return
    heads=['mlp','mlp_whitened','history_topk_ae','history_conv_sparse','dictionary_topk','unet_l0']
    index={(r['dataset'],r['head'],r['source'],r['seed']):r for r in records}
    for dataset,dc in cfg['datasets'].items():
        bundle=build_data(dc);test=bundle.splits['test'];channel=test.target.shape[-1]-1
        target=test.target[0,:,channel];history=test.context[0,:,channel]
        arrays={};limits=[target,history[-12:]]
        for head in heads:
            for source in cfg['sources']:
                record=index[(dataset,head,source,cfg['seeds'][0])]
                with np.load(Path(record['result_directory'])/'samples'/f"{record['name']}.npz") as archive:
                    if int(archive['indices'][0])!=0:raise ValueError('First declared path context changed')
                    paths=archive['generated'][0,:,:,channel]
                low,high=np.quantile(paths,[.05,.95],axis=0);mean=paths.mean(0)
                arrays[head,source]=(mean,low,high);limits.extend([low,high])
        low=min(float(x.min()) for x in limits);high=max(float(x.max()) for x in limits)
        pad=max((high-low)*.06,.05);future=np.arange(len(target));past=np.arange(-min(12,len(history)),0)
        fig,axes=plt.subplots(len(heads),2,figsize=(11,14),sharex=True,sharey=True,constrained_layout=True)
        for i,head in enumerate(heads):
            for j,source in enumerate(cfg['sources']):
                ax=axes[i,j];mean,lower,upper=arrays[head,source]
                ax.plot(past,history[-len(past):],color='#75858B',lw=1.5)
                ax.axvline(-.5,color='#9FAAAF',lw=.8,ls=':')
                ax.fill_between(future,lower,upper,color=COLORS[source],alpha=.18,label='Marginal 90% interval')
                ax.plot(future,mean,color=COLORS[source],lw=1.7,label='Ensemble mean')
                ax.plot(future,target,color='#152D36',lw=1.7,label='Observed future')
                ax.set_ylim(low-pad,high+pad);ax.grid(alpha=.12)
                if j==0:ax.set_ylabel(LABELS[head]+'\nTrain-standardized units',fontsize=9)
                if i==0:ax.set_title('Temporally white source' if source=='white' else 'Smooth GP source')
                if i==len(heads)-1:ax.set_xlabel('Time relative to forecast start')
        axes[0,0].legend(fontsize=8,frameon=False,loc='best')
        channel_label=bundle.metadata['channel_names'][channel]
        fig.suptitle(DATA_LABELS[dataset]+f' · {channel_label}\nFirst declared test context; seed {cfg["seeds"][0]}; shared axes; no outcome-based selection',fontsize=13)
        plot_save(fig,out,'paths_'+dataset)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',default='results/sparse-extension')
    p.add_argument('--output',default='output/sparse-analysis');p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();root=Path(a.results);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg,fingerprint,records,datasets,complete,missing=load_results(root,a.allow_partial)
    agg=aggregate(records);seed_rows,context_rows=contrasts(records,cfg,complete)
    solver=[{'dataset':r['dataset'],'head':r['head'],'source':r['source'],'seed':r['seed'],**r['solver_sensitivity']}
            for r in records if r['solver_sensitivity'] is not None]
    summary={'status':'complete' if complete else 'partial','fingerprint':fingerprint,'config':cfg,
        'expected_new_runs':len(cfg['heads'])*len(cfg['datasets'])*len(cfg['sources'])*len(cfg['seeds']),
        'observed_new_runs':sum(r['origin']=='new' for r in records),'frozen_control_runs':96,'missing':missing,
        'datasets':datasets,'aggregate':agg,'contrasts':seed_rows,'context_contrasts':context_rows,'solver_sensitivity':solver,
        'training':{'total_new_seconds':sum(r['seconds'] for r in records if r['origin']=='new'),
                    'new_checkpoints_at_final_step':sum(r['best_step']==r['steps'] for r in records if r['origin']=='new')},
        'interpretation':['All new neural methods are inside conditional flow matching; Elastic Net is an external baseline.',
            'Whitening transforms every joint future coordinate and weights velocity errors; balanced MLP changes loss only.',
            'Sparse/dense history, dictionary and expert pairs share width; convolution adds learned thresholds and L1.',
            'L0 is compared to the original same-width U-Net, but gates add a training stochasticity/regularization intervention.',
            'Raw velocity PCA metrics are compared within source; source effects use forecasts against the same targets.',
            'Three seeds, fixed dataset periods, short training and many unadjusted exploratory comparisons limit inference.',
            'Stored interval estimates are conditional on fitted models and chosen context-block sizes; not population guarantees.',
            'Sparse masks/activations/routing use dense computation; no sparse-kernel acceleration is claimed.']}
    save_json(out/'summary.json',summary)
    csv_write(out/'aggregate.csv',[{'dataset':r['dataset'],'head':r['head'],'source':r['source'],'n':r['n'],
        **{m+'_'+stat:value for m,v in r['metrics'].items() for stat,value in v.items()},
        **{'activity_'+m+'_'+stat:value for m,v in r['activity'].items() for stat,value in v.items()}} for r in agg])
    csv_write(out/'run_metrics.csv',[{'dataset':r['dataset'],'head':r['head'],'source':r['source'],'seed':r['seed'],
        'origin':r['origin'],'best_step':r['best_step'],'seconds':r['seconds'],**metric_row(r),**r.get('validation_activity',{})} for r in records])
    csv_write(out/'seed_contrasts.csv',[{**r,'terms':json.dumps(r['terms']),'paired_seed_differences':json.dumps(r['paired_seed_differences'])} for r in seed_rows])
    csv_write(out/'context_contrasts.csv',[{**r,'terms':json.dumps(r['terms'])} for r in context_rows])
    style()
    for metric in PROPER:score_figure(out,cfg,agg,datasets,metric,complete)
    paired_figure(out,cfg,seed_rows,complete);activity_figure(out,cfg,agg);convergence_figure(out,cfg,agg)
    path_figures(out,cfg,records,complete)
    print(summary['status'].upper(),summary['observed_new_runs'],'new + 96 frozen controls;',out,flush=True)


if __name__=='__main__':main()
