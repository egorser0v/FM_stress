"""Verify and summarize the complete prospectively fixed identity-history follow-up."""
import argparse
import hashlib
import torch
import csv
import json
from pathlib import Path
import numpy as np
from .controlled_followup import HEADS, LOSSES, hashes, pilot_name
from .experiment import config_hash, load_config, save_json
from .metrics import PCA, evaluate, paired_bootstrap_difference, fit_pca
from .data import GPConfig, sample_dataset


def analyze(results,output):
    root,out=Path(results),Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(root/'config.json');selection=load_config(root/'selection.json')
    assert load_config(root/'source_manifest.json')==hashes(), 'Numerical source changed'
    assert selection['all_pilots_complete'] and selection['config_sha256']==config_hash(cfg)
    for name, digest in selection['pilot_record_sha256'].items():
        assert hashlib.sha256((root/'pilots'/'runs'/(name+'.json')).read_bytes()).hexdigest()==digest
    p=load_config(root/'geometry.json')['pca']['gp']
    pca=PCA(np.array(p['mean']),np.array(p['components']),np.array(p['eigenvalues']),p['floor'])
    target_pca=fit_pca(sample_dataset(GPConfig(**cfg['data']),cfg['n_train'],cfg['data_seed']).target)
    path_moments=[]
    runs=[];rows=[]
    for head in HEADS:
        for loss in LOSSES:
            for lr in cfg['pilot_learning_rates']:
                r=load_config(root/'pilots'/'runs'/(pilot_name(head,loss,lr)+'.json'))
                assert r['device']=='mps' and 'metrics' not in r and r['status']=='complete'
                assert r['best_validation_objective']==min(x['validation_objective'] for x in r['curve'])
            for seed in cfg['seeds']:
                name=f'gp_{head}_{loss}_seed{seed}'
                r=load_config(root/'runs'/(name+'.json'))
                assert r['device']=='mps' and r['status']=='complete' and r['config_sha256']==config_hash(cfg)
                assert r['learning_rate']==selection['selected'][head+'_'+loss]['learning_rate']
                assert r['best_validation_objective']==min(x['validation_objective'] for x in r['curve'])
                checkpoint=torch.load(root/'checkpoints'/(name+'.pt'),map_location='cpu',weights_only=False)
                assert checkpoint['step']==r['best_step'] and checkpoint['config']==cfg
                assert checkpoint['head']==head and checkpoint['loss']==loss and checkpoint['seed']==seed
                assert checkpoint['encoder']=='identity' and not any(k.startswith('encoder.') for k in checkpoint['state_dict'])
                with np.load(root/'samples'/(name+'.npz')) as sample:
                    m=evaluate(sample['prediction'],sample['velocity'],sample['generated'],sample['target'],pca)
                    generated_moment=np.mean(target_pca.project(sample['generated'])**2,axis=0)
                    target_moment=np.mean(target_pca.project(sample['target'])**2,axis=0)
                    tail_ratio=float(generated_moment[6:].sum()/target_moment[6:].sum())
                    path_moments.append({'run':name,'head':head,'loss':loss,'seed':seed,
                        'generated_pc_second_moments':generated_moment.tolist(),
                        'target_pc_second_moments':target_moment.tolist(),
                        'generated_pc7_16_second_moment':float(generated_moment[6:].sum()),
                        'target_pc7_16_second_moment':float(target_moment[6:].sum()),
                        'pc7_16_second_moment_ratio':tail_ratio})
                for key in ['pc1_mse','residual_mse','velocity_mse','generated_roughness','target_roughness',
                            'pc1_normalized_mse','residual_normalized_mse']:
                    np.testing.assert_allclose(m[key],r['metrics'][key],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(m['pc_mse'],r['metrics']['pc_mse'],rtol=1e-12,atol=1e-12)
                runs.append(r)
                row={k:r[k] for k in ['head','loss','seed','learning_rate','parameters','best_step','seconds']}
                row.update({k:m[k] for k in ['pc1_mse','residual_mse','velocity_mse','generated_roughness',
                                            'target_roughness','pc1_normalized_mse','residual_normalized_mse']})
                row.update(roughness_ratio=m['success']['roughness_ratio'],literal_success=m['success']['works'],
                           target_pca_pc7_16_moment_ratio=tail_ratio)
                rows.append(row)
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=['pc1_mse','residual_mse','velocity_mse','roughness_ratio','pc1_normalized_mse','residual_normalized_mse','target_pca_pc7_16_moment_ratio']
    aggregate=[]
    for head in HEADS:
        for loss in LOSSES:
            cell=[r for r in rows if r['head']==head and r['loss']==loss]
            aggregate.append({'head':head,'loss':loss,'n':len(cell),**{k:{'mean':float(np.mean([r[k] for r in cell])),
                              'std':float(np.std([r[k] for r in cell],ddof=1))} for k in metrics}})
    contrasts=[]
    def values(head,loss,metric):
        return np.array([next(r[metric] for r in rows if r['head']==head and r['loss']==loss and r['seed']==s) for s in cfg['seeds']])
    for metric in metrics:
        for loss in LOSSES:
            a,b=values('s4',loss,metric),values('mlp',loss,metric)
            contrasts.append({'contrast':f'S4 minus MLP ({loss})','metric':metric,'paired_values':(a-b).tolist(),
                              **paired_bootstrap_difference(a,b,seed=902,repetitions=10000)})
        for head in HEADS:
            a,b=values(head,'balanced',metric),values(head,'raw',metric)
            contrasts.append({'contrast':f'balanced minus raw ({head})','metric':metric,'paired_values':(a-b).tolist(),
                              **paired_bootstrap_difference(a,b,seed=902,repetitions=10000)})
        a=values('s4','balanced',metric)-values('mlp','balanced',metric)
        b=values('s4','raw',metric)-values('mlp','raw',metric)
        contrasts.append({'contrast':'head-by-loss interaction: balanced head gap minus raw head gap','metric':metric,
                          'paired_values':(a-b).tolist(),**paired_bootstrap_difference(a,b,seed=902,repetitions=10000)})
    summary={'status':'complete','verified_final_runs':len(rows),'verified_pilots':8,'config_sha256':config_hash(cfg),
             'selection':selection,'aggregate':aggregate,'contrasts':contrasts,
             'uncertainty':'Paired bootstrap of three optimization seeds conditional on one fresh GP dataset; exploratory, not population evidence.',
             'target_pca_path_moments':path_moments,
             'path_moment_definition':'Second moments of generated/target paths centered at the shared TRAIN TARGET mean and projected on shared TRAIN TARGET PCA; sum PCs7–16. No velocity PCA or floor used. One generated and one target draw per same1024histories.',
             'target_pca':target_pca.to_dict(),
             'floor':pca.floor,'floored_pc_indices_1based':(np.flatnonzero(pca.eigenvalues<pca.floor)+1).tolist()}
    save_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    styles={('mlp','raw'):('#186c9f','-'),('s4','raw'):('#c6643d','-'),
            ('mlp','balanced'):('#186c9f','--'),('s4','balanced'):('#c6643d','--')}
    for head in HEADS:
        for loss in LOSSES:
            cell=[r for r in runs if r['head']==head and r['loss']==loss]
            color,style=styles[(head,loss)]
            for ax,key in zip(axes,['pc_mse','pc_normalized_mse']):
                data=np.array([r['metrics'][key] for r in cell]);x=np.arange(1,data.shape[1]+1)
                ax.plot(x,data.mean(0),color=color,linestyle=style,label=head.upper()+' / '+loss)
                ax.fill_between(x,data.min(0),data.max(0),color=color,alpha=.09)
                ax.set_yscale('log');ax.set_xlabel('Training velocity principal component');ax.grid(alpha=.2)
    axes[0].set_ylabel('Raw velocity MSE');axes[1].set_ylabel('Variance-normalized velocity MSE')
    axes[0].legend(fontsize=8);fig.suptitle('Identity history: head × loss control (mean and seed range)')
    fig.tight_layout();fig.savefig(out/'per_pc_errors.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=np.arange(1,cfg['data']['horizon']+1)
    ax.plot(x,path_moments[0]['target_pc_second_moments'],color='#253943',linewidth=2,label='Target draws')
    for head in HEADS:
        for loss in LOSSES:
            moments=np.array([r['generated_pc_second_moments'] for r in path_moments if r['head']==head and r['loss']==loss])
            color,style=styles[(head,loss)]
            ax.plot(x,moments.mean(0),color=color,linestyle=style,label=head.upper()+' / '+loss)
            ax.fill_between(x,moments.min(0),moments.max(0),color=color,alpha=.09)
    ax.axvspan(6.5,16.5,color='#e5d7be',alpha=.2)
    ax.set_yscale('log');ax.set_xlabel('Common training-target principal component')
    ax.set_ylabel('Second moment around training-target mean')
    ax.set_title('Generated-path geometry; shaded region is PCs7–16')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'generated_target_pca_moments.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(11,8),sharex=True,sharey='col')
    for row,(head,loss) in enumerate((h,l) for h in HEADS for l in LOSSES):
        name=f'gp_{head}_{loss}_seed{cfg["seeds"][0]}'
        with np.load(root/'samples'/(name+'.npz')) as sample:
            for col in range(4):
                ax=axes[row,col];ax.plot(sample['target'][col],color='#253943',linewidth=1.4,label='Target draw')
                ax.plot(sample['generated'][col],color=styles[(head,loss)][0],linewidth=1.4,label='Generated draw')
                if row==0:ax.set_title(f'Fixed test context {col+1}')
                if col==0:ax.set_ylabel(head.upper()+' / '+loss)
                ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Same first four test contexts and source draws; final seed 10\nIndependent conditional draws need not overlap pointwise',fontsize=11)
    fig.tight_layout();fig.savefig(out/'generated_paths.png',dpi=180);plt.close(fig)
    text=['# Controlled follow-up: all planned runs completed','',
          'Eight validation-only learning-rate pilots and twelve final MPS runs; all saved final metrics were recomputed from sample arrays. No previous results are replaced.','',
          '| Head | Objective | PC1 MSE | Residual MSE | Normalized residual MSE | Roughness ratio | Path PC7–16 moment ratio |','|---|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        text.append(f"| {r['head'].upper()} | {r['loss']} | {r['pc1_mse']['mean']:.5f} | {r['residual_mse']['mean']:.6f} | {r['residual_normalized_mse']['mean']:.5f} | {r['roughness_ratio']['mean']:.4f} | {r['target_pca_pc7_16_moment_ratio']['mean']:.3f} |")
    text+=['','Both networks receive identical normalized history through an identity encoder. Balanced loss changes error weighting only; the S4 sequence remains indexed by time. Rate selection uses separate validation objectives and equal pilot budgets.','',summary['uncertainty'],'','Full paired contrasts, interactions, selected rates and per-run values are supplied in summary.json and runs.csv.']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',default='results/controlled_followup');parser.add_argument('--output',default='output/controlled-followup')
    a=parser.parse_args();result=analyze(a.results,a.output)
    print(json.dumps({k:result[k] for k in ['status','verified_final_runs','verified_pilots','aggregate']},indent=2))

if __name__=='__main__':main()
