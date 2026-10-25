#!/usr/bin/env python3
"""Training-cutpoint calibration strata; final output requires all84 fits."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from check_followup_normalization import read,sha,save,rebuild
OUT=ROOT/'results/followup-calibration-strata';PUBLIC=ROOT/'output/followup-calibration-strata';CONFIG=ROOT/'configs/followup_calibration_strata.json'

def require(ok,why):
    if not ok:raise ValueError(why)

def partition(std,q):
    return {'low':std<q[0],'middle':(std>=q[0])&(std<=q[1]),'high':std>q[1]}

def bin_metrics(per,masks,keys):
    out={};n=len(next(iter(per.values())))
    for name,mask in masks.items():
        out[name]={'contexts':int(mask.sum()),'means':{key:float(np.asarray(per[key])[mask].mean()) if mask.any() else None for key in keys}}
    for key in keys:
        pooled=sum(v['contexts']*v['means'][key] for v in out.values() if v['contexts'])/n
        require(np.isclose(pooled,np.mean(per[key]),rtol=1e-12,atol=1e-12),'Bin pooling fails: '+key)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    plan=read(CONFIG);root=ROOT/'results/followup-normalization';cfg=read(root/'config.json');manifest=read(root/'manifest.json');data,_,scaler=rebuild(cfg)
    trainstd=data['train'][0].std(axis=1,ddof=0).reshape(-1);teststd=data['test'][0].std(axis=1,ddof=0).reshape(-1)
    q=np.quantile(trainstd,plan['cutpoints'],method=plan['quantile_method']);masks=partition(teststd,q)
    require(np.all(sum(m.astype(int) for m in masks.values())==1),'Nonexclusive bins')
    frozen={'analysis_config_sha256':sha(CONFIG),'script_sha256':sha(__file__),'independent_data_definition_sha256':sha(ROOT/'scripts/check_followup_normalization.py'),'experiment_config_sha256':sha(root/'config.json'),'experiment_manifest_sha256':sha(root/'manifest.json'),'dataset_sha256':sha(ROOT/cfg['dataset']['path']),'train_quartiles':q.tolist(),'training_histories':len(trainstd),'scaler':scaler,'pre_stratified_outcome_freeze':True}
    if args.prepare:
        require(not (OUT/'manifest.json').exists() or read(OUT/'manifest.json')==frozen,'Existing frozen choices changed')
        save(OUT/'manifest.json',frozen);save(OUT/'config.json',plan)
        # Arithmetic edge test, with deliberately empty low/high bins as well.
        synthetic={'x':np.arange(6,dtype=float)}
        for a,b in [(1.,4.),(-1.,6.)]:bin_metrics(synthetic,partition(np.arange(6.),[a,b]),['x'])
        print('FROZEN analysis choices; no stratified forecast outcomes read');return
    require((OUT/'manifest.json').exists() and read(OUT/'manifest.json')==frozen,'Analysis not frozen or inputs changed')
    aggregate_path=ROOT/'output/followup-normalization/summary.json';aggregate=read(aggregate_path);aggregate_hash=sha(aggregate_path)
    require(aggregate['complete'] and aggregate['completed_fits']==plan['expected_fits'] and not aggregate['missing'],'All84 completed fits and fresh complete aggregate required')
    require(aggregate['fingerprint']==manifest['fingerprint'],'Aggregate experiment mismatch')
    # The independent aggregate audit ties these cached per-context arrays to saved samples.
    audited=read(ROOT/'results/followup-normalization-audit/analysis.json')
    require(audited['complete'] and audited['passed'] and audited['analysis_sha256']==aggregate_hash,'Complete independent aggregate audit required')
    keys=plan['metrics'];rows=[];inputs={};lookup={};expected=set()
    for mode in cfg['normalizations']:
        for prior in cfg['priors']+(cfg['extra_priors'] if mode in ('global','history') else []):
            for head in cfg['heads']:
                for seed in cfg['seeds']:expected.add((mode,prior,head,seed))
    observed=set()
    for path in sorted((root/'runs').glob('*.json')):
        record=read(path);ident=tuple(record[k] for k in ['normalization','prior','head','seed']);require(ident in expected and ident not in observed,'Unexpected or duplicate fit');observed.add(ident)
        require(record['status']=='complete' and record['fingerprint']==manifest['fingerprint'],'Incomplete/incompatible fit')
        inputs[str(path.relative_to(ROOT))]=sha(path)
        for budget in plan['budgets']:
            b=record['budgets'][str(budget)];sample=root/'samples'/f'{record["name"]}_b{budget}.npz';require(sha(sample)==b['samples_sha256'],'Sample hash mismatch')
            with np.load(sample) as z:
                require(np.array_equal(z['starts'],data['test'][2]),'Chronological context indices differ')
                require(np.allclose(z['target'],data['test'][1],rtol=1e-12,atol=1e-12),'Targets differ')
                recovered=z['history_model']*z['scale']+z['location']
                require(np.allclose(recovered,data['test'][0],rtol=1e-12,atol=1e-12),'Raw standardized histories differ')
            per={k:np.asarray(b['per_context'][k],float) for k in keys}
            for key,v in per.items():
                require(v.shape==(len(teststd),) and np.isfinite(v).all(),'Invalid per-context metric '+key)
                require(np.isclose(v.mean(),b['forecast_means'][key],rtol=1e-12,atol=1e-12),'Pooled mean mismatch '+key)
            bins=bin_metrics(per,masks,keys);row={'normalization':ident[0],'prior':ident[1],'head':ident[2],'seed':ident[3],'budget':budget,'bins':bins,'sample_sha256':b['samples_sha256']};rows.append(row);lookup[(*ident,budget)]=row
    require(observed==expected and len(rows)==plan['expected_budget_records'],'Missing identities')
    cells={}
    for row in rows:
        key='|'.join(str(row[k]) for k in ['normalization','prior','head','budget']);cells.setdefault(key,[]).append(row)
    table={}
    for key,rr in cells.items():
        rr.sort(key=lambda r:r['seed']);require([r['seed'] for r in rr]==sorted(cfg['seeds']),'Cell seeds differ')
        table[key]={'seeds':[r['seed'] for r in rr],'bins':{binname:{'contexts':int(mask.sum()),'mean':{metric:float(np.mean([r['bins'][binname]['means'][metric] for r in rr])) if mask.any() else None for metric in keys},'per_seed':{metric:[r['bins'][binname]['means'][metric] for r in rr] for metric in keys}} for binname,mask in masks.items()}}
    baselines={mode:{'alpha':r['alpha'],'bins':bin_metrics(r['per_context'],masks,keys),'note':'One fixed training-residual bootstrap forecast; no neural optimization-seed replicates.'} for mode,r in aggregate['baselines'].items()}
    contrasts={}
    for prior in cfg['priors']:
        for head in cfg['heads']:
            for budget in plan['budgets']:
                vals={}
                for binname,mask in masks.items():
                    vals[binname]={metric:{'per_seed':[lookup[('history_rawloss',prior,head,seed,budget)]['bins'][binname]['means'][metric]-lookup[('center',prior,head,seed,budget)]['bins'][binname]['means'][metric] for seed in cfg['seeds']],'mean':None} for metric in keys} if mask.any() else None
                    if mask.any():
                        for v in vals[binname].values():v['mean']=float(np.mean(v['per_seed']))
                contrasts[f'history_rawloss-center|{prior}|{head}|{budget}']={'seeds':cfg['seeds'],'bins':vals}
    require(sha(aggregate_path)==aggregate_hash,'Aggregate changed during analysis')
    for name,digest in inputs.items():require(sha(ROOT/name)==digest,'Run changed during analysis')
    summary={'complete':True,'scope':plan['status'],'training_std_quartiles':q.tolist(),'test_bin_counts':{k:int(v.sum()) for k,v in masks.items()},'test_starts':data['test'][2].tolist(),'test_history_std':teststd.tolist(),'bin_indices':{k:np.flatnonzero(v).tolist() for k,v in masks.items()},'validated_fits':len(observed),'budget_records':len(rows),'table':table,'ridge_baselines':baselines,'paired_contrasts':contrasts,'frozen_manifest_sha256':sha(OUT/'manifest.json'),'aggregate_sha256':aggregate_hash,'run_hashes':inputs,'checks':'Every bin partition is exhaustive; bin-weighted per-seed metrics recover overall means; every saved sample context index, target and reconstructed original-unit history matches independent CSV reconstruction.','uncertainty':plan['uncertainty']}
    save(OUT/'summary.json',summary);save(OUT/'per_seed.json',rows);PUBLIC.mkdir(parents=True,exist_ok=True)
    flat=[]
    for key,cell in table.items():
        for binname,b in cell['bins'].items():flat.append(dict(zip(['normalization','prior','head','budget'],key.split('|')))|{'bin':binname,'contexts':b['contexts']}|b['mean'])
    with (PUBLIC/'strata.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'global':'Global','center':'Center only','scale':'Scale only','history':'Center + scale','history_rawloss':'History + raw loss'}
    for metric in ['coverage_90','width_90','fair_crps']:
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        for i,prior in enumerate(cfg['priors']):
            for j,head in enumerate(cfg['heads']):
                ax=axes[i,j]
                for mode in cfg['normalizations']:
                    cell=table[f'{mode}|{prior}|{head}|{plan["plot_budget"]}'];ax.plot(range(3),[cell['bins'][b]['mean'][metric] for b in masks],marker='o',label=names[mode])
                if metric=='coverage_90':ax.axhline(.9,color='gray',ls=':',lw=1)
                ax.set_title(f'{prior.upper()} / {head.upper()}');ax.set_xticks(range(3),[f'{b}\n(n={int(m.sum())})' for b,m in masks.items()]);ax.set_ylabel(metric.replace('_',' '));ax.spines[['top','right']].set_visible(False)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center');fig.suptitle('Exploratory calibration by history variability — training quartiles');fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(PUBLIC/(metric+'.png'),dpi=180);plt.close(fig)
    print('COMPLETE: 84 fits /168 budget records; training-only strata; all pooling checks passed')
if __name__=='__main__':main()
