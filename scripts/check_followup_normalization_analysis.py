#!/usr/bin/env python3
"""Independent recomputation of normalization summaries, contrasts and ridge baselines."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from check_followup_normalization import ROOT,rebuild,transform,scores,read,save,sha,close

def interval(delta,block):
    rng=np.random.default_rng(414141);ns,n=delta.shape;estimates=np.empty(2000)
    for j in range(2000):
        begin=rng.integers(n,size=(n+block-1)//block)
        ix=np.concatenate([np.arange(v,v+block)%n for v in begin])[:n]
        seed_index=rng.integers(ns,size=ns)
        estimates[j]=np.mean(np.take(np.take(delta,seed_index,axis=0),ix,axis=1))
    return np.quantile(estimates,[.025,.975])

def independent_ridge(cfg,data,mode):
    transformed={k:transform(v,mode) for k,v in data.items()}
    designs={k:np.column_stack([v[0].reshape(len(v[0]),-1),np.ones(len(v[0]))]) for k,v in transformed.items()}
    X,VX,TX=(designs[k] for k in ('train','val','test'))
    Y,VY=(transformed[k][1].reshape(len(transformed[k][1]),-1) for k in ('train','val'))
    ws=transformed['train'][3].reshape(-1,1) if mode=='history_rawloss' else np.ones((len(X),1))
    vs=transformed['val'][3].reshape(-1,1) if mode=='history_rawloss' else np.ones((len(VX),1))
    penalty=np.eye(X.shape[1]);penalty[-1,-1]=0
    candidates=[]
    for alpha in [.1,1.,10.,100.]:
        W=np.linalg.solve((ws*X).T@(ws*X)+alpha*penalty,(ws*X).T@(ws*Y))
        candidates.append((np.square((VX@W-VY)*vs).mean(),alpha,W))
    _,alpha,W=min(candidates,key=lambda x:(x[0],x[1]))
    resid=Y-X@W
    n=len(TX);s=cfg['ensemble_size'];index=np.random.default_rng(cfg['evaluation_seed']+15).integers(len(X),size=(n,s))
    generated=(TX@W)[:,None]+resid[index]
    _,_,mu,sd,_=transformed['test'];generated=mu[:,None]+sd[:,None]*generated[...,None]
    return alpha,scores(generated,data['test'][1])

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/followup-normalization');p.add_argument('--analysis',default='output/followup-normalization/summary.json');p.add_argument('--output',default='results/followup-normalization-audit/analysis.json');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    folder=ROOT/a.results;cfg=read(folder/'config.json');summary=read(ROOT/a.analysis);checks={};on_disk=[read(p) for p in sorted((folder/'runs').glob('*.json'))]
    # Training may finish more fits after the saved analysis snapshot. Audit the
    # exact snapshot and expose newly completed but not yet analyzed fits.
    absent_at_snapshot=set(summary['missing']);records=[r for r in on_disk if r['name'] not in absent_at_snapshot]
    checks['completed_fits']=close(len(records),summary['completed_fits'])
    expected=sum((len(cfg['priors'])+len(cfg['extra_priors']) if m in ('global','history') else len(cfg['priors']))*len(cfg['heads'])*len(cfg['seeds']) for m in cfg['normalizations'])
    checks['expected_fits']=close(expected,summary['expected_fits'])
    checks['complete_flag']={'passed':summary['complete']==(len(records)==expected)}
    bycell={}
    for r in records:
        for b,br in r['budgets'].items():
            key=f"{r['normalization']}|{r['prior']}|{r['head']}|{b}"
            bycell.setdefault(key,[]).append((r['seed'],br))
    for key,rows in bycell.items():
        rows.sort();table=summary['table'][key]
        checks[key+'/seed_list']=close([s for s,_ in rows],table['seeds'])
        checks[key+'/seed_count']=close(len(rows),table['n_seeds'])
        checks[key+'/crps_list']=close([r['forecast_means']['fair_crps'] for _,r in rows],table['fair_crps_per_seed'])
        checks[key+'/steps']=close([r['selected_step'] for _,r in rows],table['best_step_per_seed'])
        for metric in rows[0][1]['forecast_means']:
            checks[key+'/'+metric]=close(np.mean([r['forecast_means'][metric] for _,r in rows]),table['mean'][metric])
    if set(bycell)!=set(summary['table']):checks['table_keys']={'passed':False}
    if summary['complete']:
        # Frozen study has 28 source/normalization/head cells at each of two
        # budgets, 140 planned paired contrasts and 10 prior/head interactions.
        checks['full_table_coverage']=close(len(summary['table']),56)
        checks['full_contrast_coverage']=close(len(summary['contrasts']),140)
        checks['full_interaction_coverage']=close(len(summary['interactions']),10)
    for name,c in summary['contrasts'].items():
        aa=np.asarray([r['per_context']['fair_crps'] for _,r in sorted(bycell[c['a']])]);bb=np.asarray([r['per_context']['fair_crps'] for _,r in sorted(bycell[c['b']])]);d=aa-bb
        checks[name+'/mean_delta']=close(d.mean(),c['mean_delta']);checks[name+'/mean_ratio']=close(aa.mean()/bb.mean(),c['mean_ratio'])
        checks[name+'/seed_deltas']=close(d.mean(1),c['seed_deltas']);checks[name+'/all_seeds_lower']={'passed':bool((d.mean(1)<0).all())==c['all_seeds_lower']}
        for block in [4,8]:checks[name+f'/block{block}']=close(interval(d,block),c[f'block{block}_interval'],rtol=1e-10,atol=1e-12)
    for key,c in summary['interactions'].items():
        mode,budget=key.split('|')
        vals=[np.asarray([r['per_context']['fair_crps'] for _,r in sorted(bycell[f'{mode}|{prior}|{head}|{budget}'])]) for prior,head in [('gp','s4'),('gp','mlp'),('white','s4'),('white','mlp')]]
        d=vals[0]-vals[1]-vals[2]+vals[3];checks[key+'/interaction_mean']=close(d.mean(),c['mean']);checks[key+'/interaction_seeds']=close(d.mean(1),c['per_seed'])
        for block in [4,8]:checks[key+f'/interaction_block{block}']=close(interval(d,block),c[f'block{block}_interval'],rtol=1e-10,atol=1e-12)
    data,_,scaler=rebuild(cfg)
    for mode in cfg['normalizations']:
        alpha,sc=independent_ridge(cfg,data,mode);saved=summary['baselines'][mode]
        checks['ridge/'+mode+'/alpha']=close(alpha,saved['alpha'])
        for k,v in sc.items():
            checks['ridge/'+mode+'/'+k]=close(v,saved['per_context'][k])
            checks['ridge/'+mode+'/'+k+'/mean']=close(v.mean(),saved['forecast_means'][k])
        checks['ridge/'+mode+'/roughness_ratio']=close(sc['generated_roughness'].mean()/max(sc['target_roughness'].mean(),1e-12),saved['forecast_means']['roughness_ratio'])
    level=data['train'][0].mean(1).ravel();testlevel=data['test'][0].mean(1).ravel();lo,hi=np.quantile(level,[.025,.975])
    checks['test_level_shift']=close(np.mean((testlevel<lo)|(testlevel>hi)),summary['test_level_outside_train95_fraction'])
    checks['test_contexts']=close(len(testlevel),summary['test_contexts'])
    for budget in cfg['budgets']:checks[f'best_at_budget/{budget}']=close(sum(r['budgets'][str(budget)]['selected_step']==budget for r in records),summary['best_at_budget'][str(budget)])
    result={'passed':all(c['passed'] for c in checks.values()),'complete':summary['complete'],'checks':len(checks),
            'completed_fits':len(records),'newer_unanalyzed_fits':len(on_disk)-len(records),'contrasts':len(summary['contrasts']),'interactions':len(summary['interactions']),
            'analysis_sha256':sha(ROOT/a.analysis),'audit_script_sha256':sha(__file__),
            'independent_definitions_sha256':sha(ROOT/'scripts/check_followup_normalization.py'),
            'failures':{k:v for k,v in checks.items() if not v['passed']}}
    save(ROOT/a.output,result);print(json.dumps(result,indent=2))
    if not result['passed'] or (not result['complete'] and not a.allow_partial):raise SystemExit(1)

if __name__=='__main__':main()
