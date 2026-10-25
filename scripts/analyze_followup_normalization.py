#!/usr/bin/env python3
"""Aggregate the frozen ETTh1 factorial; no training or model selection."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fm_stress.experiment import load_config, save_json
from fm_stress.extended_data import build_data
from fm_stress.extended_metrics import forecast_scores, means
from fm_stress.followup_normalization import transform, identities, sha


def block_interval(delta,block=4,repeats=2000,seed=414141):
    """Paired seed resampling and circular moving blocks of chronological contexts.

    A descriptive within-period uncertainty interval, not independent datasets.
    delta shape [training seeds, chronological test contexts].
    """
    rng=np.random.default_rng(seed);nseed,n=delta.shape
    values=[]
    for _ in range(repeats):
        starts=rng.integers(n,size=int(np.ceil(n/block)))
        indices=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        seeds=rng.integers(nseed,size=nseed)
        values.append(delta[seeds][:,indices].mean())
    return np.quantile(values,[.025,.975]).tolist()


def baseline(cfg,bundle,mode):
    data={s:transform(ds,mode) for s,ds in bundle.splits.items()}
    tr,va,te=(data[s][0] for s in ('train','val','test'))
    design=lambda ds:np.column_stack([ds.context.reshape(len(ds),-1),np.ones(len(ds))])
    X,VX,TX=map(design,(tr,va,te));Y=tr.target.reshape(len(tr),-1);VY=va.target.reshape(len(va),-1)
    wtr=data['train'][2].reshape(-1,1) if mode=='history_rawloss' else np.ones((len(tr),1))
    wva=data['val'][2].reshape(-1,1) if mode=='history_rawloss' else np.ones((len(va),1))
    P=np.eye(X.shape[1]);P[-1,-1]=0
    fits=[]
    for a in [.1,1.,10.,100.]:
        W=np.linalg.solve((X*wtr).T@(X*wtr)+a*P,(X*wtr).T@(Y*wtr))
        fits.append((float(np.mean(((VX@W-VY)*wva)**2)),a,W))
    _,alpha,W=min(fits,key=lambda r:r[:2]);residuals=Y-X@W
    ix=np.random.default_rng(cfg['evaluation_seed']+15).integers(len(Y),size=(len(te),cfg['ensemble_size']))
    generated=(TX@W)[:,None]+residuals[ix]
    mu,sd=data['test'][1:];generated=mu[:,None]+sd[:,None]*generated[...,None]
    scores=forecast_scores(generated,bundle.splits['test'].target)
    return {'alpha':alpha,'forecast_means':means(scores),'per_context':{k:v.tolist() for k,v in scores.items()}}


def analyze(results,out,allow_partial=False):
    cfg=load_config(results/'config.json');fp=load_config(results/'manifest.json')['fingerprint']
    records=[];missing=[]
    for mode,prior,head,seed in identities(cfg):
        name=f'{mode}_{prior}_{head}_seed{seed}';p=results/'runs'/f'{name}.json'
        if not p.exists():missing.append(name);continue
        r=load_config(p)
        if r['fingerprint']!=fp or r['status']!='complete':raise ValueError('Incompatible '+name)
        records.append(r)
    if missing and not allow_partial:raise ValueError(f'Missing {len(missing)} fits')
    out.mkdir(parents=True,exist_ok=True)
    flat=[];cells={};byid={}
    for r in records:
        for b,br in r['budgets'].items():
            key=f"{r['normalization']}|{r['prior']}|{r['head']}|{b}"
            cells.setdefault(key,[]).append((r['seed'],br))
            byid[(r['normalization'],r['prior'],r['head'],int(b),r['seed'])]=br
            flat.append({k:r[k] for k in ('normalization','prior','head','seed')}|{'budget':int(b),'selected_step':br['selected_step']}|br['forecast_means'])
    table={}
    for k,rows in cells.items():
        rows.sort()
        table[k]={'n_seeds':len(rows),'seeds':[s for s,_ in rows],
                  'mean':{m:float(np.mean([v['forecast_means'][m] for _,v in rows])) for m in rows[0][1]['forecast_means']},
                  'fair_crps_per_seed':[v['forecast_means']['fair_crps'] for _,v in rows],
                  'best_step_per_seed':[v['selected_step'] for _,v in rows]}
    contrasts={}
    def contrast(label,a,b):
        if a not in cells or b not in cells or len(cells[a])!=len(cfg['seeds']) or len(cells[b])!=len(cfg['seeds']):return
        aa=np.array([v['per_context']['fair_crps'] for _,v in sorted(cells[a])])
        bb=np.array([v['per_context']['fair_crps'] for _,v in sorted(cells[b])]);d=aa-bb
        contrasts[label]={'a':a,'b':b,'mean_delta':float(d.mean()),'mean_ratio':float(aa.mean()/bb.mean()),
                          'seed_deltas':d.mean(1).tolist(),'all_seeds_lower':bool(np.all(d.mean(1)<0)),
                          'block4_interval':block_interval(d,4),'block8_interval':block_interval(d,8)}
    pairs=[('center','global'),('scale','global'),('history','global'),('history','center'),('history','scale'),('history_rawloss','history')]
    for budget in cfg['budgets']:
        for prior in cfg['priors']:
            for head in cfg['heads']:
                for a,b in pairs:contrast(f'{a}-{b}|{prior}|{head}|{budget}',f'{a}|{prior}|{head}|{budget}',f'{b}|{prior}|{head}|{budget}')
        for mode in cfg['normalizations']:
            for prior in cfg['priors']+(cfg['extra_priors'] if mode in ('global','history') else []):
                contrast(f's4-mlp|{mode}|{prior}|{budget}',f'{mode}|{prior}|s4|{budget}',f'{mode}|{prior}|mlp|{budget}')
            for head in cfg['heads']:
                contrast(f'gp-white|{mode}|{head}|{budget}',f'{mode}|gp|{head}|{budget}',f'{mode}|white|{head}|{budget}')
                if mode in ('global','history'):
                    for prior in cfg['extra_priors']:contrast(f'{prior}-gp|{mode}|{head}|{budget}',f'{mode}|{prior}|{head}|{budget}',f'{mode}|gp|{head}|{budget}')
    for key in cells:
        mode,prior,head,b=key.split('|')
        if int(b)==cfg['budgets'][-1]:contrast(f'budget|{mode}|{prior}|{head}',key,f'{mode}|{prior}|{head}|{cfg["budgets"][0]}')
    interactions={}
    for mode in cfg['normalizations']:
        for budget in cfg['budgets']:
            ks=[f'{mode}|{p}|{h}|{budget}' for p,h in [('gp','s4'),('gp','mlp'),('white','s4'),('white','mlp')]]
            if not all(k in cells and len(cells[k])==len(cfg['seeds']) for k in ks):continue
            x=[np.array([v['per_context']['fair_crps'] for _,v in sorted(cells[k])]) for k in ks];d=x[0]-x[1]-x[2]+x[3]
            interactions[f'{mode}|{budget}']={'mean':float(d.mean()),'per_seed':d.mean(1).tolist(),'block4_interval':block_interval(d,4),'block8_interval':block_interval(d,8)}
    bundle=build_data(cfg['dataset']);baselines={m:baseline(cfg,bundle,m) for m in cfg['normalizations']}
    level=bundle.splits['train'].context.mean(1).ravel();lo,hi=np.quantile(level,[.025,.975])
    testlevel=bundle.splits['test'].context.mean(1).ravel()
    summary={'complete':not missing,'expected_fits':len(identities(cfg)),'completed_fits':len(records),'missing':missing,
             'fingerprint':fp,'config_sha256':sha(results/'config.json'),'table':table,'contrasts':contrasts,'interactions':interactions,
             'baselines':baselines,'test_contexts':len(testlevel),'test_level_outside_train95_fraction':float(np.mean((testlevel<lo)|(testlevel>hi))),
             'best_at_budget':{str(b):sum(r['budgets'][str(b)]['selected_step']==b for r in records) for b in cfg['budgets']},
             'uncertainty_note':'Descriptive paired seed and circular context-block bootstrap; one previously inspected test period. No independent-series or causal-mechanism claim.'}
    save_json(out/'summary.json',summary)
    if flat:
        with (out/'runs.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    figures(summary,out,cfg)
    print(json.dumps({'complete':summary['complete'],'fits':len(records),'contrasts':len(contrasts),'interactions':interactions},indent=2))
    return summary


def figures(s,out,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    modes=cfg['normalizations'];names=['Global','Center only','Scale only','Center + scale','History + raw loss']
    for budget in cfg['budgets']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
        for ax,prior in zip(axes,cfg['priors']):
            for j,(head,color) in enumerate([('mlp','#24618f'),('s4','#c76a38')]):
                xs=[];ys=[]
                for i,mode in enumerate(modes):
                    cell=s['table'].get(f'{mode}|{prior}|{head}|{budget}')
                    if cell and cell['n_seeds']==3:
                        xs.append(i);ys.append(cell['mean']['fair_crps']);ax.scatter([i+(j-.5)*.10]*3,cell['fair_crps_per_seed'],s=18,alpha=.55,color=color)
                ax.plot(xs,ys,'o-',color=color,label=head.upper())
            ax.set_title(f'{prior.upper()} source');ax.set_xticks(range(len(modes)),names,rotation=25,ha='right');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
        axes[0].set_ylabel('Fair CRPS (lower is better)')
        fig.suptitle(f'ETTh1: which transformation matters? Budget {budget:,} updates')
        fig.tight_layout();fig.savefig(out/f'normalization_b{budget}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    priors=['white','gp','gp_zero_level','gp_white_level'];labels=['White','GP','GP: zero level','GP: white level'];budget=cfg['budgets'][-1]
    for ax,mode in zip(axes,['global','history']):
        for head,color in [('mlp','#24618f'),('s4','#c76a38')]:
            vals=[s['table'].get(f'{mode}|{p}|{head}|{budget}',{}).get('mean',{}).get('fair_crps',np.nan) for p in priors]
            ax.plot(range(4),vals,'o-',color=color,label=head.upper())
        ax.set_title(mode.capitalize());ax.set_xticks(range(4),labels,rotation=22,ha='right');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    axes[0].set_ylabel('Fair CRPS (lower is better)');fig.suptitle('Changing only the GP source level component')
    fig.tight_layout();fig.savefig(out/'source_level.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results/followup-normalization')
    p.add_argument('--output',type=Path,default=ROOT/'output/followup-normalization');p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();analyze(a.results,a.output,a.allow_partial)
