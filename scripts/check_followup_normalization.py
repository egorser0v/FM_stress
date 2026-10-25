#!/usr/bin/env python3
"""Independent chronological-data, score and CPU-checkpoint audit of the new normalization study.

Reconstructs CSV splits/scaling and source draws without production data or metric
helpers. Replays all test/validation velocities and two complete 64-member forecast
contexts per checkpoint. Original experiments and runner are never changed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fm_stress.extended_models import create_extended_model

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def close(got,want,rtol=1e-7,atol=1e-10):
    a,b=np.asarray(got,float),np.asarray(want,float)
    return {'passed':bool(a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol)),
            'max_abs':float(np.max(np.abs(a-b),initial=0)),
            'max_relative':float(np.max(np.abs(a-b)/(np.abs(b)+1e-15),initial=0))}

def rebuild(cfg):
    d=cfg['dataset'];path=ROOT/d['path']
    with path.open(newline='') as f:
        raw=np.asarray([[float(row[c]) for c in d['columns']] for row in csv.DictReader(f)])
    ends=d['split_ends'];mean=raw[:ends[0]].mean(0);sd=raw[:ends[0]].std(0)
    sd=np.where(sd>1e-8,sd,1.);values=(raw-mean)/sd;l,h=d['lookback'],d['horizon']
    out={}
    for split,start,end in zip(('train','val','test'),[0]+ends[:-1],ends):
        starts=np.arange(start,end-l-h+1,d['strides'].get(split,1))
        cap=d.get('max_windows',{}).get(split)
        if cap and len(starts)>cap:starts=starts[np.linspace(0,len(starts)-1,cap,dtype=int)]
        out[split]=(np.stack([values[i:i+l] for i in starts]),np.stack([values[i+l:i+l+h] for i in starts]),starts)
    if len(d['columns'])!=1:raise ValueError('Audit intentionally targets univariate ETTh1 factorial')
    lag=np.subtract.outer(np.arange(h),np.arange(h))/d.get('source_lengthscale',8.)
    nug=d.get('source_nugget',1e-6)
    covariance=(np.exp(-.5*lag*lag)+nug*np.eye(h))/(1+nug)
    return out,{'white':np.eye(h),'gp':covariance},{'train_mean':mean.tolist(),'train_sd':sd.tolist(),'training_end':ends[0],'split_sizes':{k:len(v[0]) for k,v in out.items()}}

def transform(data,mode):
    h,y,starts=data
    mu=h.mean(1,keepdims=True) if mode in ('center','history','history_rawloss') else np.zeros_like(h[:,:1])
    sd=h.std(1,keepdims=True) if mode in ('scale','history','history_rawloss') else np.ones_like(h[:,:1])
    sd=np.where(sd>.01,sd,1.)
    return (h-mu)/sd,(y-mu)/sd,mu,sd,starts

def draws(covs,prior,n,seed):
    cov=covs['white' if prior=='white' else 'gp'];f=len(cov)
    eps=(np.random.default_rng(seed).standard_normal((n,f))@np.linalg.cholesky(cov).T)[...,None]
    if prior in ('gp_zero_level','gp_white_level'):eps-=eps.mean(1,keepdims=True)
    if prior=='gp_white_level':eps+=np.random.default_rng(seed+40000).standard_normal((n,1,1))/np.sqrt(f)
    return eps

def scores(samples,target):
    x,y=np.asarray(samples,float),np.asarray(target,float);n,s,f,c=x.shape
    rows={k:[] for k in ('fair_crps','joint_energy_scaled','channel_sum_crps_scaled','mean_forecast_mse','coverage_90','width_90','generated_roughness','target_roughness')}
    for start in range(0,n,8):
        a,b=x[start:start+8],y[start:start+8];af=a.reshape(len(a),s,-1);bf=b.reshape(len(b),-1)
        diff=af[:,:,None,:]-af[:,None,:,:]
        crps=(np.abs(af-bf[:,None]).mean(1)-np.abs(diff).sum((1,2))/(2*s*(s-1))).mean(1)
        energy=(np.linalg.norm(af-bf[:,None],axis=-1).mean(1)-np.linalg.norm(diff,axis=-1).sum((1,2))/(2*s*(s-1)))/np.sqrt(f*c)
        lo,hi=np.quantile(af,[.05,.95],axis=1)
        values={'fair_crps':crps,'joint_energy_scaled':energy,'channel_sum_crps_scaled':crps,
            'mean_forecast_mse':((a.mean(1)-b)**2).mean((1,2)),
            'coverage_90':((bf>=lo)&(bf<=hi)).mean(1),'width_90':(hi-lo).mean(1),
            'generated_roughness':np.abs(np.diff(a,axis=2)).mean((1,2,3)),
            'target_roughness':np.abs(np.diff(b,axis=1)).mean((1,2))}
        for k,v in values.items():rows[k].append(v)
    return {k:np.concatenate(v) for k,v in rows.items()}

@torch.no_grad()
def predict(model,x,t,h):
    return model(torch.tensor(x,dtype=torch.float32),torch.tensor(t,dtype=torch.float32),torch.tensor(h,dtype=torch.float32)).numpy()

@torch.no_grad()
def euler(model,initial,h,steps):
    x=torch.tensor(initial,dtype=torch.float32);h=torch.tensor(h,dtype=torch.float32)
    for k in range(steps):x=x+model(x,torch.full((len(x),),k/steps),h)/steps
    return x.numpy()

def check_one(cfg,record,budget,data,covs,folder,out):
    name=record['name'];b=record['budgets'][str(budget)];sample=folder/'samples'/f'{name}_b{budget}.npz';checkpoint=folder/'checkpoints'/f'{name}_b{budget}.pt'
    ident={'record_sha256':sha(folder/'runs'/f'{name}.json'),'samples_sha256':sha(sample),
           'checkpoint_sha256':sha(checkpoint),'audit_script_sha256':sha(__file__),'dataset_sha256':sha(ROOT/cfg['dataset']['path'])}
    ident.update(config_sha256=sha(folder/'config.json'),manifest_sha256=sha(folder/'manifest.json'))
    dest=out/'runs'/f'{name}_b{budget}.json'
    if dest.exists():
        old=read(dest)
        if old['identity']==ident:return old
    checks={};a=np.load(sample);mode,prior=record['normalization'],record['prior'];h,y,mu,sd,starts=transform(data['test'],mode)
    n,s,f=len(h),cfg['ensemble_size'],cfg['model']['horizon'];seed=cfg['evaluation_seed']
    eps=draws(covs,prior,n,seed+100);t=np.random.default_rng(seed+101).random(n)
    x=(1-t[:,None,None])*eps+t[:,None,None]*y;u=y-eps;initial=draws(covs,prior,n*s,seed+200).reshape(n,s,f,1)
    checks['samples_hash']={'passed':ident['samples_sha256']==b['samples_sha256']}
    checks['checkpoint_hash']={'passed':ident['checkpoint_sha256']==b['checkpoint_sha256']}
    for k,v in {'target':data['test'][1],'test_x':x,'test_time':t,'velocity':u,'history_model':h,'location':mu,'scale':sd,'ensemble_source':initial,'starts':starts}.items():checks['rebuilt_'+k]=close(v,a[k],rtol=1e-12,atol=1e-12)
    checks['inverse_forecast']=close(mu[:,None]+sd[:,None]*a['generated_model'],a['generated'],rtol=1e-12,atol=1e-12)
    sc=scores(a['generated'],a['target']);means={k:float(v.mean()) for k,v in sc.items()}
    means['roughness_ratio']=means['generated_roughness']/max(means['target_roughness'],1e-12)
    for k,v in sc.items():checks['per_context_'+k]=close(v,b['per_context'][k])
    for k,v in means.items():checks['forecast_mean_'+k]=close(v,b['forecast_means'][k])
    ix=np.linspace(0,n-1,min(n,64),dtype=np.int64)
    for k,v in sc.items():checks['original64_'+k]=close(v[ix].mean(),b['original64_context_means'][k])
    checks['velocity_model']=close(np.mean((a['prediction']-u)**2),b['velocity_mse_model_units'])
    checks['velocity_global']=close(np.mean(((a['prediction']-u)*sd)**2),b['velocity_mse_global_units'])
    source_global=mu[:,None]+sd[:,None]*initial
    checks['prior_mean']=close(source_global.mean(),b['prior_global_mean_level'])
    checks['prior_variance']=close(source_global.var(1).mean(),b['prior_global_mean_variance'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=create_extended_model(**state['options']);model.load_state_dict(state['state_dict']);model.eval()
    checks['checkpoint_identity']={'passed':all(state[k]==v for k,v in {'mode':mode,'prior':prior,'seed':record['seed'],'fingerprint':record['fingerprint']}.items()) and state['options']['head']==record['head']}
    eligible=[r for r in record['curve'] if r['step']<=budget];best=min(eligible,key=lambda r:r['validation_loss'])
    checks['validation_prefix_selection']={'passed':state['step']==best['step']==b['selected_step']}
    pp=predict(model,x,t,h);checks['cpu_all_test_velocity']=close(pp,a['prediction'],rtol=1e-4,atol=2e-6)
    vh,vy,_,vsd,_=transform(data['val'],mode);ve=draws(covs,prior,len(vh),seed);vt=np.random.default_rng(seed+1).random(len(vh));vx=(1-vt[:,None,None])*ve+vt[:,None,None]*vy
    pv=predict(model,vx,vt,vh);err=pv-(vy-ve)
    if mode=='history_rawloss':err=err*vsd
    checks['cpu_validation_objective']=close(np.square(err).mean(),best['validation_loss'],rtol=1e-4,atol=1e-7)
    ni=min(2,n);raw=euler(model,initial[:ni].reshape(-1,f,1),np.repeat(h[:ni],s,axis=0),cfg['euler_steps']).reshape(ni,s,f,1)
    checks['cpu_euler_model']=close(raw,a['generated_model'][:ni],rtol=1e-4,atol=3e-5)
    checks['cpu_euler_global']=close(mu[:ni,None]+sd[:ni,None]*raw,a['generated'][:ni],rtol=1e-4,atol=3e-5)
    result={'name':name,'budget':budget,'identity':ident,'passed':all(v['passed'] for v in checks.values()),
            'n_test_pairs':n,'n_forecast_contexts':ni,'members':s,'checks':checks,'forecast_means':means}
    save(dest,result);print(name,budget,'PASS' if result['passed'] else 'FAIL',flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/followup-normalization');p.add_argument('--output',default='results/followup-normalization-audit');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    folder=ROOT/a.results;out=ROOT/a.output;cfg=read(folder/'config.json');manifest=read(folder/'manifest.json');torch.set_num_threads(2)
    data,covs,scaler=rebuild(cfg);all_results=[];records=[];missing=[];bad=[]
    stable_hash=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    if stable_hash(cfg)!=manifest['config_sha256']:bad.append('Frozen config hash mismatch')
    if stable_hash({k:v for k,v in manifest.items() if k!='fingerprint'})!=manifest['fingerprint']:bad.append('Manifest fingerprint mismatch')
    if manifest['dataset_sha256']!=sha(ROOT/cfg['dataset']['path']):bad.append('Dataset hash mismatch')
    for name,hsh in manifest['sources'].items():
        if sha(ROOT/name)!=hsh:bad.append('Frozen source changed: '+name)
    for mode in cfg['normalizations']:
        for prior in cfg['priors']+(cfg['extra_priors'] if mode in ('global','history') else []):
            for head in cfg['heads']:
                for seed in cfg['seeds']:
                    name=f'{mode}_{prior}_{head}_seed{seed}';path=folder/'runs'/f'{name}.json'
                    if not path.exists():missing.append(name);continue
                    rec=read(path);records.append(rec)
                    if rec['fingerprint']!=manifest['fingerprint'] or rec['status']!='complete':bad.append(name+': fingerprint/status')
                    if (rec['normalization'],rec['prior'],rec['head'],rec['seed'])!=(mode,prior,head,seed) or rec['name']!=name:bad.append(name+': identity')
                    for budget in cfg['budgets']:all_results.append(check_one(cfg,rec,budget,data,covs,folder,out))
    for seed in cfg['seeds']:
        rows=[r for r in records if r['seed']==seed]
        if rows and any(r['batch_hashes']!=rows[0]['batch_hashes'] for r in rows):bad.append('Unpaired training random draws for seed '+str(seed))
    expected_names={r['name'] for r in records}|set(missing)
    extra={p.stem for p in (folder/'runs').glob('*.json')}-expected_names
    if extra:bad.append('Unexpected run identities: '+str(sorted(extra)))
    failures=[{'name':r['name'],'budget':r['budget'],'checks':{k:v for k,v in r['checks'].items() if not v['passed']}} for r in all_results if not r['passed']]
    summary={'complete':not missing,'passed':not bad and not failures,'expected_fits':len(records)+len(missing),
        'verified_fits':len(records),'verified_budget_artifacts':len(all_results),'missing':missing,'problems':bad,
        'failed_checks':failures,'scaler_reconstruction':scaler,'cpu_threads':2,'torch':torch.__version__,
        'source_manifest_sha256':sha(folder/'manifest.json'),'config_sha256':sha(folder/'config.json'),
        'audit_script_sha256':sha(__file__)}
    save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('missing','failed_checks')},indent=2))
    if bad or failures or (missing and not a.allow_partial):raise SystemExit(1)

if __name__=='__main__':main()
