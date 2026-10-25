"""Post-review normalization/GP-level factorial. Original experiments are immutable."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from .experiment import config_hash, load_config, save_json, tensor, predict, generate
from .extended_data import build_data, WindowDataset
from .extended_experiment import evaluation_indices
from .extended_metrics import forecast_scores, means
from .extended_models import create_extended_model, parameter_counts
from .models import optimizer_groups

ROOT = Path(__file__).resolve().parents[1]
NORMALIZATIONS = ('global', 'center', 'scale', 'history', 'history_rawloss')
PRIORS = ('white', 'gp', 'gp_zero_level', 'gp_white_level')
SOURCE_FILES = ('fm_stress/followup_normalization.py', 'fm_stress/extended_data.py',
                'fm_stress/extended_experiment.py', 'fm_stress/extended_metrics.py',
                'fm_stress/extended_models.py', 'fm_stress/models.py', 'fm_stress/experiment.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def transform(ds, mode):
    if mode not in NORMALIZATIONS:
        raise ValueError(mode)
    mu = ds.context.mean(axis=1, keepdims=True) if mode in ('center', 'history', 'history_rawloss') else np.zeros_like(ds.context[:, :1])
    sd = ds.context.std(axis=1, keepdims=True) if mode in ('scale', 'history', 'history_rawloss') else np.ones_like(mu)
    sd = np.where(sd > .01, sd, 1.)
    return WindowDataset((ds.context-mu)/sd, (ds.target-mu)/sd, ds.starts), mu, sd


def prior_numpy(covariances, n, prior, seed, horizon):
    if prior not in PRIORS:
        raise ValueError(prior)
    family = 'white' if prior == 'white' else 'gp'
    z = np.random.default_rng(seed).standard_normal((n, horizon))
    eps = (z @ np.linalg.cholesky(covariances[family]).T)[..., None]
    if prior in ('gp_zero_level', 'gp_white_level'):
        eps = eps - eps.mean(axis=1, keepdims=True)
    if prior == 'gp_white_level':
        eps = eps + np.random.default_rng(seed+40000).standard_normal((n, 1, 1))/np.sqrt(horizon)
    return eps


def training_prior(z, chol, prior, horizon, level=None):
    eps = (z @ chol.T).reshape(len(z), horizon, 1)
    if prior in ('gp_zero_level', 'gp_white_level'):
        eps = eps - eps.mean(dim=1, keepdim=True)
    if prior == 'gp_white_level':
        if level is None:
            raise ValueError('Replacement-level normal draw is required')
        eps = eps + level/np.sqrt(horizon)
    return eps


def objective(prediction, target, sd, mode):
    error = prediction-target
    return (error*sd if mode == 'history_rawloss' else error).square().mean()


def identities(cfg):
    return [(mode, prior, head, seed) for mode in cfg['normalizations']
            for prior in (cfg['priors']+cfg['extra_priors'] if mode in ('global', 'history') else cfg['priors'])
            for head in cfg['heads'] for seed in cfg['seeds']]


def prepare(cfg, out):
    out = Path(out)
    manifest = {'config_sha256':config_hash(cfg), 'sources':{p:sha(ROOT/p) for p in SOURCE_FILES},
                'dataset_sha256':sha(ROOT/cfg['dataset']['path'])}
    manifest['fingerprint'] = config_hash(manifest)
    for name, value in (('config.json',cfg),('manifest.json',manifest)):
        path = out/name
        if path.exists() and load_config(path) != value:
            raise ValueError('Frozen study changed: '+str(path))
        save_json(path,value)
    return manifest['fingerprint']


def evaluate_budget(cfg, bundle, model, mode, prior, budget, best_step, out, name, device):
    # Test transforms, pairs and forecasting are first computed after ALL training.
    ds, mu, sd = transform(bundle.splits['test'], mode)
    if cfg['n_ensemble_contexts'] != len(ds):
        raise ValueError('This protocol scores all test windows; n_ensemble_contexts must equal split size')
    f = ds.target.shape[1]
    eps = prior_numpy(bundle.source_covariances,len(ds),prior,cfg['evaluation_seed']+100,f)
    tt = np.random.default_rng(cfg['evaluation_seed']+101).random(len(ds))
    tx=(1-tt[:,None,None])*eps+tt[:,None,None]*ds.target
    tu=ds.target-eps
    with torch.no_grad():
        prediction=predict(model,tensor(tx,device),tensor(tt,device),tensor(ds.context,device)).cpu().numpy()
    n,s=len(ds),cfg['ensemble_size']
    initial=prior_numpy(bundle.source_covariances,n*s,prior,cfg['evaluation_seed']+200,f)
    hist=np.repeat(ds.context,s,axis=0)
    generated_model=generate(model,tensor(initial,device),tensor(hist,device),cfg['euler_steps'],cfg['generation_batch_size']).reshape(n,s,f,1)
    generated=mu[:,None]+sd[:,None]*generated_model
    scores=forecast_scores(generated,bundle.splits['test'].target)
    original_indices=evaluation_indices(ds,64)
    # Prefix members kept constant, paired within the new study; old CUDA scores are not pooled.
    source_global=mu[:,None]+sd[:,None]*initial.reshape(n,s,f,1)
    output=out/'samples'/f'{name}_b{budget}.npz'
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,generated=generated,generated_model=generated_model,target=bundle.splits['test'].target,
                        prediction=prediction,velocity=tu,test_x=tx,test_time=tt,history_model=ds.context,
                        location=mu,scale=sd,ensemble_source=initial.reshape(n,s,f,1),starts=ds.starts)
    record={'budget':budget,'selected_step':best_step,'forecast_means':means(scores),
            'per_context':{k:v.tolist() for k,v in scores.items()},
            'original64_context_means':{k:float(v[original_indices].mean()) for k,v in scores.items()},
            'velocity_mse_model_units':float(np.mean((prediction-tu)**2)),
            'velocity_mse_global_units':float(np.mean(((prediction-tu)*sd)**2)),
            'prior_global_mean_level':float(source_global.mean()),
            'prior_global_mean_variance':float(source_global.var(axis=1).mean()),
            'samples_sha256':sha(output)}
    if budget==cfg['budgets'][-1] and cfg['solver_check']:
        ni=8
        fine=generate(model,tensor(initial[:ni*s],device),tensor(hist[:ni*s],device),128,cfg['generation_batch_size']).reshape(ni,s,f,1)
        fine_global=mu[:ni,None]+sd[:ni,None]*fine
        fine_scores=forecast_scores(fine_global,bundle.splits['test'].target[:ni])
        record['solver_check']={'contexts':ni,'euler_steps':128,
                                'endpoint_rmse':float(np.sqrt(np.mean((fine_global-generated[:ni])**2))),
                                'fair_crps_change':float(fine_scores['fair_crps'].mean()-scores['fair_crps'][:ni].mean())}
    return record


def train_run(cfg,bundle,out,mode,prior,head,seed,fingerprint,device):
    out=Path(out);name=f'{mode}_{prior}_{head}_seed{seed}';rp=out/'runs'/f'{name}.json'
    if rp.exists():
        record=load_config(rp)
        assert record['fingerprint']==fingerprint
        for budget in cfg['budgets']:
            assert sha(out/'checkpoints'/f'{name}_b{budget}.pt')==record['budgets'][str(budget)]['checkpoint_sha256']
            assert sha(out/'samples'/f'{name}_b{budget}.npz')==record['budgets'][str(budget)]['samples_sha256']
        print('SKIP',name,flush=True);return record
    started=time.perf_counter();options={**cfg['model'],'head':head}
    torch.manual_seed(seed);model=create_extended_model(**options).to(device)
    optimizer=torch.optim.AdamW(optimizer_groups(model,cfg['learning_rate'],cfg['weight_decay']))
    train,_,trsd=transform(bundle.splits['train'],mode)
    val,_,vsd=transform(bundle.splits['val'],mode)
    y,h,train_sd=tensor(train.target,device),tensor(train.context,device),tensor(trsd,device)
    f=cfg['model']['horizon'];family='white' if prior=='white' else 'gp'
    chol=tensor(np.linalg.cholesky(bundle.source_covariances[family]),device)
    ve=prior_numpy(bundle.source_covariances,len(val),prior,cfg['evaluation_seed'],f)
    vt=np.random.default_rng(cfg['evaluation_seed']+1).random(len(val))
    vx=(1-vt[:,None,None])*ve+vt[:,None,None]*val.target;vu=val.target-ve
    vx,vu,vt,vh,vsd=[tensor(x,device) for x in (vx,vu,vt,val.context,vsd)]
    torch.manual_seed(seed+30000)
    if device.type=='mps':torch.mps.manual_seed(seed+30000)
    level_generator=torch.Generator().manual_seed(seed+40000)
    best=float('inf');best_step=0;curve=[];batch_hashes={};selected={}
    checkpoint=out/'checkpoints'/f'{name}_current.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    print('START',name,'device',device,flush=True)
    for step in range(1,cfg['steps']+1):
        model.train();ix=torch.randint(len(y),(cfg['batch_size'],),device=device)
        z=torch.randn(len(ix),f,device=device)
        level=torch.randn(len(ix),1,1,generator=level_generator).to(device) if prior=='gp_white_level' else None
        eps=training_prior(z,chol,prior,f,level);t=torch.rand(len(ix),device=device)
        if step in (1,cfg['steps']):
            batch_hashes[str(step)]={k:array_sha(a.detach().cpu().numpy()) for k,a in [('indices',ix),('normals',z),('time',t)]}
        yy,hh=y[ix],h[ix];x=(1-t[:,None,None])*eps+t[:,None,None]*yy;u=yy-eps
        optimizer.zero_grad(set_to_none=True)
        loss=objective(model(x,t,hh),u,train_sd[ix],mode);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']);optimizer.step()
        if step%cfg['eval_every']==0 or step in cfg['budgets']:
            model.eval()
            with torch.no_grad():
                vp=predict(model,vx,vt,vh);validation=float(objective(vp,vu,vsd,mode).cpu())
            if not np.isfinite(validation):raise FloatingPointError(name)
            curve.append({'step':step,'train_loss':float(loss.detach().cpu()),'validation_loss':validation,'seconds':time.perf_counter()-started})
            if validation<best:
                best,best_step=validation,step
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'options':options,
                            'step':step,'mode':mode,'prior':prior,'seed':seed,'fingerprint':fingerprint},checkpoint)
            if step in cfg['budgets']:
                shutil.copyfile(checkpoint,out/'checkpoints'/f'{name}_b{step}.pt');selected[step]=best_step
            save_json(out/'training'/f'{name}.json',curve)
            if step in cfg['budgets']:print('BUDGET',name,step,'best',best_step,'seconds',round(time.perf_counter()-started,1),flush=True)
    budgets={}
    for budget in cfg['budgets']:
        bp=out/'checkpoints'/f'{name}_b{budget}.pt'
        model.load_state_dict(torch.load(bp,map_location=device,weights_only=True)['state_dict']);model.eval()
        budgets[str(budget)]=evaluate_budget(cfg,bundle,model,mode,prior,budget,selected[budget],out,name,device)
        budgets[str(budget)]['checkpoint_sha256']=sha(bp)
    record={'name':name,'status':'complete','normalization':mode,'prior':prior,'head':head,'seed':seed,
            'fingerprint':fingerprint,'config_sha256':config_hash(cfg),'device':str(device),'torch':torch.__version__,
            'parameters':parameter_counts(model),'budgets':budgets,'curve':curve,'batch_hashes':batch_hashes,
            'seconds':time.perf_counter()-started}
    save_json(rp,record);checkpoint.unlink()
    print('DONE',name,'CRPS',round(budgets[str(cfg['budgets'][-1])]['forecast_means']['fair_crps'],6),'seconds',round(record['seconds'],1),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/followup_normalization.json');p.add_argument('--output',default='results/followup-normalization')
    p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--head');p.add_argument('--seed',type=int)
    p.add_argument('--mode');p.add_argument('--prior');p.add_argument('--limit',type=int)
    a=p.parse_args();cfg=load_config(a.config);device=torch.device(a.device)
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS required; no silent fallback')
    torch.set_num_threads(2)
    environment=Path(a.output)/'environment.json'
    if environment.exists() and load_config(environment)['device'] != str(device):
        raise ValueError('Different device already recorded; use a separate output directory')
    fp=prepare(cfg,a.output);bundle=build_data(cfg['dataset'])
    save_json(environment,{'torch':torch.__version__,'numpy':np.__version__,'device':str(device),'platform':platform.platform()})
    rows=[r for r in identities(cfg) if (a.mode is None or r[0]==a.mode) and (a.prior is None or r[1]==a.prior)
          and (a.head is None or r[2]==a.head) and (a.seed is None or r[3]==a.seed)]
    for mode,prior,head,seed in rows[:a.limit]:train_run(cfg,bundle,a.output,mode,prior,head,seed,fp,device)


if __name__=='__main__':main()
