"""Frozen-budget exploratory joint-channel/source/head comparison on MPS."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .extended_data import build_data, sample_source
from .extended_models import create_extended_model, parameter_counts
from .extended_metrics import forecast_scores, means
from .experiment import save_json, config_hash, tensor, predict, generate
from .metrics import fit_pca, velocity_metrics
from .models import optimizer_groups


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = ['extended_experiment.py','extended_data.py','extended_models.py','extended_metrics.py',
             'models.py','metrics.py','experiment.py','ensemble_evaluation.py','data.py','robustness.py']
    return {p: hashlib.sha256((root/'fm_stress'/p).read_bytes()).hexdigest() for p in files}


def model_options(ds, head, width, cfg):
    return dict(head=head, lookback=ds.context.shape[1], horizon=ds.target.shape[1],
                channels=ds.target.shape[2], width=width, density=cfg['density'])


def select_widths(ds, cfg):
    # Capacity only; no observed validation/test outcomes enter this choice.
    selected = {}
    for head in cfg['heads']:
        choices = []
        for width in range(4, 113):
            torch.manual_seed(0)
            model = create_extended_model(**model_options(ds, head, width, cfg))
            counts = parameter_counts(model)
            choices.append((abs(counts['effective_trainable']-cfg['parameter_target']), width, counts))
        _, width, counts = min(choices)
        selected[head] = {'width': width, **counts}
    return selected


def pairs(bundle, split, source, seed):
    ds = bundle.splits[split]
    e = sample_source(bundle, len(ds), source, seed).astype(np.float32)
    t = np.random.default_rng(seed+1).random(len(ds)).astype(np.float32)
    y = ds.target.astype(np.float32)
    return (1-t[:,None,None])*e+t[:,None,None]*y, y-e, t, e


def geometry(bundle, source, seed):
    train, val = bundle.splits['train'], bundle.splits['val']
    e = sample_source(bundle, len(train), source, seed)
    pca = fit_pca((train.target-e).reshape(len(train), -1))
    ev = sample_source(bundle, len(val), source, seed+1)
    u = (val.target-ev).reshape(len(val), -1)
    variance = pca.project(u, center=False).var(0, ddof=1)
    energy = variance/variance.sum()
    channel_e2 = []
    for c in range(train.target.shape[-1]):
        cp = fit_pca(train.target[:,:,c]-e[:,:,c])
        vv = cp.project(val.target[:,:,c]-ev[:,:,c], center=False).var(0, ddof=1)
        channel_e2.append(float(vv[:2].sum()/vv.sum()))
    return pca, {'pca': pca.to_dict(), 'validation_top2_energy': float(energy[:2].sum()),
                 'validation_top2C_energy': float(energy[:2*train.target.shape[-1]].sum()),
                 'per_channel_top2_energy': channel_e2,
                 'geometry_is_diagnostic_only': True}


def evaluation_indices(ds, count):
    return np.linspace(0, len(ds)-1, min(count, len(ds)), dtype=int)


def target_geometry(bundle):
    train, val = bundle.splits['train'], bundle.splits['val']
    pca = fit_pca(train.target.reshape(len(train), -1))
    vv = pca.project(val.target.reshape(len(val), -1), center=False).var(0, ddof=1)
    c = train.target.shape[-1]
    per_channel = []
    for j in range(c):
        cp = fit_pca(train.target[:,:,j])
        cv = cp.project(val.target[:,:,j], center=False).var(0, ddof=1)
        per_channel.append(float(cv[:2].sum()/cv.sum()))
    return {'pca':pca.to_dict(),'validation_top2_energy':float(vv[:2].sum()/vv.sum()),
            'validation_top2C_energy':float(vv[:2*c].sum()/vv.sum()),
            'per_channel_top2_energy':per_channel,
            'validation_temporal_roughness':float(np.abs(np.diff(val.target,axis=1)).mean()),
            'validation_within_window_temporal_std':float(val.target.std(axis=1).mean()),
            'geometry_is_diagnostic_only':True}


def fit_baselines(bundle, cfg, out):
    train, val, test = [bundle.splits[s] for s in ('train','val','test')]
    indices = evaluation_indices(test, cfg['n_ensemble_contexts'])
    h, target = test.context[indices], test.target[indices]
    f, c = target.shape[1:]
    x = np.column_stack((train.context.reshape(len(train), -1), np.ones(len(train))))
    y = train.target.reshape(len(train), -1)
    vx = np.column_stack((val.context.reshape(len(val), -1), np.ones(len(val))))
    tx = np.column_stack((h.reshape(len(h), -1), np.ones(len(h))))
    penalty = np.eye(x.shape[1]); penalty[-1,-1] = 0
    candidates = []
    gram, rhs = x.T@x, x.T@y
    for alpha in cfg['ridge_alphas']:
        w = np.linalg.solve(gram+alpha*penalty, rhs)
        error = float(np.mean((vx@w-val.target.reshape(len(val), -1))**2))
        candidates.append((error, alpha, w))
    val_error, alpha, w = min(candidates, key=lambda r:r[:2])
    point = {'persistence': np.repeat(h[:,-1:], f, axis=1),
             'ridge': (tx@w).reshape(-1,f,c)}
    period = cfg.get('seasonal_period')
    if period and h.shape[1] >= period:
        point['seasonal_naive'] = h[:,-period:][:,np.arange(f)%period]
    records = {}
    samples = {}
    for name, pred in point.items():
        xx = np.repeat(pred[:,None], cfg['ensemble_size'], axis=1)
        score = forecast_scores(xx, target)
        records[name] = {'means': means(score), 'per_context': {k:v.tolist() for k,v in score.items()},
                         'type': 'deterministic point forecast / degenerate predictive distribution'}
    # Whole-vector residual resampling preserves residual horizon/channel dependence.
    # This is a simple in-sample residual baseline, not calibrated uncertainty.
    residuals = y-x@w
    draw = np.random.default_rng(cfg['evaluation_seed']+15).integers(len(y), size=(len(h),cfg['ensemble_size']))
    xx = point['ridge'][:,None] + residuals[draw].reshape(len(h),cfg['ensemble_size'],f,c)
    score = forecast_scores(xx, target)
    records['ridge_residual_bootstrap'] = {'means': means(score), 'per_context': {k:v.tolist() for k,v in score.items()},
        'type': 'TRAIN in-sample whole-vector residual bootstrap; may understate uncertainty'}
    samples['ridge_residual_bootstrap'] = xx
    save_json(out/'baselines.json', {'selected_ridge_alpha':alpha,'ridge_validation_mse':val_error,
        'ridge_candidates': [{'alpha':a,'validation_mse':e} for e,a,_ in candidates], 'records': records})
    np.savez_compressed(out/'baseline_samples.npz', indices=indices, ridge_weights=w,
                        ridge_residual_draw_indices=draw, **point, **samples)


def train_cell(bundle, cfg, head, source, seed, options, pca, out, fingerprint, device):
    name = f'{source}_{head}_seed{seed}'
    path = out/'runs'/f'{name}.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['fingerprint'] != fingerprint or old['status'] != 'complete':
            raise RuntimeError('Refusing mixed run provenance')
        cp = out/'checkpoints'/f'{name}.pt'
        if not cp.exists() or hashlib.sha256(cp.read_bytes()).hexdigest() != old['checkpoint_sha256']:
            raise RuntimeError('Completed checkpoint missing or changed')
        if not (out/'samples'/f'{name}.npz').exists():
            raise RuntimeError('Completed samples missing')
        print('SKIP', out.name, name, flush=True); return
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = create_extended_model(**options).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model,cfg['learning_rate'],cfg['weight_decay']))
    train, val, test = [bundle.splits[s] for s in ('train','val','test')]
    y, h = tensor(train.target,device), tensor(train.context,device)
    chol = tensor(np.linalg.cholesky(bundle.source_covariances[source]),device)
    vx, vu, vt, _ = pairs(bundle,'val',source,cfg['evaluation_seed'])
    vx, vu, vt, vh = [tensor(a,device) for a in (vx,vu,vt,val.context)]
    # Pair minibatches/source normals/times across heads AFTER architecture initialization.
    torch.manual_seed(seed+30000)
    if device.type == 'mps': torch.mps.manual_seed(seed+30000)
    curve = []; best = float('inf'); best_step = 0
    checkpoint = out/'checkpoints'/f'{name}.pt'; checkpoint.parent.mkdir(parents=True,exist_ok=True)
    print('START',out.name,name,parameter_counts(model),flush=True)
    for step in range(1,cfg['steps']+1):
        model.train()
        ix = torch.randint(len(y),(cfg['batch_size'],),device=device)
        yy, hh = y[ix], h[ix]
        eps = (torch.randn(len(ix),chol.shape[0],device=device)@chol.T).reshape_as(yy)
        t = torch.rand(len(ix),device=device)
        xx = (1-t[:,None,None])*eps+t[:,None,None]*yy
        optimizer.zero_grad(set_to_none=True)
        loss = (model(xx,t,hh)-(yy-eps)).square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']); optimizer.step()
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            model.eval()
            with torch.no_grad(): validation = float((predict(model,vx,vt,vh)-vu).square().mean().cpu())
            if not np.isfinite(validation): raise FloatingPointError(name)
            curve.append({'step':step,'train_loss':float(loss.detach().cpu()),'validation_mse':validation,
                          'seconds':time.perf_counter()-started})
            if validation < best:
                best, best_step = validation, step
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'options':options,'step':step,'seed':seed,'fingerprint':fingerprint},checkpoint)
            save_json(out/'training'/f'{name}.json',curve)
            print(out.name,name,step,'val',round(validation,6),flush=True)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['state_dict']);model.eval()
    tx, tu, tt, _ = pairs(bundle,'test',source,cfg['evaluation_seed']+100)
    with torch.no_grad():
        prediction = predict(model,tensor(tx,device),tensor(tt,device),tensor(test.context,device)).cpu().numpy()
    vm = velocity_metrics(prediction.reshape(len(test),-1),tu.reshape(len(test),-1),pca)
    ix = evaluation_indices(test,cfg['n_ensemble_contexts'])
    n, s, f, c = len(ix),cfg['ensemble_size'],test.target.shape[1],test.target.shape[2]
    eps = sample_source(bundle,n*s,source,cfg['evaluation_seed']+200)
    hh = np.repeat(test.context[ix],s,axis=0)
    gen = generate(model,tensor(eps,device),tensor(hh,device),cfg['euler_steps'],cfg['generation_batch_size']).reshape(n,s,f,c)
    scores = forecast_scores(gen,test.target[ix])
    # Separate fixed first 8 selected contexts, first training seed, doubled solver steps.
    solver = None
    if seed == cfg['seeds'][0]:
        ns = min(8,n)
        refined = generate(model,tensor(eps[:ns*s],device),tensor(hh[:ns*s],device),
                           2*cfg['euler_steps'],cfg['generation_batch_size']).reshape(ns,s,f,c)
        refined_scores = forecast_scores(refined,test.target[ix[:ns]])
        solver = {'contexts':ns,'euler_steps':2*cfg['euler_steps'],
                  'endpoint_rmse':float(np.sqrt(np.mean((refined.astype(float)-gen[:ns])**2))),
                  'base':means({k:v[:ns] for k,v in scores.items()}), 'refined':means(refined_scores)}
        (out/'solver').mkdir(exist_ok=True)
        np.savez_compressed(out/'solver'/f'{name}.npz',generated=refined)
    (out/'samples').mkdir(exist_ok=True)
    np.savez_compressed(out/'samples'/f'{name}.npz',generated=gen,prediction=prediction,indices=ix)
    record = {'name':name,'status':'complete','fingerprint':fingerprint,'head':head,'source':source,'seed':seed,
              'device':str(device),'model_options':options,'parameters':parameter_counts(model),
              'steps':cfg['steps'],'best_step':best_step,'best_validation_mse':best,'training':curve,
              'velocity_metrics':vm,'forecast_means':means(scores),
              'per_context':{k:v.tolist() for k,v in scores.items()},'solver_sensitivity':solver,
              'seconds':time.perf_counter()-started,
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    save_json(path,record)
    print('DONE',out.name,name,'CRPS',round(record['forecast_means']['fair_crps'],6),
          'seconds',round(record['seconds'],1),flush=True)
    del model,optimizer
    if device.type == 'mps': torch.mps.empty_cache()


def run(cfg, out, device, dataset_filter=None):
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no fallback')
    torch.set_num_threads(4)
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    manifest = {'config':cfg,'source_sha256':source_hashes(), 'dataset_sha256': {
        dc['path']: hashlib.sha256(Path(dc['path']).read_bytes()).hexdigest()
        for dc in cfg['datasets'].values() if dc['kind']=='real'}}
    fingerprint = config_hash(manifest); manifest['fingerprint'] = fingerprint
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != manifest:
        raise RuntimeError('Source/configuration changed: use a fresh output directory')
    save_json(out/'manifest.json',manifest)
    save_json(out/'environment.json',{'device':str(device),'torch':torch.__version__,'numpy':np.__version__,
                                     'platform':platform.platform()})
    for name, data_cfg in cfg['datasets'].items():
        if dataset_filter and name != dataset_filter: continue
        dest = out/name; dest.mkdir(exist_ok=True)
        bundle = build_data(data_cfg)
        save_json(dest/'data_metadata.json',bundle.metadata)
        cell_cfg = dict(cfg); cell_cfg['seasonal_period'] = data_cfg.get('seasonal_period')
        capacities = select_widths(bundle.splits['train'],cfg)
        save_json(dest/'capacities.json',capacities)
        pcas, geo = {}, {}
        for source in cfg['sources']:
            pcas[source], geo[source] = geometry(bundle,source,cfg['evaluation_seed']+300)
        geo['target'] = target_geometry(bundle)
        save_json(dest/'geometry.json',geo)
        fit_baselines(bundle,cell_cfg,dest)
        for seed in cfg['seeds']:
            for source in cfg['sources']:
                for head in cfg['heads']:
                    options = model_options(bundle.splits['train'],head,capacities[head]['width'],cfg)
                    train_cell(bundle,cell_cfg,head,source,seed,options,pcas[source],dest,fingerprint,device)
    print('SUITE FINISHED',out,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/extension.json');p.add_argument('--output',default='results/extension')
    p.add_argument('--device',choices=('cpu','mps'),default='mps');p.add_argument('--dataset')
    a = p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,torch.device(a.device),a.dataset)


if __name__ == '__main__': main()
