#!/usr/bin/env python3
"""Independent CPU replay of Vasilii's frozen kernel-scan checkpoints.

No training and no modification of original results. Data/scores use the earlier
independent audit definitions, not production data/metric functions. Checkpoint
architecture reuse is unavoidable. Full 8,192-pair velocity replay for all 360
fits; full original 256-context x 32-member Euler64 ensembles and 1,024 paths
for RBF F64 white/GP MLP/S4 and OU F64 GP MLP/S4 (18 fits).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
import check_kernel_scan as independent
from fm_stress.models import create_model
from fm_stress.experiment import CoordinateModel

OUT = ROOT / 'results/followup-reproduction'
SRC = ROOT / 'results/kernel-scan'
RTOL, ATOL = 1e-4, 1e-8

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def compare(got, want):
    a, b = np.asarray(got, float), np.asarray(want, float)
    return {'passed': bool(np.allclose(a, b, rtol=RTOL, atol=ATOL)),
            'max_abs': float(np.max(np.abs(a-b))),
            'max_relative': float(np.max(np.abs(a-b)/(np.abs(b)+1e-15)))}

def velocity(pred, truth, pca):
    error = np.asarray(pred, float)-truth
    pc = np.mean((error @ np.asarray(pca['components']).T)**2, axis=0)
    npc = pc / np.maximum(pca['eigenvalues'], pca['floor'])
    return {'velocity_mse': float(np.mean(error**2)), 'pc1_mse': float(pc[0]),
            'pc2_mse': float(pc[1]), 'residual_mse': float(pc[2:].mean()),
            'pc_mse': pc.tolist(), 'pc1_normalized_mse': float(npc[0]),
            'pc2_normalized_mse': float(npc[1]),
            'residual_normalized_mse': float(npc[2:].mean()),
            'pc_normalized_mse': npc.tolist(), 'variance_floor': pca['floor']}

@torch.no_grad()
def predict(model, x, t, h, batch=256):
    return np.concatenate([model(torch.as_tensor(x[i:i+batch], dtype=torch.float32),
                                 torch.as_tensor(t[i:i+batch], dtype=torch.float32),
                                 torch.as_tensor(h[i:i+batch], dtype=torch.float32)).numpy()
                           for i in range(0, len(x), batch)])

@torch.no_grad()
def euler(model, initial, history, steps, batch=256):
    out=[]
    for i in range(0,len(initial),batch):
        x=torch.as_tensor(initial[i:i+batch],dtype=torch.float32).clone()
        h=torch.as_tensor(history[i:i+batch],dtype=torch.float32)
        for k in range(steps):
            x=x+model(x,torch.full((len(x),),k/steps),h)/steps
        out.append(x.numpy())
    return np.concatenate(out)

def focus(rec):
    return rec['geometry_id']=='F64_l8' and rec['head'] in ('mlp','s4') and (
        rec['kernel']=='rbf' or (rec['kernel']=='ou' and rec['source']=='gp'))

def model_from(rec, checkpoint, cfg):
    model=CoordinateModel(create_model('s4' if rec['head']=='s4' else 'mlp',
        horizon=rec['geometry']['horizon'],lookback=rec['geometry']['lookback'],
        width=rec['width'],**cfg['model']))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict']);model.eval()
    return model, state

def replay(path, cfg, do_generation, cache):
    rec=read(path);group=path.parent.parent;geo=read(group/'geometry.json')
    key=str(path.relative_to(SRC));dest=OUT/('generation_runs' if do_generation else 'velocity_runs')/key.replace('/','__')
    checkpoint=group/'checkpoints'/(path.stem+'.pt')
    provenance={'record_sha256':sha(path),'checkpoint_sha256':sha(checkpoint),
                'audit_script_sha256':sha(__file__),'independent_audit_sha256':sha(independent.__file__),
                'config_sha256':sha(SRC/'config.json'),'generation_requested':do_generation}
    if dest.exists():
        old=read(dest)
        if old['provenance']==provenance:return old
        raise ValueError('Changed audit inputs; use a fresh output directory: '+str(dest))
    started=time.perf_counter()
    g=rec['geometry'];spec={**g,'threshold':g['normalization_threshold']}
    cachekey=(rec['kernel'],rec['geometry_id'],rec['data_seed'])
    if cache.get('key')!=cachekey:
        cache.clear();cache.update(key=cachekey,test=independent.windows(spec,cfg['n_test'],rec['data_seed']+2))
    test=cache['test'];x,u,t,eps=independent.pairs(spec,test.target,rec['source'],rec['data_seed'],'test')
    model,state=model_from(rec,checkpoint,cfg)
    pred=predict(model,x,t,test.context)
    metrics=velocity(pred,u,geo['sources'][rec['source']]['pca'])
    fixed=velocity(pred,u,geo['target_pca'])
    checks={k:compare(v,rec['metrics'][k]) for k,v in metrics.items()}
    checks.update({'target_basis_'+k:compare(v,rec['fixed_target_basis_metrics'][k]) for k,v in fixed.items()})
    checks['checkpoint_hash']={'passed':provenance['checkpoint_sha256']==rec['checkpoint_sha256']}
    checks['checkpoint_identity']={'passed':all(state[k]==rec[k] for k in
        ('kernel','geometry_id','data_seed','seed','source','head','fingerprint'))}
    checks['checkpoint_step']={'passed':state['step']==rec['best_step']}
    group_values=independent.group_means(np.asarray(metrics['pc_mse']),rec['pc_groups'])
    checks.update({'group_'+k:compare(v,rec['group_mse'][k]) for k,v in group_values.items() if v is not None})
    result={'identity':{k:rec[k] for k in ('kernel','geometry_id','data_seed','seed','source','head')},
            'n_velocity_pairs':len(x),'provenance':provenance,'checks':checks,
            'metrics':metrics,'fixed_target_basis_metrics':fixed,'group_mse':group_values,
            'generation':None,'velocity_seconds':time.perf_counter()-started}
    if do_generation:
        nc,ns=cfg['n_ensemble_contexts'],cfg['ensemble_size']
        e=independent.source_draw(spec,nc*ns,rec['source'],rec['data_seed']+1000)
        ensemble=euler(model,e,np.repeat(test.context[:nc],ns,0),cfg['euler_steps']).reshape(nc,ns,-1)
        # Independent O(S^2) CRPS and energy-score formula; deliberately not production code.
        scores=independent.forecast_scores(ensemble,test.target[:nc])
        generated=euler(model,eps[:cfg['n_paths']],test.context[:cfg['n_paths']],cfg['euler_steps'])
        genrough=independent.roughness(generated);targetrough=independent.roughness(test.target[:cfg['n_paths']])
        checks.update({'forecast_'+k:compare(v,rec['per_context'][k]) for k,v in scores.items()})
        checks.update({'forecast_mean_'+k:compare(float(v.mean()),rec['forecast_means'][k]) for k,v in scores.items()})
        checks['generated_roughness']=compare(genrough,rec['metrics']['generated_roughness'])
        checks['target_roughness']=compare(targetrough,rec['metrics']['target_roughness'])
        npz=OUT/'samples'/key.replace('/','__').replace('.json','.npz');npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(npz,prediction=pred,generated=generated,ensemble=ensemble,
                            path_target=test.target[:cfg['n_paths']],ensemble_target=test.target[:nc])
        result['generation']={'contexts':nc,'members':ns,'paths':cfg['n_paths'],
            'euler_steps':cfg['euler_steps'],'forecast_means':{k:float(v.mean()) for k,v in scores.items()},
            'per_context':{k:v.tolist() for k,v in scores.items()},'generated_roughness':genrough,
            'samples':str(npz.relative_to(ROOT)),'samples_sha256':sha(npz)}
    result['seconds']=time.perf_counter()-started
    result['passed']=all(c['passed'] for c in checks.values());save(dest,result)
    print(key,'PASS' if result['passed'] else 'FAIL',round(result['seconds'],2),'s',flush=True)
    return result

def integrity():
    out={}
    for study in ('kernel-scan','etth1-normalization'):
        folder=ROOT/'results'/study;m=read(folder/'manifest.json');bad=[]
        for name,h in m['source_hashes'].items():
            if sha(ROOT/name)!=h:bad.append(name)
        counts={k:0 for k in ('records','checkpoints_ok','checkpoints_missing','checkpoints_wrong','samples_ok','samples_missing','samples_wrong')}
        for p in folder.rglob('runs/*.json'):
            rec=read(p);counts['records']+=1
            for typ,ext,field in [('checkpoints','.pt','checkpoint_sha256'),('samples','.npz','samples_sha256')]:
                f=p.parent.parent/typ/(p.stem+ext)
                counts[typ+('_missing' if not f.exists() else '_ok' if sha(f)==rec[field] else '_wrong')]+=1
        out[study]={'counts':counts,'changed_frozen_sources':bad}
    out['missing_referenced_protocols']=[n for n in ('docs/etth1_level_hypothesis.tex','docs/etth1_prior_level_protocol.md') if not (ROOT/n).exists()]
    save(OUT/'integrity.json',out)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['benchmark','velocity','generation','all'],default='all')
    a=p.parse_args();torch.set_num_threads(2)
    cfg=read(SRC/'config.json');integrity()
    paths=sorted(SRC.rglob('runs/*.json'))
    if a.phase=='benchmark':paths=[SRC/'rbf/F64_l8/draw307041/runs'/f'white_{h}.json' for h in ('mlp','s4')]
    elif a.phase=='generation':paths=[p for p in paths if focus(read(p))]
    records=[];cache={}
    for path in paths:
        gen=a.phase in ('all','generation') and focus(read(path))
        records.append(replay(path,cfg,gen,cache))
    save(OUT/f'{a.phase}_summary.json',{'phase':a.phase,'expected_runs':len(paths),'verified_runs':len(records),
        'complete':len(records)==len(paths),'passed':all(r['passed'] for r in records),
        'n_generation_runs':sum(r['generation'] is not None for r in records),
        'cpu_threads':2,'torch':torch.__version__,'rtol':RTOL,'atol':ATOL,
        'summed_seconds':sum(r['seconds'] for r in records),
        'failed_checks':[{'identity':r['identity'],'checks':{k:v for k,v in r['checks'].items() if not v['passed']}}
                         for r in records if not r['passed']]})

if __name__=='__main__':main()
