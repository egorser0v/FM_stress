#!/usr/bin/env python3
"""Post-hoc train/validation geometry audit; never a tuning/selection gate.

The original optional real-data task requires near-collinearity AFTER the same
history-only normalization. This reports that criterion for the existing ETTh1
factorial without retroactively treating the factorial as a qualifying Task 3 run.
"""
import json
import numpy as np
from check_followup_normalization import ROOT,read,rebuild,transform,draws,sha,save

def top2(train,val):
    values,vectors=np.linalg.eigh(np.cov(train,rowvar=False))
    Q=vectors[:,np.argsort(values)[::-1]].T
    ev=np.var(val@Q.T,axis=0,ddof=1)
    return float(ev[:2].sum()/ev.sum())

def main():
    cfg=read(ROOT/'configs/followup_normalization.json');data,cov,_=rebuild(cfg);rows={}
    seeds=[681041,681042,681043,681044,681045]
    for mode in cfg['normalizations']:
        _,ty,*_=transform(data['train'],mode);_,vy,*_=transform(data['val'],mode)
        ty,vy=ty[...,0],vy[...,0]
        r={'target_top2_energy':top2(ty,vy),
           'target_mean_within_patch_std':float(np.std(vy,axis=1).mean()),
           'target_roughness':float(np.abs(np.diff(vy,axis=1)).mean()),'sources':{}}
        for source in ['white','gp']:
            energy=[top2(ty-draws(cov,source,len(ty),s)[...,0],vy-draws(cov,source,len(vy),s+100)[...,0]) for s in seeds]
            r['sources'][source]={'validation_top2_energy':energy,'min':min(energy),'max':max(energy),'mean':float(np.mean(energy))}
        r['gp_energy_gate_passes']=[v>.9 for v in r['sources']['gp']['validation_top2_energy']]
        rows[mode]=r
    result={'status':'post-hoc descriptive geometry audit; no test data or trained scores used; no run selection',
            'data':'ETTh1 OT; original chronological train/validation splits',
            'train_source_seeds':seeds,'validation_source_seed_offset':100,
            'threshold':.9,'normalizations':rows,
            'original_optional_real_task_qualifies':all(rows['history']['gp_energy_gate_passes']),
            'interpretation':'The original real-data option requires the history-normalized geometry. Other modes may pass different numerical gates but are separate pipeline interventions. A failure does not invalidate this exploratory study; it prevents labeling it as the original qualifying real-data column.',
            'config_sha256':sha(ROOT/'configs/followup_normalization.json'),
            'dataset_sha256':sha(ROOT/cfg['dataset']['path']),'script_sha256':sha(__file__)}
    save(ROOT/'results/followup-reproduction/scope_geometry.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
