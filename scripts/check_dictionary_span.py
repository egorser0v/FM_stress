#!/usr/bin/env python3
"""Diagnose the fixed velocity span without modifying models or experiments.

For v(x,t,h)=D a(x,t,h), projection onto span(D)'s orthogonal complement is
invariant along every ODE trajectory. TopK changes coefficients but not this
global span. A dictionary with 64 atoms can therefore constrain F*C>64 tasks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from fm_stress.extended_data import build_data
from fm_stress.metrics import PCA
from fm_stress.sparse_experiment import make_model
from check_extension import rebuild_pairs,draw_source,close


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_one(path,bundle,cfg,fingerprint):
    record=read(path);dest=path.parent.parent;name=record['name']
    cp=dest/'checkpoints'/f'{name}.pt';sp=dest/'samples'/f'{name}.npz'
    if record['fingerprint']!=fingerprint or record['status']!='complete' or sha(cp)!=record['checkpoint_sha256'] or sha(sp)!=record['samples_sha256']:
        raise AssertionError('Changed/incomplete dictionary run '+name)
    checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    p=checkpoint['pca'];pca=PCA(np.asarray(p['mean']),np.asarray(p['components']),np.asarray(p['eigenvalues']),p['floor'])
    model=make_model(checkpoint['options'],pca);model.load_state_dict(checkpoint['state_dict']);model.eval()
    if model.transform_input:raise AssertionError('Dictionary diagnostic assumes observation-coordinate head')
    # Actual float32 normalization in the field; SVD/projections use float64.
    dictionary=F.normalize(model.base.dictionary.detach(),dim=0).double().numpy()
    u,s,_=np.linalg.svd(dictionary,full_matrices=True)
    tolerance=max(dictionary.shape)*np.finfo(np.float32).eps*s[0]
    rank=int((s>tolerance).sum());d=dictionary.shape[0];null_basis=u[:,rank:]
    archive=np.load(sp);generated=archive['generated'].astype(float);indices=archive['indices']
    n,ensemble,horizon,channels=generated.shape
    test=bundle.splits['test']
    expected_indices=np.linspace(0,len(test)-1,min(len(test),cfg['n_ensemble_contexts']),dtype=int)
    np.testing.assert_array_equal(indices,expected_indices)
    if generated.shape!=(len(expected_indices),cfg['ensemble_size'],*test.target.shape[1:]):
        raise AssertionError('Dictionary forecast shape differs from configured data')
    eps=draw_source(bundle.source_covariances[record['source']],n*ensemble,cfg['evaluation_seed']+200).astype(np.float32).astype(float)
    endpoints=generated.reshape(n*ensemble,d)
    endpoint_null=(endpoints-eps)@null_basis
    invariant_max=float(np.max(np.abs(endpoint_null))) if endpoint_null.size else 0.
    invariant_rmse=float(np.sqrt(np.mean(endpoint_null**2))) if endpoint_null.size else 0.
    # Float32 Euler accumulation may drift by tiny rounding errors, never an
    # appreciable independent update in these theoretically frozen directions.
    if invariant_max>5e-4:
        raise AssertionError(f'{name}: orthogonal endpoint invariance failed ({invariant_max})')
    target=bundle.splits['test'].target[indices].reshape(n,d)
    mean=endpoints.reshape(n,ensemble,d).mean(1)
    target_null_energy=float(np.square(target@null_basis).sum()/n/d)
    observed_mean_null_error=float(np.square((mean-target)@null_basis).sum()/n/d)
    observed_total_mean_error=float(np.square(mean-target).mean())
    if observed_mean_null_error>observed_total_mean_error+1e-10:
        raise AssertionError('Orthogonal error larger than full error')
    source_null_variance=float(np.trace(null_basis.T@bundle.source_covariances[record['source']]@null_basis)/d)
    tx,tu,tt=rebuild_pairs(bundle,record['source'],cfg['evaluation_seed']+100)
    velocity_floor=float(np.square(tu.reshape(len(tu),d).astype(float)@null_basis).sum()/len(tu)/d)
    prediction=archive['prediction'].reshape(len(tu),d).astype(float)
    predicted_null=prediction@null_basis
    prediction_null_max=float(np.max(np.abs(predicted_null))) if predicted_null.size else 0.
    if prediction_null_max>1e-4:raise AssertionError('Velocity leaves dictionary span')
    velocity_mse=float(record['velocity_metrics']['velocity_mse'])
    if velocity_floor>velocity_mse+2e-6:raise AssertionError('Velocity lower bound exceeds observed loss')
    val=bundle.splits['val'];vx,_,vt=rebuild_pairs(bundle,record['source'],cfg['evaluation_seed'],'val')
    codes=[]
    with torch.no_grad():
        for start in range(0,len(vx),256):
            codes.append(model.base.coefficients(torch.as_tensor(vx[start:start+256]),torch.as_tensor(vt[start:start+256]),
                torch.as_tensor(val.context[start:start+256],dtype=torch.float32)).numpy())
    codes=np.concatenate(codes);usage=np.mean(codes!=0,axis=0);active=usage>0
    visited_rank=int(np.linalg.matrix_rank(dictionary[:,active],tol=tolerance)) if active.any() else 0
    return {'dataset':dest.name,'name':name,'head':record['head'],'source':record['source'],'seed':record['seed'],
        'checkpoint_sha256':record['checkpoint_sha256'],'samples_sha256':record['samples_sha256'],
        'joint_dimension':d,'dictionary_atoms':dictionary.shape[1],'dictionary_rank':rank,'orthogonal_dimension':d-rank,
        'rank_tolerance':float(tolerance),'dictionary_singular_values':s.tolist(),
        'endpoint_orthogonal_invariance_max_abs':invariant_max,'endpoint_orthogonal_invariance_rmse':invariant_rmse,
        'predicted_velocity_orthogonal_max_abs':prediction_null_max,
        'test_velocity_mse':velocity_mse,'test_velocity_orthogonal_mse_floor':velocity_floor,
        'floor_share_of_test_velocity_mse':velocity_floor/max(velocity_mse,1e-30),
        'selected_target_orthogonal_energy_per_coordinate':target_null_energy,
        'source_orthogonal_variance_per_coordinate':source_null_variance,
        'expected_finite_ensemble_mean_orthogonal_mse':target_null_energy+source_null_variance/ensemble,
        'observed_ensemble_mean_orthogonal_mse':observed_mean_null_error,
        'observed_ensemble_mean_total_mse':observed_total_mean_error,
        'validation_atom_usage_fraction':usage.tolist(),'validation_unused_atoms':int((usage==0).sum()),
        'validation_used_atom_union':int(active.sum()),'validation_visited_dictionary_rank':visited_rank,
        'validation_mean_active_atoms_per_example':float((codes!=0).sum(1).mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/sparse-extension')
    parser.add_argument('--output',default='results/diagnostics/dictionary_span.json')
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();root=Path(args.results);manifest=read(root/'manifest.json');cfg=manifest['config']
    unsigned={k:v for k,v in manifest.items() if k!='fingerprint'}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest()!=manifest['fingerprint']:
        raise AssertionError('Invalid manifest fingerprint')
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/'fm_stress'/name)!=digest:raise AssertionError('Frozen source changed '+name)
    for name,digest in manifest['dataset_sha256'].items():
        if sha(name)!=digest:raise AssertionError('Frozen dataset changed '+name)
    torch.set_num_threads(2)
    expected={(d,h,s,k) for d in cfg['datasets'] for h in ('dictionary_dense','dictionary_topk') for s in cfg['sources'] for k in cfg['seeds']}
    records=[];observed=set()
    for dataset,dc in cfg['datasets'].items():
        paths=sorted((root/dataset/'runs').glob('*dictionary*.json'))
        if not paths:continue
        bundle=build_data(dc)
        if bundle.metadata!=read(root/dataset/'data_metadata.json'):
            raise AssertionError('Dictionary diagnostic data metadata differs '+dataset)
        for path in paths:
            item=audit_one(path,bundle,cfg,manifest['fingerprint'])
            identity=(dataset,item['head'],item['source'],item['seed'])
            if identity not in expected or identity in observed:raise AssertionError('Unexpected dictionary run')
            observed.add(identity);records.append(item)
        print('VERIFIED dictionary span',dataset,len(paths),'runs',flush=True)
    missing=sorted(expected-observed)
    if missing and not args.allow_partial:raise AssertionError(f'Missing {len(missing)} dictionary runs')
    report={'complete':not missing,'fingerprint':manifest['fingerprint'],'expected_runs':len(expected),'verified_runs':len(records),
        'missing':missing,'runs':records,
        'interpretation':['Every velocity lies in the fixed learned dictionary span, so its orthogonal endpoint projection equals initial source projection.',
            '64 atoms restrict joint dimensions above64; sparse and dense controls share this restriction. This is architectural capacity, not a numerical solver bug.',
            'TopK8 is instantaneous sparsity; trajectories can use a union of up to64 atoms. Matrix rank and observed validation usage are distinct diagnostics.',
            'The velocity orthogonal floor is an exact lower bound for these fixed test velocity labels; the mean-forecast expectation is over new independent source ensembles, not a deterministic bound for a stored random ensemble.',
            'Observed mean-forecast null error is a component of full mean MSE. No CRPS or energy lower bound is claimed.',
            'Validation unused atoms are unused on the recorded validation interpolants only, not proven dead everywhere.'],
        'diagnostic_source_sha256':sha(__file__)}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE' if not missing else 'PARTIAL',len(records),'/',len(expected),output,flush=True)


if __name__=='__main__':main()
