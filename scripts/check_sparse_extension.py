#!/usr/bin/env python3
"""Independent CPU audit of sparse/whitening fits and stored forecasting arrays.

Reuses independent NumPy/SciPy equations from check_extension, never production
forecast or velocity metric functions. Replays every validation/test velocity;
does not claim to independently reproduce stochastic optimization trajectories.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from fm_stress.extended_data import build_data
from fm_stress.extended_models import parameter_counts
from fm_stress.metrics import PCA
from fm_stress.sparse_experiment import make_model
from check_extension import (load, sha, stable_hash, close, independent_scores,
    mean_scores, compare_scores, rebuild_pairs, validate_data, validate_geometry,
    validate_target_geometry, independent_velocity)


def replay(model, x, t, h):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            values = [torch.as_tensor(a[start:start+256], dtype=torch.float32) for a in (x,t,h)]
            outputs.append(model(*values).numpy())
    return np.concatenate(outputs)


def pca_object(data):
    return PCA(np.asarray(data['mean']), np.asarray(data['components']),
               np.asarray(data['eigenvalues']), data['floor'])


def audit_baseline(bundle, cfg, dest):
    record = load(dest/'sparse_baselines.json')
    archive = np.load(dest/'sparse_baseline_samples.npz')
    provenance=record['provenance']
    if record['status']!='complete' or stable_hash(provenance)!=record['fingerprint']:
        raise AssertionError('Elastic Net provenance fingerprint')
    if sha(dest/'sparse_baseline_samples.npz')!=record['samples_sha256']:
        raise AssertionError('Elastic Net sample hash')
    for name,digest in provenance['source_sha256'].items():
        if sha(ROOT/'fm_stress'/name)!=digest:raise AssertionError('Elastic Net source changed '+name)
    if stable_hash(bundle.metadata)!=provenance['metadata_sha256']:
        raise AssertionError('Elastic Net data metadata provenance')
    for key in ('n_ensemble_contexts','ensemble_size','evaluation_seed'):
        if provenance['config'][key]!=cfg[key]:raise AssertionError('Elastic Net config '+key)
    train, val, test = [bundle.splits[k] for k in ('train','val','test')]
    ix = np.linspace(0, len(test)-1, min(len(test),cfg['n_ensemble_contexts']), dtype=int)
    close(archive['indices'], ix, 'Elastic Net selected contexts')
    x, y = train.context.reshape(len(train),-1), train.target.reshape(len(train),-1)
    w, b = archive['coefficients'], archive['intercept']
    mask = np.any(w != 0, axis=0)
    close(mask, archive['active_input_mask'], 'Elastic Net sparse feature mask')
    if int(mask.sum()) != record['active_input_features']:
        raise AssertionError('Elastic Net active feature count')
    if hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() != record['training_context_sha256']:
        raise AssertionError('Elastic Net fit history data differs')
    if hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest() != record['training_target_sha256']:
        raise AssertionError('Elastic Net fit target data differs')
    candidates = record['candidates']
    grid = {(a,r) for a in cfg['elastic_alphas'] for r in cfg['elastic_l1_ratios']}
    if {(v['alpha'],v['l1_ratio']) for v in candidates} != grid:
        raise AssertionError('Elastic Net candidate grid differs from configuration')
    selected = min((i for i,v in enumerate(candidates) if v['converged']),
                   key=lambda i:(candidates[i]['validation_mse'],candidates[i]['alpha'],candidates[i]['l1_ratio']))
    if selected != record['selected_candidate_index']:
        raise AssertionError('Elastic Net validation selection')
    close(record['selected_alpha'],candidates[selected]['alpha'],'Elastic selected alpha')
    close(record['selected_l1_ratio'],candidates[selected]['l1_ratio'],'Elastic selected ratio')
    val_prediction = val.context.reshape(len(val),-1)@w.T+b
    close(np.mean((val_prediction-val.target.reshape(len(val),-1))**2),record['validation_mse'],'Elastic validation replay')
    residual = y-(x@w.T+b)
    close(residual,archive['training_residuals'],'Elastic training residuals')
    # Convex optimality certificate in original (uncompressed) training space.
    # Reconstruct primal/dual independently from saved coefficients, not a fitted
    # sklearn object and not the QR module's certificate helper.
    lam1=record['selected_alpha']*record['selected_l1_ratio']
    lam2=record['selected_alpha']*(1-record['selected_l1_ratio'])
    xc,yc=x-x.mean(0),y-y.mean(0)
    coef=w.T;norms=np.sqrt(np.square(coef).sum(axis=1));n=len(x)
    primal=float(np.square(residual).sum()/(2*n)+lam1*norms.sum()+lam2*np.square(coef).sum()/2)
    theta=residual/n
    dual_norms=np.linalg.norm(xc.T@theta,axis=1)
    if lam2:
        conjugate=float(np.square(np.maximum(dual_norms-lam1,0)).sum()/(2*lam2))
    else:
        theta=theta*min(1.,lam1/max(float(dual_norms.max()),1e-30));conjugate=0.
    dual=float((yc*theta).sum()-n*np.square(theta).sum()/2-conjugate)
    tolerance=float(record['tol']*np.square(yc).sum()/n)
    certificate=record['selected_original_certificate']
    for key,value in {'primal_objective':primal,'dual_objective':dual,'duality_gap':primal-dual,
                      'dual_gap_tolerance':tolerance}.items():
        close(value,certificate[key],'Elastic original-space '+key,atol=1e-8)
    if not -1e-9<=primal-dual<=1.1*tolerance+1e-10 or not certificate['dual_certificate_passed']:
        raise AssertionError('Elastic Net original-space convex certificate failed')
    gradient=-xc.T@residual/n+lam2*coef
    group_kkt=np.maximum(np.linalg.norm(gradient,axis=1)-lam1,0)
    active=norms>0
    group_kkt[active]=np.linalg.norm(gradient[active]+lam1*coef[active]/norms[active,None],axis=1)
    close(group_kkt.max(),certificate['maximum_feature_kkt_residual'],'Elastic original-space KKT',atol=1e-8)
    point = (test.context[ix].reshape(len(ix),-1)@w.T+b).reshape(test.target[ix].shape)
    close(point,archive['multitask_elastic_net'],'Elastic point prediction')
    draws = np.random.default_rng(cfg['evaluation_seed']+15).integers(len(train),size=(len(ix),cfg['ensemble_size']))
    close(draws,archive['residual_draw_indices'],'Elastic whole-vector residual indices')
    generated = point[:,None]+residual[draws].reshape(len(ix),cfg['ensemble_size'],*point.shape[1:])
    close(generated,archive['multitask_elastic_net_residual_bootstrap'],'Elastic bootstrap reconstruction')
    compare_scores(independent_scores(generated,test.target[ix]),record['records']['multitask_elastic_net_residual_bootstrap'],
                   'Elastic probabilistic scores',mean_key='means')
    compare_scores(independent_scores(np.repeat(point[:,None],cfg['ensemble_size'],axis=1),test.target[ix]),
                   record['records']['multitask_elastic_net'],'Elastic deterministic scores',mean_key='means')


def audit_frozen_controls(bundle,cfg,reference,dataset):
    """Reference JSON hashes alone do not protect historical NPZ sample arrays."""
    test=bundle.splits['test']
    ix=np.linspace(0,len(test)-1,min(len(test),cfg['n_ensemble_contexts']),dtype=int)
    expected={(s,h,k) for s in cfg['sources'] for h in ('mlp','sparse_mlp','s4','unet') for k in cfg['seeds']}
    observed=set()
    for path in sorted((reference/dataset/'runs').glob('*.json')):
        record=load(path);identity=(record['source'],record['head'],record['seed'])
        if identity not in expected or identity in observed or record['status']!='complete':
            raise AssertionError('Frozen control identity/status differs '+str(path))
        observed.add(identity)
        cp=reference/dataset/'checkpoints'/f"{record['name']}.pt"
        if sha(cp)!=record['checkpoint_sha256']:
            raise AssertionError('Frozen control checkpoint changed '+str(cp))
        archive=np.load(reference/dataset/'samples'/f"{record['name']}.npz")
        close(archive['indices'],ix,'Frozen control evaluation contexts')
        compare_scores(independent_scores(archive['generated'],test.target[ix]),record,'Frozen '+record['name'])
    if observed!=expected:raise AssertionError('Missing frozen controls '+dataset)
    return len(observed)


def audit_run(path,bundle,cfg,pca,fingerprint,capacities):
    record = load(path); name=record['name']; dest=path.parent.parent
    if record['fingerprint'] != fingerprint or record['status'] != 'complete' or record['device'] != 'mps':
        raise AssertionError(name+' provenance/status/device')
    head, source, seed = record['head'],record['source'],record['seed']
    if name != f'{source}_{head}_seed{seed}' or record['steps'] != cfg['steps']:
        raise AssertionError(name+' identity/budget')
    if record['coordinate_input'] != (head=='mlp_whitened') or record['variance_weighted_objective'] != (head in ('mlp_whitened','mlp_balanced')):
        raise AssertionError(name+' coordinate/objective flags')
    if record['pca_floored_components'] != int((np.asarray(pca['eigenvalues']) < pca['floor']).sum()):
        raise AssertionError(name+' floored PCA count')
    curve=record['training']
    expected=list(range(cfg['eval_every'],cfg['steps']+1,cfg['eval_every']))
    if not expected or expected[-1] != cfg['steps']:expected.append(cfg['steps'])
    if [r['step'] for r in curve] != expected or curve != load(dest/'training'/f'{name}.json'):
        raise AssertionError(name+' validation log steps/content')
    winner=min(curve,key=lambda row:row['validation_objective'])
    if record['best_step'] != winner['step']:
        raise AssertionError(name+' validation-selected step')
    close(record['best_validation_objective'],winner['validation_objective'],name+' selected objective')
    for row in curve:
        close(row['train_loss'],row['train_cfm']+row['train_aux'],name+' training auxiliary composition',atol=2e-5,rtol=2e-6)
    cp=dest/'checkpoints'/f'{name}.pt'; sp=dest/'samples'/f'{name}.npz'
    if sha(cp)!=record['checkpoint_sha256'] or sha(sp)!=record['samples_sha256']:
        raise AssertionError(name+' saved artifact hash')
    checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    if checkpoint['options']!=record['model_options'] or checkpoint['step']!=record['best_step'] or checkpoint['fingerprint']!=fingerprint or checkpoint['seed']!=seed:
        raise AssertionError(name+' checkpoint metadata')
    if checkpoint['pca'] != pca:
        raise AssertionError(name+' checkpoint PCA differs from train geometry')
    model=make_model(checkpoint['options'],pca_object(pca))
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    if parameter_counts(model)!=record['parameters'] or record['model_options']['width']!=capacities[head]['width']:
        raise AssertionError(name+' parameter/width accounting')
    close(model.q.detach().numpy(),np.asarray(pca['components']).T,name+' checkpoint q',atol=1e-7)
    close(model.scales.detach().numpy(),np.sqrt(np.maximum(pca['eigenvalues'],pca['floor'])),name+' checkpoint scales',atol=1e-6)
    val=bundle.splits['val']; vx,vu,vt=rebuild_pairs(bundle,source,cfg['evaluation_seed'],'val')
    vp=replay(model,vx,vt,val.context)
    error=(vp-vu).reshape(len(vp),-1)
    raw=float(np.mean(error**2))
    objective_error=error.astype(float)
    if record['variance_weighted_objective']:
        objective_error=objective_error@np.asarray(pca['components']).T/np.sqrt(np.maximum(pca['eigenvalues'],pca['floor']))
    objective=float(np.square(objective_error).mean())
    close(raw,winner['validation_mse'],name+' full CPU validation raw MSE',atol=3e-5,rtol=3e-3)
    close(objective,record['best_validation_objective'],name+' independent CPU validation objective',atol=4e-5,rtol=3e-3)
    test=bundle.splits['test'];tx,tu,tt=rebuild_pairs(bundle,source,cfg['evaluation_seed']+100)
    archive=np.load(sp);prediction,generated=archive['prediction'],archive['generated']
    if prediction.shape!=test.target.shape:
        raise AssertionError(name+' velocity prediction shape')
    cpu=replay(model,tx,tt,test.context)
    close(cpu,prediction,name+' full CPU test velocity replay',atol=3e-4,rtol=3e-3)
    error32=(prediction-tu).astype(float).reshape(len(test),-1)
    vm=independent_velocity(error32,np.zeros_like(error32),pca)
    vm['velocity_mse']=float(np.mean((prediction-tu)**2))
    for key,value in vm.items():close(value,record['velocity_metrics'][key],name+' velocity '+key,atol=1e-8)
    ix=np.linspace(0,len(test)-1,min(len(test),cfg['n_ensemble_contexts']),dtype=int)
    close(archive['indices'],ix,name+' selected forecast contexts')
    if generated.shape!=(len(ix),cfg['ensemble_size'],*test.target.shape[1:]):
        raise AssertionError(name+' generated shape')
    scores=independent_scores(generated,test.target[ix]);compare_scores(scores,record,name)
    solver=record['solver_sensitivity']
    if seed==cfg['seeds'][0]:
        n=min(8,len(ix));refined=np.load(dest/'solver'/f'{name}.npz')['generated']
        if solver is None or solver['contexts']!=n or solver['euler_steps']!=2*cfg['euler_steps'] or refined.shape!=(n,*generated.shape[1:]):
            raise AssertionError(name+' solver subset')
        close(solver['endpoint_rmse'],np.sqrt(np.mean((refined.astype(float)-generated[:n])**2)),name+' solver endpoint')
        for label,value in [('base',mean_scores({k:v[:n] for k,v in scores.items()})),('refined',mean_scores(independent_scores(refined,test.target[ix[:n]])))]:
            for key,number in value.items():close(number,solver[label][key],name+' solver '+label+' '+key)
    elif solver is not None:raise AssertionError(name+' unexpected solver refinement')
    with torch.no_grad():model(torch.as_tensor(vx),torch.as_tensor(vt),torch.as_tensor(val.context,dtype=torch.float32))
    for key,value in model.activity_statistics().items():
        number=float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        # Nonzero support can change at exact thresholds across float32 devices.
        tolerance=3e-3 if 'fraction' in key else 2e-4
        close(number,record['validation_activity'][key],name+' activity '+key,atol=tolerance,rtol=3e-3)
    return {'name':name,'full_validation_replayed':True,'full_test_velocity_replayed':True,
            'validation_objective_cpu':objective,'validation_objective_mps':record['best_validation_objective'],
            'validation_objective_absolute_difference':abs(objective-record['best_validation_objective']),
            'test_velocity_max_abs_difference':float(np.max(np.abs(cpu-prediction))),
            'first_batch_sha256':record['first_batch_sha256'],
            'last_batch_sha256':record['last_batch_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/sparse-extension')
    parser.add_argument('--config',default='configs/sparse_extension.json')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--baseline-results',help='Reuse matching, independently verified sparse baseline artifacts from another suite')
    parser.add_argument('--output')
    args=parser.parse_args();root=Path(args.results);cfg=load(args.config);torch.set_num_threads(2)
    manifest=load(root/'manifest.json');fingerprint=manifest['fingerprint']
    if manifest['config']!=cfg or stable_hash({k:v for k,v in manifest.items() if k!='fingerprint'})!=fingerprint:
        raise AssertionError('Manifest/config/fingerprint differs')
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/'fm_stress'/name)!=digest:raise AssertionError('Numerical source changed: '+name)
    for path,digest in manifest['dataset_sha256'].items():
        if sha(path)!=digest:raise AssertionError('Raw dataset changed: '+path)
    reference=Path(cfg['reference_results'])
    if sha(reference/'manifest.json')!=manifest['reference_manifest_sha256']:
        raise AssertionError('Frozen reference manifest changed')
    reference_manifest=load(reference/'manifest.json')
    for name,digest in reference_manifest['source_sha256'].items():
        if sha(ROOT/'fm_stress'/name)!=digest:raise AssertionError('Old numerical source changed: '+name)
    for key in ('seeds','sources','steps','batch_size','learning_rate','weight_decay','clip_grad',
                'eval_every','evaluation_seed','n_ensemble_contexts','ensemble_size','euler_steps'):
        if cfg[key]!=reference_manifest['config'][key]:raise AssertionError('Reference protocol mismatch: '+key)
    reference_runs=manifest['reference_run_sha256']
    if len(reference_runs)!=96:raise AssertionError('Expected 96 frozen reference runs')
    for path,digest in reference_runs.items():
        if sha(path)!=digest:raise AssertionError('Frozen reference record changed: '+path)
    expected={(d,s,h,k) for d in cfg['datasets'] for s in cfg['sources'] for h in cfg['heads'] for k in cfg['seeds']}
    observed=set();verified=[];datasets=[];batch_groups={}
    for dataset,dc in cfg['datasets'].items():
        dest=root/dataset
        baseline_dest=Path(args.baseline_results)/dataset if args.baseline_results else dest
        required=('data_metadata.json','geometry.json','capacities.json')
        if not all((dest/file).exists() for file in required) or not all((baseline_dest/file).exists() for file in ('sparse_baselines.json','sparse_baseline_samples.npz')):
            if args.allow_partial:continue
            raise AssertionError('Missing dataset artifacts '+dataset)
        bundle=build_data(dc);metadata=load(dest/'data_metadata.json');validate_data(bundle,dc,metadata)
        if metadata!=load(reference/dataset/'data_metadata.json'):
            raise AssertionError('Dataset mismatch against frozen controls')
        geometry=load(dest/'geometry.json');capacities=load(dest/'capacities.json')
        pcas={s:validate_geometry(bundle,geometry[s],s,cfg['evaluation_seed']+300) for s in cfg['sources']}
        validate_target_geometry(bundle,geometry['target']);audit_baseline(bundle,cfg,baseline_dest)
        frozen_count=audit_frozen_controls(bundle,cfg,reference,dataset)
        count=0
        for path in sorted((dest/'runs').glob('*.json')):
            record=load(path);identity=(dataset,record['source'],record['head'],record['seed'])
            if identity not in expected or identity in observed:raise AssertionError('Unexpected/duplicate '+str(identity))
            observed.add(identity)
            result=audit_run(path,bundle,cfg,pcas[record['source']],fingerprint,capacities)
            verified.append(dict(dataset=dataset,**result));count+=1
            group=(dataset,record['source'],record['seed'])
            digest=(record['first_batch_sha256'],record['last_batch_sha256'])
            if group in batch_groups and batch_groups[group]!=digest:raise AssertionError('First training batch differs '+str(group))
            batch_groups[group]=digest
        datasets.append({'dataset':dataset,'verified_runs':count,'frozen_controls_rescored':frozen_count,
                         'data_geometry_and_elastic_baseline_verified':True,'baseline_directory':str(baseline_dest)})
        print('VERIFIED',dataset,count,'runs',flush=True)
    missing=sorted(expected-observed)
    if missing and not args.allow_partial:raise AssertionError(f'Missing {len(missing)} fits: {missing[:5]}')
    report={'complete':not missing,'expected_runs':len(expected),'verified_runs':len(verified),'missing':missing,
            'fingerprint':fingerprint,'datasets':datasets,'runs':verified,'paired_first_batch_groups':len(batch_groups),
            'frozen_control_forecasts_verified':sum(d['frozen_controls_rescored'] for d in datasets),
            'checks':['Frozen numerical source, data, reference run and artifact hashes',
                'Corresponding frozen control forecast archives rescored against unchanged records and exact matched targets',
                'Independent raw-window reconstruction and train-only scaler/source covariance',
                'Train velocity/target PCA and full-rank whitened coordinate buffers',
                'Validation checkpoint selection and independent full CPU validation objective replay',
                'Full CPU test velocity replay and independent raw-coordinate PC metrics',
                'Independent off-diagonal proper forecasts, roughness and doubled-Euler stored arrays',
                'Elastic Net coefficient/validation/residual reconstruction and forecast scores',
                'Cross-head first/last-batch hashes; not a reproduction of complete optimization trajectories']}
    output=Path(args.output) if args.output else root/'independent_verification.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE' if not missing else 'PARTIAL',len(verified),'/',len(expected),'verified;',output)


if __name__=='__main__':main()
