#!/usr/bin/env python3
"""Independent CPU verification of extension data, checkpoints and all saved scores.

Default requires every configured dataset/source/head/seed (96 in the declared
suite). --allow-partial verifies all currently completed records but reports
incomplete status. Original records are never changed. Does not run MPS.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.distance import pdist
import torch
from fm_stress.extended_data import build_data
from fm_stress.extended_models import create_extended_model,parameter_counts


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def close(a,b,where,atol=1e-9,rtol=1e-8):
    try:np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
    except AssertionError as exc:raise AssertionError(where+': '+str(exc)) from exc


def independent_scores(samples,target):
    """Explicit off-diagonal U-statistic, independent of production score code."""
    x,y=np.asarray(samples,float),np.asarray(target,float)
    if x.ndim!=4 or y.shape!=(x.shape[0],x.shape[2],x.shape[3]) or not np.isfinite(x).all():
        raise AssertionError('Invalid saved forecast arrays')
    n,s,f,c=x.shape
    xx,yy=x.reshape(n,s,-1),y.reshape(n,-1)
    crps=[];energy=[];sum_crps=[]
    i,j=np.triu_indices(s,1)
    for h in range(n):
        # Each unordered pair contributes once: this equals half the ordered
        # off-diagonal expectation used in fair ensemble scoring.
        crps.append(float(np.abs(xx[h]-yy[h]).mean()-np.abs(xx[h,i]-xx[h,j]).sum()/(s*(s-1)*f*c)))
        energy.append(float((np.linalg.norm(xx[h]-yy[h],axis=1).mean()-pdist(xx[h]).sum()/(s*(s-1)))/np.sqrt(f*c)))
        a,b=x[h].sum(-1)/np.sqrt(c),y[h].sum(-1)/np.sqrt(c)
        sum_crps.append(float(np.abs(a-b).mean()-np.abs(a[i]-a[j]).sum()/(s*(s-1)*f)))
    low,high=np.quantile(xx,[.05,.95],axis=1,method='linear')
    return dict(fair_crps=np.array(crps),joint_energy_scaled=np.array(energy),
        channel_sum_crps_scaled=np.array(sum_crps),mean_forecast_mse=((x.mean(1)-y)**2).mean((1,2)),
        coverage_90=((yy>=low)&(yy<=high)).mean(1),width_90=(high-low).mean(1),
        generated_roughness=np.abs(x[:,:,1:]-x[:,:,:-1]).mean((1,2,3)),
        target_roughness=np.abs(y[:,1:]-y[:,:-1]).mean((1,2)))


def mean_scores(scores):
    result={k:float(v.mean()) for k,v in scores.items()}
    result['roughness_ratio']=result['generated_roughness']/max(result['target_roughness'],1e-12)
    return result


def compare_scores(scores,record,where,mean_key='forecast_means'):
    for metric,value in scores.items():
        close(value,record['per_context'][metric],where+' per_context '+metric)
    for metric,value in mean_scores(scores).items():
        close(value,record[mean_key][metric],where+' mean '+metric)


def draw_source(cov,n,seed):
    return np.random.default_rng(seed).standard_normal((n,len(cov)))@np.linalg.cholesky(cov).T


def rebuild_pairs(bundle,source,seed,split='test'):
    ds=bundle.splits[split]
    eps=draw_source(bundle.source_covariances[source],len(ds),seed).reshape(ds.target.shape).astype(np.float32)
    times=np.random.default_rng(seed+1).random(len(ds)).astype(np.float32)
    y=ds.target.astype(np.float32)
    return (1-times[:,None,None])*eps+times[:,None,None]*y,y-eps,times


def validate_data(bundle,dc,metadata):
    if metadata!=bundle.metadata:raise AssertionError('Saved/reconstructed data metadata differs')
    c=metadata['channels'];l=metadata['lookback'];f=metadata['horizon']
    if dc['kind']=='real':
        if dc.get('format','csv')=='csv':
            with Path(dc['path']).open(newline='') as fp:
                rows=list(csv.DictReader(fp,delimiter=dc.get('delimiter',',')))
            raw=np.array([[float(row[name]) for name in metadata['channel_names']] for row in rows])
        else:
            raw=np.loadtxt(dc['path'],delimiter=dc.get('delimiter',','),ndmin=2)
            if 'channel_indices' in dc:raw=raw[:,dc['channel_indices']]
        end=metadata['split_row_bounds']['train'][1]
        train=raw[:end]
        close(metadata['scaler_mean'],train.mean(0),'real train mean')
        close(metadata['scaler_scale'],np.where(train.std(0)>1e-8,train.std(0),1),'real train scale')
        mean,scale=np.array(metadata['scaler_mean']),np.array(metadata['scaler_scale'])
        for split,ds in bundle.splits.items():
            start,end=metadata['split_row_bounds'][split]
            if not np.all((ds.starts>=start)&(ds.starts+l+f<=end)):
                raise AssertionError('Window crosses chronological split')
            for i,t in enumerate(ds.starts):
                close(ds.context[i],(raw[t:t+l]-mean)/scale,'real history row reconstruction')
                close(ds.target[i],(raw[t+l:t+l+f]-mean)/scale,'real target row reconstruction')
        if sha(dc['path'])!=metadata['file']['file_sha256']:raise AssertionError('Real file metadata hash differs')
    else:
        # Independent Kronecker construction vs production two-sided einsum.
        t=np.arange(l+f)
        kt=np.exp(-.5*((t[:,None]-t)/dc.get('lengthscale',8.))**2)+dc.get('nugget',1e-6)*np.eye(l+f)
        rho=dc.get('channel_correlation',.6)
        kc=(1-rho)*np.eye(c)+rho*np.ones((c,c)) if c>1 else np.eye(1)
        z=np.random.default_rng(dc.get('seed',97041)).normal(size=(dc['n_train'],(l+f)*c))
        raw=(z@np.kron(np.linalg.cholesky(kt),np.linalg.cholesky(kc)).T).reshape(-1,l+f,c)
        close(metadata['scaler_mean'],raw.mean((0,1)),'synthetic train mean')
        close(metadata['scaler_scale'],raw.std((0,1)),'synthetic train scale')
        norm=(raw-np.array(metadata['scaler_mean']))/np.array(metadata['scaler_scale'])
        close(bundle.splits['train'].context,norm[:,:l],'synthetic joint contexts')
        close(bundle.splits['train'].target,norm[:,l:],'synthetic joint targets')
    points=bundle.splits['train'].target.reshape(-1,c)
    empirical=np.cov(points,rowvar=False)
    if c==1:correlation=np.eye(1)
    else:
        std=np.sqrt(np.maximum(np.diag(empirical),0));safe=np.where(std>1e-12,std,1)
        correlation=empirical/np.outer(safe,safe);np.fill_diagonal(correlation,1)
    shrink=dc.get('source_channel_shrinkage',.05)
    correlation=(1-shrink)*correlation+shrink*np.eye(c)
    close(correlation,metadata['source_channel_correlation'],'train-only source channel correlation')
    tt=np.arange(f);nugget=dc.get('source_nugget',1e-6)
    kt=(np.exp(-.5*((tt[:,None]-tt)/dc.get('source_lengthscale',8.))**2)+nugget*np.eye(f))/(1+nugget)
    for source,expected in [('white',np.kron(np.eye(f),correlation)),('gp',np.kron(kt,correlation))]:
        close(bundle.source_covariances[source],expected,source+' covariance')
        close(np.diag(expected),np.ones(f*c),source+' unit source variance')
        if np.linalg.eigvalsh(expected).min()<=0:raise AssertionError('Nonpositive source covariance')


def validate_geometry_arrays(train_values,val_values,geo,label):
    u=train_values.reshape(len(train_values),-1)
    v=val_values.reshape(len(val_values),-1)
    p=geo['pca'];q=np.array(p['components']);mean=np.array(p['mean']);ev=np.array(p['eigenvalues'])
    close(mean,u.mean(0),label+' PCA training mean')
    close(q@q.T,np.eye(len(q)),label+' PCA orthonormality',atol=1e-10)
    centered=u-u.mean(0);covariance=centered.T@centered/(len(u)-1)
    close(q@covariance@q.T,np.diag(ev),label+' PCA diagonalizes train covariance',atol=1e-8)
    if np.any(np.diff(ev)>1e-10):raise AssertionError(label+' PCA eigenvalues unsorted')
    close(p['floor'],max(ev[0]*1e-6,1e-8),label+' PCA floor')
    variance=(v@q.T).var(0,ddof=1);energy=variance/variance.sum()
    close(geo['validation_top2_energy'],energy[:2].sum(),label+' validation E2')
    close(geo['validation_top2C_energy'],energy[:2*train_values.shape[-1]].sum(),label+' validation E2C')
    per_channel=[]
    for c in range(train_values.shape[-1]):
        a=train_values[:,:,c];b=val_values[:,:,c]
        ac=a-a.mean(0)
        eigenvalues,eigenvectors=np.linalg.eigh(ac.T@ac/(len(a)-1))
        axes=eigenvectors[:,np.argsort(eigenvalues)[::-1]]
        vv=(b@axes).var(0,ddof=1)
        per_channel.append(vv[:2].sum()/vv.sum())
    close(geo['per_channel_top2_energy'],per_channel,label+' per-channel heldout E2')
    if geo.get('geometry_is_diagnostic_only') is not True:raise AssertionError('Extension geometry scope mislabeled')
    return p


def validate_geometry(bundle,geo,source,seed):
    train,val=bundle.splits['train'],bundle.splits['val']
    cov=bundle.source_covariances[source]
    u=train.target-draw_source(cov,len(train),seed).reshape(train.target.shape)
    v=val.target-draw_source(cov,len(val),seed+1).reshape(val.target.shape)
    return validate_geometry_arrays(u,v,geo,source)


def validate_target_geometry(bundle,geo):
    train,val=bundle.splits['train'].target,bundle.splits['val'].target
    validate_geometry_arrays(train,val,geo,'target')
    close(geo['validation_temporal_roughness'],np.abs(val[:,1:]-val[:,:-1]).mean(),'target temporal roughness')
    close(geo['validation_within_window_temporal_std'],val.std(axis=1).mean(),'target temporal std')


def independent_velocity(prediction,velocity,pca):
    pred,truth=np.asarray(prediction,float).reshape(len(prediction),-1),np.asarray(velocity,float).reshape(len(velocity),-1)
    err=pred-truth;q=np.asarray(pca['components']);mse=((err@q.T)**2).mean(0)
    scaled=mse/np.maximum(pca['eigenvalues'],pca['floor'])
    return dict(velocity_mse=(err**2).mean(),pc1_mse=mse[0],pc2_mse=mse[1],residual_mse=mse[2:].mean(),
        pc_mse=mse,pc1_normalized_mse=scaled[0],pc2_normalized_mse=scaled[1],
        residual_normalized_mse=scaled[2:].mean(),pc_normalized_mse=scaled,variance_floor=pca['floor'])


def audit_baselines(bundle,cfg,dc,dest):
    record=load(dest/'baselines.json');archive=np.load(dest/'baseline_samples.npz')
    train,val,test=[bundle.splits[s] for s in ('train','val','test')]
    ix=np.linspace(0,len(test)-1,min(cfg['n_ensemble_contexts'],len(test)),dtype=int)
    close(archive['indices'],ix,'baseline selected contexts')
    x=np.column_stack([train.context.reshape(len(train),-1),np.ones(len(train))]);y=train.target.reshape(len(train),-1)
    vx=np.column_stack([val.context.reshape(len(val),-1),np.ones(len(val))])
    tx=np.column_stack([test.context[ix].reshape(len(ix),-1),np.ones(len(ix))])
    penalty=np.eye(x.shape[1]);penalty[-1,-1]=0
    candidates=[]
    for alpha in cfg['ridge_alphas']:
        weights=np.linalg.solve(x.T@x+alpha*penalty,x.T@y)
        error=((vx@weights-val.target.reshape(len(val),-1))**2).mean()
        candidates.append((error,alpha,weights))
    best=min(candidates,key=lambda row:row[:2])
    close(record['selected_ridge_alpha'],best[1],'ridge validation alpha')
    close(record['ridge_validation_mse'],best[0],'ridge validation error')
    close(archive['ridge_weights'],best[2],'ridge weights',atol=1e-8)
    for saved,(error,alpha,_) in zip(record['ridge_candidates'],candidates):
        close(saved['validation_mse'],error,'ridge candidate validation');close(saved['alpha'],alpha,'ridge candidate alpha')
    f,c=test.target.shape[1:]
    expected={'persistence':np.repeat(test.context[ix,-1:],f,axis=1),
              'ridge':(tx@best[2]).reshape(-1,f,c)}
    period=dc.get('seasonal_period')
    if period and test.context.shape[1]>=period:
        expected['seasonal_naive']=test.context[ix,-period:][:,np.arange(f)%period]
    for name,pred in expected.items():
        close(archive[name],pred,'baseline forecast '+name,atol=1e-8)
        scores=independent_scores(np.repeat(pred[:,None],cfg['ensemble_size'],axis=1),test.target[ix])
        compare_scores(scores,record['records'][name],'baseline '+name,mean_key='means')
    draws=np.random.default_rng(cfg['evaluation_seed']+15).integers(len(y),size=(len(ix),cfg['ensemble_size']))
    close(archive['ridge_residual_draw_indices'],draws,'baseline residual draw indices')
    generated=expected['ridge'][:,None]+(y-x@best[2])[draws].reshape(len(ix),cfg['ensemble_size'],f,c)
    close(archive['ridge_residual_bootstrap'],generated,'residual bootstrap forecasts',atol=1e-8)
    compare_scores(independent_scores(generated,test.target[ix]),record['records']['ridge_residual_bootstrap'],'residual bootstrap',mean_key='means')


def audit_run(record,path,bundle,cfg,pca,fingerprint,capacities,replay):
    name=record['name'];dest=path.parent.parent
    if record['fingerprint']!=fingerprint or record['status']!='complete' or record['device']!='mps':
        raise AssertionError(name+' inconsistent provenance/status/device')
    if record['steps']!=cfg['steps']:raise AssertionError(name+' inconsistent spent steps')
    curve=record['training']
    expected_steps=list(range(cfg['eval_every'],cfg['steps']+1,cfg['eval_every']))
    if not expected_steps or expected_steps[-1]!=cfg['steps']:expected_steps.append(cfg['steps'])
    if [row['step'] for row in curve]!=expected_steps:raise AssertionError(name+' incomplete or reordered validation curve')
    if curve!=load(dest/'training'/f'{name}.json'):raise AssertionError(name+' training log differs')
    winner=min(curve,key=lambda r:r['validation_mse'])
    close(record['best_validation_mse'],winner['validation_mse'],name+' best validation')
    if record['best_step']!=winner['step']:raise AssertionError(name+' wrong checkpoint selection')
    checkpoint_path=dest/'checkpoints'/f'{name}.pt'
    if sha(checkpoint_path)!=record['checkpoint_sha256']:raise AssertionError(name+' checkpoint hash differs')
    ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if ckpt['fingerprint']!=fingerprint or ckpt['step']!=record['best_step'] or ckpt['options']!=record['model_options'] or ckpt['seed']!=record['seed']:
        raise AssertionError(name+' checkpoint metadata differs')
    ds=bundle.splits['test'];tx,tu,tt=rebuild_pairs(bundle,record['source'],cfg['evaluation_seed']+100)
    archive=np.load(dest/'samples'/f'{name}.npz');prediction=archive['prediction'];generated=archive['generated']
    ix=np.linspace(0,len(ds)-1,min(cfg['n_ensemble_contexts'],len(ds)),dtype=int)
    close(ix,archive['indices'],name+' context selection')
    if generated.shape!=(len(ix),cfg['ensemble_size'],*ds.target.shape[1:]) or prediction.shape!=ds.target.shape:
        raise AssertionError(name+' array dimensions differ')
    # Production subtracts float32 arrays before float64 PCA projection.
    error32=(prediction-tu).astype(float).reshape(len(ds),-1)
    vm=independent_velocity(error32,np.zeros_like(error32),pca)
    # Production raw velocity MSE stays float32, unlike PC projection.
    vm['velocity_mse']=np.mean((prediction-tu)**2)
    for key,value in vm.items():close(value,record['velocity_metrics'][key],name+' velocity '+key,atol=1e-8)
    scores=independent_scores(generated,ds.target[ix]);compare_scores(scores,record,name)
    solver=record['solver_sensitivity']
    if record['seed']==cfg['seeds'][0]:
        if solver is None:raise AssertionError(name+' missing required solver refinement')
        n=min(8,len(ix));refined=np.load(dest/'solver'/f'{name}.npz')['generated']
        if solver['contexts']!=n or solver['euler_steps']!=2*cfg['euler_steps'] or refined.shape!=(n,*generated.shape[1:]):
            raise AssertionError(name+' solver subset mismatch')
        close(solver['endpoint_rmse'],np.sqrt(((refined.astype(float)-generated[:n])**2).mean()),name+' solver endpoint')
        for label,vals in [('base',mean_scores({k:v[:n] for k,v in scores.items()})),('refined',mean_scores(independent_scores(refined,ds.target[ix[:n]])))]:
            for key,value in vals.items():close(value,solver[label][key],name+' solver '+label+' '+key)
    elif solver is not None:raise AssertionError(name+' undeclared solver refinement')
    model=create_extended_model(**ckpt['options']);model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    if parameter_counts(model)!=record['parameters']:raise AssertionError(name+' parameter accounting mismatch')
    if record['model_options']['width']!=capacities[record['head']]['width']:raise AssertionError(name+' capacity width mismatch')
    vx,vu,vt=rebuild_pairs(bundle,record['source'],cfg['evaluation_seed'],split='val')
    val_history=bundle.splits['val'].context
    validation_predictions=[]
    with torch.no_grad():
        for i in range(0,len(vx),256):
            validation_predictions.append(model(torch.as_tensor(vx[i:i+256]),torch.as_tensor(vt[i:i+256]),
                torch.as_tensor(val_history[i:i+256],dtype=torch.float32)).numpy())
    validation_cpu_mse=float(np.mean((np.concatenate(validation_predictions)-vu)**2))
    close(validation_cpu_mse,record['best_validation_mse'],name+' independently replayed checkpoint validation MSE',atol=3e-5,rtol=2e-3)
    if replay:
        with torch.no_grad():
            cpu=model(torch.as_tensor(tx[:8]),torch.as_tensor(tt[:8]),torch.as_tensor(ds.context[:8],dtype=torch.float32)).numpy()
        close(cpu,prediction[:8],name+' CPU replay vs MPS predictions',atol=3e-4,rtol=3e-3)
    return {'name':name,'device':record['device'],'checkpoint_sha256':record['checkpoint_sha256'],'cpu_prediction_replayed':replay,
            'validation_cpu_mse':validation_cpu_mse,'validation_mps_mse':record['best_validation_mse'],
            'validation_absolute_difference':abs(validation_cpu_mse-record['best_validation_mse'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/extension')
    parser.add_argument('--config',default='configs/extension.json')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--skip-prediction-replay',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args();root=Path(args.results);cfg=load(args.config)
    torch.set_num_threads(2)
    manifest=load(root/'manifest.json');fingerprint=manifest['fingerprint']
    if manifest['config']!=cfg:raise AssertionError('Manifest does not match requested full configuration')
    if stable_hash({k:v for k,v in manifest.items() if k!='fingerprint'})!=fingerprint:raise AssertionError('Invalid manifest fingerprint')
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/'fm_stress'/name)!=digest:raise AssertionError('Numerical source changed: '+name)
    for path,digest in manifest['dataset_sha256'].items():
        if sha(path)!=digest:raise AssertionError('Input dataset changed: '+path)
    expected={(dataset,source,head,seed) for dataset in cfg['datasets'] for source in cfg['sources'] for head in cfg['heads'] for seed in cfg['seeds']}
    observed=set();verified=[];datasets=[]
    for name,dc in cfg['datasets'].items():
        dest=root/name
        if not all((dest/file).exists() for file in ('data_metadata.json','capacities.json','geometry.json','baselines.json','baseline_samples.npz')):
            if args.allow_partial:continue
            raise AssertionError('Missing dataset outputs '+name)
        bundle=build_data(dc);validate_data(bundle,dc,load(dest/'data_metadata.json'))
        capacities=load(dest/'capacities.json');geometries=load(dest/'geometry.json')
        pcas={source:validate_geometry(bundle,geometries[source],source,cfg['evaluation_seed']+300) for source in cfg['sources']}
        validate_target_geometry(bundle,geometries['target'])
        audit_baselines(bundle,cfg,dc,dest)
        count=0
        for path in sorted((dest/'runs').glob('*.json')):
            record=load(path);identity=(name,record['source'],record['head'],record['seed'])
            if identity not in expected or identity in observed:raise AssertionError('Unexpected/duplicate run '+str(identity))
            observed.add(identity)
            verified.append(dict(dataset=name,**audit_run(record,path,bundle,cfg,pcas[record['source']],fingerprint,capacities,not args.skip_prediction_replay)))
            count+=1
        datasets.append({'dataset':name,'verified_runs':count,'split_sizes':bundle.metadata['split_sizes'],'data_and_baselines_verified':True})
        print('VERIFIED',name,count,'runs',flush=True)
    missing=sorted(expected-observed)
    if missing and not args.allow_partial:raise AssertionError(f'Missing {len(missing)} configured runs: {missing[:5]}')
    report={'complete':not missing,'expected_runs':len(expected),'verified_runs':len(verified),'missing':missing,
        'fingerprint':fingerprint,'datasets':datasets,'runs':verified,
        'checks':['manifest and numerical-source hashes','raw-data hashes and train-only standardization',
            'chronological window boundaries and exact row reconstruction','shared train-fit source channel covariance',
            'train-only target/velocity PCA eigenbases; held-out E2/E2C/per-channel spectra and target temporal variation','validation checkpoint selection, full CPU replay of chosen checkpoint validation MSE, and checkpoint hashes',
            'independent off-diagonal proper scores and all saved velocity metrics','fixed doubled-Euler subset arrays',
            'train-only ridge fit/validation selection and all deterministic/resampled baselines',
            'optional first-eight CPU checkpoint prediction replay against saved MPS outputs']}
    output=Path(args.output) if args.output else root/'independent_verification.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE' if not missing else 'PARTIAL',len(verified),'/',len(expected),'verified;',output)


if __name__=='__main__':main()
