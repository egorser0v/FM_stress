#!/usr/bin/env python3
"""Post-hoc covariance-scaled finite differences; covariance radius is not likelihood."""
from pathlib import Path
import json, hashlib, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from fm_stress.normalized_history_oracle import NormalizedHistoryOracle
from fm_stress.data import GPConfig,kernel_covariance,source_covariance
from fm_stress.experiment import CoordinateModel,save_json
from fm_stress.models import create_model
OUT=ROOT/'results/followup-oracle/finite_perturbation'
CONFIG={'status':'post-hoc supplementary finite-perturbation diagnostic, no training',
'n_histories':256,'directions_per_history':4,'direction_seed':9252026,
'scales':[.01,.1],'times':[.1,.3,.5,.7,.9],'quadrature_orders':[64,128],
'input_subspace':'original velocity PCA components 3:F',
'direction_geometry':'delta = scale * residual-PC conditional-input-covariance Cholesky * standard normal',
'output_metrics':['all components','residual PCs 3:F'],'torch_threads':2,
'interpretation':'Covariance-scaled perturbations are local geometric probes, not likelihood or support certificates.'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(x): return float(np.sqrt(np.square(x).mean()))
def metrics(got,ref):
    e=rms(got-ref);d=rms(ref)
    return {'absolute_rms_error':e,'reference_rms':d,'prediction_rms':rms(got),'relative_rms_error':e/max(d,1e-15),
            'reference_min_row_norm':float(np.linalg.norm(ref,axis=-1).min())}
def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    cfg=json.loads((ROOT/'results/main/config.json').read_text()); geo=json.loads((ROOT/'results/main/geometry.json').read_text())
    gp=GPConfig(**cfg['data']); cells=[(s,k,i) for s,k in [('white','mlp'),('white','s4'),('gp','mlp'),('gp','s4'),('gp','whitened_mlp')] for i in cfg['seeds']]
    files=[Path(__file__),ROOT/'fm_stress/normalized_history_oracle.py',ROOT/'fm_stress/models.py',ROOT/'fm_stress/experiment.py',ROOT/'fm_stress/data.py',ROOT/'results/main/config.json',ROOT/'results/main/geometry.json']
    files += [ROOT/f'results/main/checkpoints/{s}_{k}_seed{i}.pt' for s,k,i in cells]
    files += [ROOT/f'results/followup-oracle/oracle_{s}.npz' for s in ['white','gp']]
    files += [ROOT/f'results/followup-oracle/model_{s}_{k}_seed{i}.npz' for s,k,i in cells]
    frozen={'config':CONFIG,'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'frozen_before_model_probes':True}
    save_json(OUT/'manifest.json',frozen);save_json(OUT/'config.json',CONFIG)
    n=CONFIG['n_histories'];nd=CONFIG['directions_per_history'];z=np.random.default_rng(CONFIG['direction_seed']).normal(size=(n,nd,gp.horizon-2));np.save(OUT/'directions.npy',z)
    refs={};diagnostics=[]
    for source in ['white','gp']:
        old=np.load(ROOT/f'results/followup-oracle/oracle_{source}.npz');h=old['history'];y=old['target'];eps=old['source']
        cs=source_covariance(gp,source);o=NormalizedHistoryOracle(kernel_covariance(gp),gp.lookback,cs,gp.normalization_threshold)
        Q=np.asarray(geo['pca'][source]['components']);R=Q[2:];_,cy=o.target_moments(h);hh=np.repeat(h,nd,axis=0)
        refs[source]=[]
        for ti,t in enumerate(CONFIG['times']):
            x=(1-t)*eps+t*y;cx=t*t*cy+(1-t)**2*cs;cr=R@cx@R.T
            chol=np.linalg.cholesky(cr);delta_unit=np.einsum('nij,nkj->nki',chol,z)@R
            for scale in CONFIG['scales']:
                delta=scale*delta_unit;shift=(x[:,None,:]+delta).reshape(-1,gp.horizon)
                shifted={}
                for order in CONFIG['quadrature_orders']:
                    shifted[order]=np.concatenate([o.flow(shift[a:a+256],t,hh[a:a+256],order=order,jacobian=False)['velocity'] for a in range(0,len(shift),256)]).reshape(n,nd,-1)
                dv=shifted[128]-old['matched_velocity'][ti,:,None,:]
                linear=np.einsum('nij,nkj->nki',old['matched_jacobian'][ti],delta)
                rad=np.sqrt(np.einsum('nki,nki->nk',delta,np.linalg.solve(cx,delta.swapaxes(1,2)).swapaxes(1,2)))
                diag={'source':source,'time':t,'scale':scale,'quadrature':metrics(shifted[64]-old['matched_velocity'][ti,:,None,:],dv),
                      'oracle_linearization':metrics(linear,dv),'oracle_residual_linearization':metrics(linear@R.T,dv@R.T),
                      'mahalanobis_radius_mean':float(rad.mean()),'mahalanobis_radius_max':float(rad.max()),'delta_norm_mean':float(np.linalg.norm(delta,axis=-1).mean())}
                diagnostics.append(diag)
                refs[source].append({'ti':ti,'time':t,'scale':scale,'x':x,'history':h,'shift':shift,'delta':delta,'oracle_change':dv,'R':R})
                np.savez_compressed(OUT/f'reference_{source}_t{t}_s{scale}.npz',delta=delta,oracle_change=dv,oracle_linear_change=linear,mahalanobis_radius=rad)
        print('reference complete',source,flush=True)
    rows=[]
    for source,kind,seed in cells:
        name=f'{source}_{kind}_seed{seed}';state=torch.load(ROOT/f'results/main/checkpoints/{name}.pt',map_location='cpu',weights_only=False)
        model=CoordinateModel(create_model('s4' if kind=='s4' else 'mlp',horizon=gp.horizon,lookback=gp.lookback,**cfg['model']));model.load_state_dict(state['state_dict']);model.eval()
        previous=np.load(ROOT/f'results/followup-oracle/model_{name}.npz');saved={}
        for r in refs[source]:
            t=r['time'];ti=r['ti'];h=r['history'];hh=np.repeat(h,nd,axis=0)
            with torch.no_grad():
                base=model(torch.tensor(r['x'],dtype=torch.float32),torch.full((n,),t),torch.tensor(h,dtype=torch.float32)).numpy()
                pred=np.concatenate([model(torch.tensor(r['shift'][a:a+256],dtype=torch.float32),torch.full((min(256,len(hh)-a),),t),torch.tensor(hh[a:a+256],dtype=torch.float32)).numpy() for a in range(0,len(hh),256)]).reshape(n,nd,-1)
            if not np.allclose(base,previous['prediction'][ti],rtol=1e-5,atol=2e-6):raise AssertionError('Archived base prediction mismatch')
            change=pred-base[:,None,:];lin=np.einsum('nij,nkj->nki',previous['jacobian'][ti],r['delta']);ref=r['oracle_change'];R=r['R']
            rows.append({'source':source,'kind':kind,'seed':seed,'time':t,'scale':r['scale'],'full':metrics(change,ref),
                         'residual':metrics(change@R.T,ref@R.T),'model_linearization':metrics(lin,change),'model_residual_linearization':metrics(lin@R.T,change@R.T)})
            saved[f'change_t{t}_s{r["scale"]}']=change
        np.savez_compressed(OUT/f'model_{name}.npz',**saved);print(name,'complete',flush=True)
    summary=[]
    for source,kind in [('white','mlp'),('white','s4'),('gp','mlp'),('gp','s4'),('gp','whitened_mlp')]:
        for scale in CONFIG['scales']:
            rr=[r for r in rows if r['source']==source and r['kind']==kind and r['scale']==scale]
            entry={'source':source,'kind':kind,'scale':scale,'n_rows':len(rr)}
            for part in ['full','residual','model_linearization','model_residual_linearization']:
                entry[part]={key:float(np.mean([r[part][key] for r in rr])) for key in rr[0][part]}
            entry['residual_relative_error_per_seed']={str(i):float(np.mean([r['residual']['relative_rms_error'] for r in rr if r['seed']==i])) for i in cfg['seeds']}
            summary.append(entry)
    for path,expected in frozen['sha256'].items():
        if sha(ROOT/path)!=expected:raise AssertionError('Frozen input changed: '+path)
    result={'config':CONFIG,'rows':rows,'summary':summary,'numerical_checks':diagnostics,'hashes_unchanged':True,'elapsed_seconds':time.time()-started}
    save_json(OUT/'results.json',result); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
