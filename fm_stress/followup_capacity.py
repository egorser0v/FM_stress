"""Output-rank and training-budget controls for the F64 source/head comparison.

Standalone supplement. Reuses released heads and kernel-scan data/metrics;
never overwrites earlier experiments. Two prefix-selected checkpoints are
scored only after the full training trajectory finishes.
"""
from __future__ import annotations
import argparse, copy, platform, time
from pathlib import Path
import numpy as np
import torch
from . import kernel_scan as ks
from .experiment import CoordinateModel, config_hash, generate, load_config, predict, save_json, tensor
from .metrics import evaluate, velocity_metrics
from .models import create_model, count_parameters, optimizer_groups

OWN_FILES = ['fm_stress/followup_capacity.py', 'configs/followup_capacity.json',
             'docs/followup_capacity_protocol.md']

def build_model(cfg, geometry, head, seed, device):
    spec = cfg['architectures'][head]
    torch.manual_seed(seed)
    model = create_model(spec['family'], horizon=geometry.horizon, lookback=geometry.lookback,
                         **{**cfg['model'], 'width': spec['width'], 'depth': spec['depth']})
    torch.manual_seed(10000 + seed)
    for module in model.encoder.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    return CoordinateModel(model).to(device)

def prefix_eligible(step, budget):
    return step <= budget

def prepare(cfg, out):
    root = Path(__file__).resolve().parents[1]
    files = list(dict.fromkeys(list(ks.SOURCE_FILES) + OWN_FILES))
    m = {'config_sha256': config_hash(cfg), 'source_hashes': {p: ks.file_hash(root / p) for p in files}}
    m['fingerprint'] = config_hash(m)
    for name, value in [('config.json', cfg), ('manifest.json', m)]:
        p = Path(out) / name
        if p.exists() and load_config(p) != value:
            raise ValueError('Provenance changed: use fresh output directory: ' + str(p))
        save_json(p, value)
    return m['fingerprint']

def output_null_diagnostic(model, velocity, source_cov, target, source_paths, generated, prediction):
    """Orthogonal projection onto complement of the fixed output-weight span.

    For q in null(W_out.T), q.T v = q.T b at every x,t,h. Euler consequently
    preserves source variation exactly in these directions. This statement is
    global and exact, unlike a local numerical Jacobian-rank estimate.
    """
    if not hasattr(model.base, 'output_projection') or model.base.__class__.__name__ != 'AdaLNMLP':
        return {'applicable': False}
    w = model.base.output_projection.weight.detach().cpu().double().numpy()
    b = model.base.output_projection.bias.detach().cpu().double().numpy()
    u, sv, _ = np.linalg.svd(w, full_matrices=True)
    tol = np.finfo(np.float64).eps * max(w.shape) * sv[0]
    rank = int(np.sum(sv > tol))
    q = u[:, rank:]
    f = w.shape[0]
    projection = q @ q.T
    forced = (velocity-b) @ q
    pn = (prediction-b) @ q
    endpoint = (generated-source_paths-b) @ q
    return {'applicable': True, 'output_weight_rank': rank, 'null_dimension': f-rank,
            'output_weight_singular_values': sv.tolist(), 'projection': projection.tolist(),
            'forced_velocity_mse_per_coordinate': float(np.square(forced).sum(-1).mean()/f),
            'source_null_variance_per_coordinate': float(np.trace(q.T@source_cov@q)/f),
            'target_null_variance_per_coordinate': float(np.var(target@q,axis=0).sum()/f),
            'prediction_null_identity_max_abs': float(np.max(np.abs(pn), initial=0)),
            'generated_null_identity_max_abs': float(np.max(np.abs(endpoint), initial=0))}

def jacobian_diagnostic(model, source, target, history):
    """Three CPU probes at t=.1,.5,.9, one fixed test history; descriptive only."""
    probe = copy.deepcopy(model).cpu().eval()
    hh = tensor(history[:1], 'cpu')
    rows = []
    for time_value in [.1,.5,.9]:
        tt = torch.tensor([time_value])
        xx = tensor((1-time_value)*source[0]+time_value*target[0], 'cpu')
        j = torch.autograd.functional.jacobian(lambda a: probe(a[None],tt,hh)[0], xx,
                                              vectorize=True).detach().double().numpy()
        sv = np.linalg.svd(j,compute_uv=False)
        rows.append({'time':time_value,'singular_values':sv.tolist(),
                     'numerical_rank_relative_1e-5':int(np.sum(sv>sv[0]*1e-5)),
                     'frobenius_norm':float(np.linalg.norm(j))})
    return {'contexts':1,'threshold_relative':1e-5,'note':'Local numerical rank; not global expressivity proof.', 'probes':rows}

def evaluate_checkpoint(cfg,bundle,model,checkpoint,out,kernel,ds,seed,source,head,budget,fingerprint,training):
    started=time.perf_counter(); root=ks.group_dir(out,kernel,'F64_l8',ds)
    key=f'{source}_{head}_budget{budget}'
    recpath=root/'runs'/f'{key}.json'; samplepath=root/'samples'/f'{key}.npz'
    if recpath.exists():
        old=load_config(recpath)
        if old['fingerprint']!=fingerprint or old['checkpoint_sha256']!=ks.file_hash(checkpoint) or old['samples_sha256']!=ks.file_hash(samplepath):
            raise ValueError('Result mismatch: '+key)
        return old
    ck=torch.load(checkpoint,map_location=next(model.parameters()).device,weights_only=False)
    model.load_state_dict(ck['state_dict']); model.eval(); device=next(model.parameters()).device
    geometry,pca=bundle['geometry'],bundle['pcas'][source]
    test=ks.sample_windows(geometry,cfg['n_test'],ds+2)
    tx,tu,tt,eps=ks.fixed_pairs(geometry,test,source,ds,'test')
    pred=predict(model,tensor(tx,device),tensor(tt,device),tensor(test.context,device)).cpu().numpy()
    n=min(cfg['n_paths'],len(test))
    generated=generate(model,tensor(eps[:n],device),tensor(test.context[:n],device),cfg['euler_steps'],cfg['generation_batch_size'])
    nc,ns=min(cfg['n_ensemble_contexts'],len(test)),cfg['ensemble_size']
    ee=ks.source_draw(geometry,nc*ns,source,ds+ks.ENSEMBLE_OFFSET).reshape(nc,ns,-1)
    hh=tensor(np.repeat(test.context[:nc],ns,axis=0),device)
    ensemble=generate(model,tensor(ee.reshape(-1,geometry.horizon),device),hh,cfg['euler_steps'],cfg['generation_batch_size']).reshape(nc,ns,-1)
    scores=ks.context_scores(ensemble,test.target[:nc]); metrics=evaluate(pred,tu,generated,test.target[:n],pca)
    forecast={k:float(v.mean()) for k,v in scores.items()}
    oracle=ks.oracle_reference(cfg,bundle,test,source,ds,(tx,tu,tt,eps),fingerprint,root/f'oracle_{source}.json')
    solver={}; extras={}
    if ds==cfg['replicates'][0]['data_seed']:
        ni=min(cfg['solver_check']['n_contexts'],nc)
        fine=generate(model,tensor(ee[:ni].reshape(-1,geometry.horizon),device),hh[:ni*ns],cfg['solver_check']['euler_steps'],cfg['generation_batch_size']).reshape(ni,ns,-1)
        sc=ks.context_scores(fine,test.target[:ni]); sb=ks.context_scores(ensemble[:ni],test.target[:ni])
        solver={'contexts':ni,'steps':cfg['solver_check']['euler_steps'],'endpoint_rmse':float(np.sqrt(np.mean((fine-ensemble[:ni])**2))),
                'score_changes':{k:float(sc[k].mean()-sb[k].mean()) for k in sc}}
        extras['ensemble_fine_solver']=fine
    null=output_null_diagnostic(model,tu,ks.source_covariance(geometry,source),test.target,eps[:n],generated,pred)
    jac=jacobian_diagnostic(model,eps,test.target,test.context)
    samplepath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(samplepath,prediction=pred,generated=generated,ensemble=ensemble,
                        path_target=test.target[:n],path_history=test.context[:n],ensemble_target=test.target[:nc],**extras)
    record={'status':'complete','kernel':kernel,'geometry':geometry.to_dict(),'data_seed':ds,'seed':seed,
            'source':source,'head':head,'architecture':cfg['architectures'][head],
            'budget':budget,'total_training_steps':cfg['steps'],'best_step':ck['step'],
            'best_validation_mse':ck['validation'],'parameters':count_parameters(model),
            'device':str(device),'torch':torch.__version__,'fingerprint':fingerprint,
            'checkpoint_sha256':ks.file_hash(checkpoint),'samples_sha256':ks.file_hash(samplepath),
            'metrics':metrics,'group_mse':ks.group_means(metrics['pc_mse'],bundle['groups']),
            'fixed_target_basis_metrics':velocity_metrics(pred,tu,bundle['target_pca']),
            'forecast_means':forecast,'per_context':{k:v.tolist() for k,v in scores.items()},
            'oracle_gap':ks.oracle_gap(ks.group_means(metrics['pc_mse'],bundle['groups']),forecast,oracle),
            'output_null_diagnostic':null,'jacobian':jac,'solver_sensitivity':solver,
            'test_input_hashes':{k:ks.array_hash(v) for k,v in [('raw_history',test.raw_context),('raw_target',test.raw_target),('source',eps),('times',tt),('ensemble_source',ee)]},
            'training':training,'evaluation_seconds':time.perf_counter()-started}
    save_json(recpath,record)
    print('SCORED',kernel,ds,source,head,budget,'CRPS',forecast['fair_crps'],'residual',metrics['residual_mse'],flush=True)
    return record

def train_one(cfg,bundle,out,kernel,ds,seed,source,head,device,fingerprint):
    root=ks.group_dir(out,kernel,'F64_l8',ds); key=f'{source}_{head}'
    donepath=root/'training'/f'{key}.json'
    checkpoints={b:root/'checkpoints'/f'{key}_budget{b}.pt' for b in cfg['budgets']}
    model=build_model(cfg,bundle['geometry'],head,seed,device)
    if donepath.exists() and load_config(donepath).get('complete'):
        training=load_config(donepath)
        if training['fingerprint']!=fingerprint or any(training['checkpoint_hashes'][str(b)]!=ks.file_hash(p) for b,p in checkpoints.items()):
            raise ValueError('Training provenance mismatch')
    else:
        start=time.perf_counter(); geometry=bundle['geometry']; data=bundle['data']
        encoder_hash=ks.array_hash(np.concatenate([p.detach().cpu().numpy().ravel() for p in model.base.encoder.parameters()]))
        opt=torch.optim.AdamW(optimizer_groups(model,cfg['learning_rate'],cfg['weight_decay']))
        y,h=tensor(data['train'].target,device),tensor(data['train'].context,device)
        chol=tensor(np.linalg.cholesky(ks.source_covariance(geometry,source)),device)
        vx,vu,vt,_=ks.fixed_pairs(geometry,data['val'],source,ds,'val')
        vx,vu,vt,vh=[tensor(a,device) for a in (vx,vu,vt,data['val'].context)]
        torch.manual_seed(seed+20000)
        if device.type=='mps':torch.mps.manual_seed(seed+20000)
        best={b:float('inf') for b in cfg['budgets']}; curve=[]; hashes={}
        print('START',kernel,ds,source,head,'params',count_parameters(model),flush=True)
        for step in range(1,cfg['steps']+1):
            model.train(); ix=torch.randint(len(y),(cfg['batch_size'],),device=device)
            z=torch.randn(len(ix),geometry.horizon,device=device); t=torch.rand(len(ix),device=device)
            if step in [1,*cfg['budgets']]:
                hashes[str(step)]={k:ks.array_hash(a.detach().cpu().numpy()) for k,a in [('indices',ix),('normals',z),('time',t)]}
            yy=y[ix]; eps=z@chol.T; x=(1-t[:,None])*eps+t[:,None]*yy
            opt.zero_grad(set_to_none=True); loss=(model(x,t,h[ix])-(yy-eps)).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']);opt.step()
            if step%cfg['eval_every']==0 or step==cfg['steps']:
                model.eval()
                with torch.no_grad():val=float((predict(model,vx,vt,vh)-vu).square().mean().cpu())
                if not np.isfinite(val):raise FloatingPointError(key)
                row={'step':step,'validation_mse':val,'seconds':time.perf_counter()-start};curve.append(row)
                for b in cfg['budgets']:
                    if prefix_eligible(step,b) and val<best[b]:
                        best[b]=val; checkpoints[b].parent.mkdir(parents=True,exist_ok=True)
                        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step,'validation':val,'budget':b,'fingerprint':fingerprint,'config':cfg,'head':head,'seed':seed,'data_seed':ds,'kernel':kernel,'source':source},checkpoints[b])
                save_json(donepath,{'complete':False,'curve':curve,'fingerprint':fingerprint})
                print('PROGRESS',kernel,ds,source,head,step,'val',round(val,6),'seconds',round(row['seconds'],1),flush=True)
        training={'complete':True,'fingerprint':fingerprint,'encoder_initial_sha256':encoder_hash,'batch_hashes':hashes,'curve':curve,'seconds':time.perf_counter()-start,'checkpoint_hashes':{str(b):ks.file_hash(p) for b,p in checkpoints.items()}}
        save_json(donepath,training)
    return [evaluate_checkpoint(cfg,bundle,model,p,out,kernel,ds,seed,source,head,b,fingerprint,training) for b,p in checkpoints.items()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/followup_capacity.json');p.add_argument('--output',default='results/followup-capacity');p.add_argument('--device',default='mps');p.add_argument('--kernel');p.add_argument('--head');p.add_argument('--source');p.add_argument('--data-seed',type=int);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    cfg=load_config(a.config);out=Path(a.output);device=torch.device(a.device)
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent fallback')
    torch.set_num_threads(4);fp=prepare(cfg,out)
    env={'device':str(device),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()}
    ep=out/'environment.json'
    if ep.exists() and load_config(ep)!=env:raise ValueError('Environment changed')
    save_json(ep,env)
    for r in cfg['replicates']:
        if a.data_seed and a.data_seed!=r['data_seed']:continue
        for kernel,heads in cfg['kernel_heads'].items():
            if a.kernel and kernel!=a.kernel:continue
            bundle=ks.prepare_group(cfg,out,kernel,'F64_l8',r['data_seed'],fp)
            if a.prepare_only:continue
            for source in cfg['sources']:
                if a.source and source!=a.source:continue
                for head in heads:
                    if a.head and head!=a.head:continue
                    train_one(cfg,bundle,out,kernel,r['data_seed'],r['seed'],source,head,device,fp)
if __name__=='__main__':main()
