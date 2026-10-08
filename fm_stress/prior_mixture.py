"""Original-setting covariance-mixture sweep and a complete GP coordinate/loss control."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .data import GPConfig, sample_dataset, kernel_covariance
from .metrics import fit_pca, spectrum, evaluate, velocity_metrics, PCA
from .models import create_model, optimizer_groups, count_parameters
from .experiment import config_hash, load_config, save_json, tensor, predict, generate
from .ensemble_evaluation import context_scores
from .prior_mixture_diagnostics import gradient_diagnostics

SOURCE_FILES = ('fm_stress/prior_mixture.py','fm_stress/prior_mixture_diagnostics.py',
                'fm_stress/models.py','fm_stress/data.py','fm_stress/metrics.py',
                'fm_stress/experiment.py','fm_stress/ensemble_evaluation.py')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def source_name(alpha):
    return 'white' if alpha == 0 else 'gp' if alpha == 1 else f'mix{round(alpha*100):03d}'


def source_covariance(cfg, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('Mixture weight must be in [0,1]')
    gp=GPConfig(**cfg['data'])
    k=kernel_covariance(gp,gp.horizon)/(gp.variance+gp.nugget)
    return (1-alpha)*np.eye(gp.horizon)+alpha*k


def source_draw(cfg,n,alpha,seed):
    return np.random.default_rng(seed).standard_normal((n,cfg['data']['horizon'])) @ np.linalg.cholesky(source_covariance(cfg,alpha)).T


def identities(cfg):
    return [(int(d),float(a),h,int(s)) for d in cfg['data_seeds'] for a in cfg['alphas']
            for h in (cfg['heads']+(cfg.get('gp_control_heads',[]) if a==1 else [])) for s in cfg['seeds']]


def manifest(cfg):
    root=Path(__file__).resolve().parents[1]
    m={'config_sha256':config_hash(cfg),'source_hashes':{n:file_hash(root/n) for n in SOURCE_FILES}}
    m['fingerprint']=config_hash(m)
    return m


def prepare(cfg,out):
    out=Path(out); m=manifest(cfg)
    for name,record in [('config.json',cfg),('manifest.json',m)]:
        path=out/name
        if path.exists() and load_config(path)!=record:
            raise ValueError('Frozen configuration or numerical source changed: '+str(path))
        save_json(path,record)
    bundles={}
    for d in cfg['data_seeds']:
        gp=GPConfig(**cfg['data'])
        data={split:sample_dataset(gp,cfg['n_'+split],d+i) for i,split in enumerate(('train','val'))}
        tpca=fit_pca(data['train'].target)
        geo={'data_seed':d,'config_sha256':config_hash(cfg),'fingerprint':m['fingerprint'],
             'target_pca':tpca.to_dict(),'validation_target':spectrum(data['val'].target,tpca),
             'test_inspected':False,'sources':{},'split_hashes':{split:{'history':array_hash(ds.context),'target':array_hash(ds.target)} for split,ds in data.items()}}
        pcas={}
        for alpha in cfg['alphas']:
            name=source_name(alpha)
            eps=source_draw(cfg,cfg['n_train'],alpha,d+100)
            pca=fit_pca(data['train'].target-eps); pcas[name]=pca
            ve=source_draw(cfg,cfg['n_val'],alpha,d+110)
            spca=fit_pca(eps)
            geo['sources'][name]={'alpha':alpha,'covariance':source_covariance(cfg,alpha).tolist(),
                'pca':pca.to_dict(),'validation_velocity':spectrum(data['val'].target-ve,pca),
                'validation_source':spectrum(ve,spca),
                'floored_directions':int(np.sum(pca.eigenvalues<pca.floor))}
        t=geo['validation_target']
        geo['gate_pass']=bool(geo['sources']['gp']['validation_velocity']['top2_energy']>.9 and
            geo['sources']['white']['validation_velocity']['top2_energy']<.9 and
            t['mean_within_patch_std']>.02 and t['roughness']>.002)
        path=out/f'dataset_{d}'/'geometry.json'
        if path.exists() and load_config(path)!=geo:
            raise ValueError('Geometry differs from frozen record')
        save_json(path,geo)
        bundles[d]=(gp,data,pcas,geo)
    if not all(b[-1]['gate_pass'] for b in bundles.values()):
        raise ValueError('At least one independent dataset failed geometry gate; no training permitted')
    return bundles


class MixtureHead(torch.nn.Module):
    def __init__(self,base,pca,whiten_input=False,balanced=False):
        super().__init__();self.base=base;self.whiten_input=whiten_input;self.balanced=balanced
        self.register_buffer('q',tensor(pca.components.T,'cpu'));self.register_buffer('scales',tensor(pca.scales,'cpu'))
    def coordinates(self,x):return (x@self.q)/self.scales
    def inverse(self,z):return (z*self.scales)@self.q.T
    def forward(self,x,t,h):
        return self.inverse(self.base(self.coordinates(x),t,h)) if self.whiten_input else self.base(x,t,h)


def make_model(cfg,head,pca,seed,device):
    if head not in ('mlp','s4','whitened_mlp','balanced_mlp','whitened_raw_mlp'):
        raise ValueError(head)
    torch.manual_seed(seed)
    base=create_model('s4' if head=='s4' else 'mlp',horizon=cfg['data']['horizon'],lookback=cfg['data']['lookback'],**cfg['model'])
    torch.manual_seed(10000+seed)
    for module in base.encoder.modules():
        if hasattr(module,'reset_parameters'):module.reset_parameters()
    return MixtureHead(base,pca,head in ('whitened_mlp','whitened_raw_mlp'),head in ('whitened_mlp','balanced_mlp')).to(device)


def objective(model,pred,u):
    e=pred-u
    return (model.coordinates(e) if model.balanced else e).square().mean()


def fixed_pairs(cfg,ds,alpha,data_seed,split):
    offset={'val':800,'test':900}[split]
    e=source_draw(cfg,len(ds),alpha,data_seed+offset)
    t=np.random.default_rng(data_seed+offset+1).uniform(0,1,len(ds))
    return (1-t[:,None])*e+t[:,None]*ds.target,ds.target-e,t,e


def learning_rate(cfg,step):
    frac=(step-1)/max(cfg['steps']-1,1)
    return cfg['final_learning_rate']+.5*(cfg['learning_rate']-cfg['final_learning_rate'])*(1+math.cos(math.pi*frac))


def train_run(cfg,bundle,out,data_seed,alpha,head,seed,device):
    gp,data,pcas,geo=bundle;name=source_name(alpha);pca=pcas[name]
    root=Path(out);dest=root/f'dataset_{data_seed}';run=f'{name}_{head}_seed{seed}'
    fingerprint=load_config(root/'manifest.json')['fingerprint'];rp=dest/'runs'/(run+'.json')
    checkpoint=dest/'checkpoints'/(run+'.pt');samplepath=dest/'samples'/(run+'.npz')
    if rp.exists():
        old=load_config(rp)
        if old['fingerprint']!=fingerprint or old['checkpoint_sha256']!=file_hash(checkpoint) or old['samples_sha256']!=file_hash(samplepath):
            raise ValueError('Cannot reuse incompatible result '+run)
        print('SKIP',data_seed,run,flush=True);return old
    started=time.perf_counter();model=make_model(cfg,head,pca,seed,device)
    encoder_hash=array_hash(np.concatenate([p.detach().cpu().numpy().ravel() for p in model.base.encoder.parameters()]))
    optimizer=torch.optim.AdamW(optimizer_groups(model,cfg['learning_rate'],cfg['weight_decay']))
    y,h=tensor(data['train'].target,device),tensor(data['train'].context,device)
    chol=tensor(np.linalg.cholesky(source_covariance(cfg,alpha)),device)
    vx,vu,vt,_=fixed_pairs(cfg,data['val'],alpha,data_seed,'val')
    vx,vu,vt,vh=[tensor(a,device) for a in (vx,vu,vt,data['val'].context)]
    nprobe=cfg['gradient_probe_size'];probe=(vx[:nprobe],vt[:nprobe],vh[:nprobe],vu[:nprobe],pca)
    grads=[]
    if 0 in cfg['gradient_steps']:grads.append({'step':0,**gradient_diagnostics(model,*probe)})
    torch.manual_seed(30000+seed)
    if device.type=='mps':torch.mps.manual_seed(30000+seed)
    curve=[];best=float('inf');best_step=0;batch_hashes={}
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    print('START',data_seed,run,'device',device,'params',count_parameters(model),flush=True)
    for step in range(1,cfg['steps']+1):
        model.train();lr=learning_rate(cfg,step)
        for group in optimizer.param_groups:group['lr']=lr
        ix=torch.randint(len(y),(cfg['batch_size'],),device=device)
        z=torch.randn(len(ix),gp.horizon,device=device);e=z@chol.T
        t=torch.rand(len(ix),device=device);yy,hh=y[ix],h[ix]
        if step in (1,cfg['steps']):
            batch_hashes[str(step)]={k:array_hash(a.detach().cpu().numpy()) for k,a in [('indices',ix),('normals',z),('time',t)]}
        x=(1-t[:,None])*e+t[:,None]*yy;u=yy-e
        optimizer.zero_grad(set_to_none=True);loss=objective(model,model(x,t,hh),u)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']);optimizer.step()
        if step in cfg['gradient_steps']:
            grads.append({'step':step,**gradient_diagnostics(model,*probe)})
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            model.eval()
            with torch.no_grad():
                prediction=predict(model,vx,vt,vh);val=objective(model,prediction,vu).item()
                raw=(prediction-vu).square().mean().item()
            if not np.isfinite(val):raise FloatingPointError(run)
            curve.append({'step':step,'train_objective':loss.item(),'validation_objective':val,'validation_raw_mse':raw,'learning_rate':lr,'seconds':time.perf_counter()-started})
            if val<best:
                best=val;best_step=step
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'config':cfg,'data_seed':data_seed,'alpha':alpha,'head':head,'seed':seed,'step':step,
                    'pca':pca.to_dict(),'fingerprint':fingerprint},checkpoint)
            save_json(dest/'training'/(run+'.json'),curve)
            print('PROGRESS',data_seed,run,step,'val',round(val,6),flush=True)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['state_dict']);model.eval()
    grads.append({'step':best_step,'selected_checkpoint':True,**gradient_diagnostics(model,*probe)})
    # No test access before training/checkpoint selection has finished.
    test=sample_dataset(gp,cfg['n_test'],data_seed+2)
    tx,tu,tt,teps=fixed_pairs(cfg,test,alpha,data_seed,'test')
    pred=predict(model,tensor(tx,device),tensor(tt,device),tensor(test.context,device)).cpu().numpy()
    n=min(cfg['n_paths'],len(test));gen=generate(model,tensor(teps[:n],device),tensor(test.context[:n],device),cfg['euler_steps'],cfg['generation_batch_size'])
    nc=min(cfg['n_ensemble_contexts'],len(test));ns=cfg['ensemble_size']
    es=source_draw(cfg,nc*ns,alpha,data_seed+1000).reshape(nc,ns,gp.horizon)
    eh=np.repeat(test.context[:nc],ns,axis=0)
    ensemble=generate(model,tensor(es.reshape(-1,gp.horizon),device),tensor(eh,device),cfg['euler_steps'],cfg['generation_batch_size']).reshape(nc,ns,gp.horizon)
    scores=context_scores(ensemble,test.target[:nc]);solver={};extra={}
    if seed==cfg['seeds'][0]:
        ni=min(8,nc);gen128=generate(model,tensor(es[:ni].reshape(-1,gp.horizon),device),tensor(eh[:ni*ns],device),2*cfg['euler_steps'],cfg['generation_batch_size']).reshape(ni,ns,gp.horizon)
        s128=context_scores(gen128,test.target[:ni]);s64=context_scores(ensemble[:ni],test.target[:ni])
        solver={'n_contexts':ni,'euler_steps':2*cfg['euler_steps'],'endpoint_rmse':float(np.sqrt(np.mean((gen128.astype(float)-ensemble[:ni])**2))),
                'score_changes':{k:float(s128[k].mean()-s64[k].mean()) for k in s64}}
        extra['generated128']=gen128
    samplepath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(samplepath,prediction=pred,velocity=tu,time=tt,x=tx,history=test.context,target=test.target,source=teps,
                        generated=gen,ensemble=ensemble,ensemble_source=es,ensemble_history=test.context[:nc],ensemble_target=test.target[:nc],**extra)
    tp=geo['target_pca'];tpca=PCA(np.array(tp['mean']),np.array(tp['components']),np.array(tp['eigenvalues']),tp['floor'])
    record={'run':run,'data_seed':data_seed,'source':name,'alpha':alpha,'head':head,'seed':seed,
        'config_sha256':config_hash(cfg),'fingerprint':fingerprint,'status':'complete','steps':cfg['steps'],
        'parameters':count_parameters(model),'device':str(device),'best_step':best_step,'best_validation_objective':best,
        'encoder_initial_sha256':encoder_hash,'batch_hashes':batch_hashes,'curve':curve,'gradient_diagnostics':grads,
        'metrics':evaluate(pred,tu,gen,test.target[:n],pca),'fixed_target_basis_metrics':velocity_metrics(pred,tu,tpca),
        'forecast_means':{k:float(v.mean()) for k,v in scores.items()},'per_context':{k:v.tolist() for k,v in scores.items()},
        'solver_sensitivity':solver,'test_spectrum':spectrum(tu,pca),
        'checkpoint_sha256':file_hash(checkpoint),'samples_sha256':file_hash(samplepath),'seconds':time.perf_counter()-started}
    save_json(rp,record)
    print('DONE',data_seed,run,'seconds',round(record['seconds'],1),'CRPS',round(record['forecast_means']['fair_crps'],6),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/prior_mixture.json');p.add_argument('--output',default='results/prior-mixture')
    p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--data-seed',type=int);p.add_argument('--head');p.add_argument('--seed',type=int)
    a=p.parse_args();cfg=load_config(a.config);torch.set_num_threads(4);device=torch.device(a.device)
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
    bundles=prepare(cfg,a.output)
    if a.prepare_only:
        print('GEOMETRY',[(d,b[-1]['gate_pass']) for d,b in bundles.items()],flush=True);return
    save_json(Path(a.output)/('environment_'+(a.head or 'all')+'.json'),{'device':str(device),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()})
    for d,alpha,head,seed in identities(cfg):
        if a.data_seed is not None and a.data_seed!=d:continue
        if a.head is not None and a.head!=head:continue
        if a.seed is not None and a.seed!=seed:continue
        train_run(cfg,bundles[d],a.output,d,alpha,head,seed,device)

if __name__=='__main__':main()
