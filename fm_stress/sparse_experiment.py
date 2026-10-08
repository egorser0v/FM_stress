"""Predeclared sparse/whitening extension; previous source and results immutable."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .extended_data import build_data,sample_source
from .extended_models import create_extended_model,parameter_counts
from .extended_metrics import forecast_scores,means
from .extended_experiment import pairs,geometry,target_geometry,evaluation_indices,source_hashes as old_hashes
from .experiment import save_json,config_hash,tensor,predict,generate
from .metrics import velocity_metrics
from .models import optimizer_groups
from .sparse_coordinates import JointCoordinates
from .sparse_moe import TinyMoE

PAIRS={'history_topk_ae':'history_dense_ae','history_conv_sparse':'history_conv_dense',
       'dictionary_topk':'dictionary_dense','moe_topk':'moe_dense'}


def hashes():
    root=Path(__file__).parent
    return {**old_hashes(),**{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
        ('sparse_experiment.py','sparse_coordinates.py','sparse_models.py','sparse_moe.py','sparse_baselines.py')}}


def make_base(options):
    opts=dict(options);head=opts.pop('head')
    if head in ('mlp_whitened','mlp_balanced','mlp','unet'):
        return create_extended_model(head='mlp' if head.startswith('mlp') else head,**opts)
    if head in ('moe_dense','moe_topk'):
        return TinyMoE(**opts,topk=2 if head=='moe_topk' else None)
    from .sparse_models import create_sparse_model
    return create_sparse_model(head=head,**opts)


def make_model(options,pca):
    return JointCoordinates(make_base(options),pca,
                            transform_input=options['head']=='mlp_whitened',
                            balanced_loss=options['head']=='mlp_balanced')


def model_options(ds,head,width,cfg,seed):
    opts=dict(head=head,lookback=ds.context.shape[1],horizon=ds.target.shape[1],channels=ds.target.shape[2],width=width)
    if not head.startswith('mlp') and head not in ('unet','moe_dense','moe_topk'):
        opts.update(cfg['sparse_model']);opts['gate_seed']=seed+71000
    return opts


def select_widths(ds,cfg,reference):
    capacities={}
    # Identical width within each sparse/dense pair. Group capacity selection
    # uses the dense member only, no data outcomes. L0 uses original U-Net width.
    for head in cfg['heads']:
        if head in ('mlp_whitened','mlp_balanced'):
            width=reference['mlp']['width']
        elif head=='unet_l0':
            width=reference['unet']['width']
        elif head in PAIRS:
            width=capacities[PAIRS[head]]['width']
        else:
            choices=[]
            for w in range(4,113):
                torch.manual_seed(0)
                model=make_base(model_options(ds,head,w,cfg,cfg['seeds'][0]))
                count=parameter_counts(model)['allocated_trainable']
                choices.append((abs(count-cfg['parameter_target']),w))
            width=min(choices)[1]
        torch.manual_seed(0)
        model=make_base(model_options(ds,head,width,cfg,cfg['seeds'][0]))
        capacities[head]={'width':width,**parameter_counts(model)}
    return capacities


def cpu_state(model):
    return {k:v.detach().cpu() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in model.state_dict().items()}


def scalar_stats(model):
    return {k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in model.activity_statistics().items()}


def train_cell(bundle,cfg,head,source,seed,options,pca,out,fingerprint,device,train_only=False):
    name=f'{source}_{head}_seed{seed}';path=out/'runs'/f'{name}.json'
    checkpoint=out/'checkpoints'/f'{name}.pt';sample_path=out/'samples'/f'{name}.npz'
    if path.exists():
        old=json.loads(path.read_text())
        if old['fingerprint']!=fingerprint or old['status']!='complete':raise RuntimeError('Mixed run provenance')
        for key,file in [('checkpoint_sha256',checkpoint),('samples_sha256',sample_path)]:
            if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=old[key]:raise RuntimeError(f'Corrupted {file}')
        print('SKIP',out.name,name,flush=True);return
    start=time.perf_counter();torch.manual_seed(seed)
    if device.type=='mps':torch.mps.manual_seed(seed)
    model=make_model(options,pca).to(device)
    optimizer=torch.optim.AdamW(optimizer_groups(model,cfg['learning_rate'],cfg['weight_decay']))
    train,val,test=[bundle.splits[s] for s in ('train','val','test')]
    y,h=tensor(train.target,device),tensor(train.context,device)
    chol=tensor(np.linalg.cholesky(bundle.source_covariances[source]),device)
    vx,vu,vt,_=pairs(bundle,'val',source,cfg['evaluation_seed'])
    vx,vu,vt,vh=[tensor(a,device) for a in (vx,vu,vt,val.context)]
    torch.manual_seed(seed+30000)
    if device.type=='mps':torch.mps.manual_seed(seed+30000)
    curve=[];best=float('inf');best_step=0;batch_hash=None;last_batch_hash=None
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    print('START',out.name,name,parameter_counts(model),flush=True)
    for step in range(1,cfg['steps']+1):
        model.train()
        ix=torch.randint(len(y),(cfg['batch_size'],),device=device)
        yy,hh=y[ix],h[ix]
        eps=(torch.randn(len(ix),chol.shape[0],device=device)@chol.T).reshape_as(yy)
        t=torch.rand(len(ix),device=device)
        if step in (1,cfg['steps']):
            digest=hashlib.sha256(ix.cpu().numpy().tobytes()+eps.cpu().numpy().tobytes()+t.cpu().numpy().tobytes()).hexdigest()
            if step==1:batch_hash=digest
            last_batch_hash=digest
        xx=(1-t[:,None,None])*eps+t[:,None,None]*yy
        optimizer.zero_grad(set_to_none=True)
        prediction=model(xx,t,hh)
        main_loss=model.objective(prediction,yy-eps)
        aux=model.auxiliary_loss();loss=main_loss+aux
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']);optimizer.step()
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            model.eval()
            # Selection always held-out CFM objective; reconstruction/gating
            # penalties are regularizers, not checkpoint-selection metrics.
            with torch.no_grad():
                vp=predict(model,vx,vt,vh)
                validation=float(model.objective(vp,vu).cpu())
                raw=float((vp-vu).square().mean().cpu())
            if not np.isfinite(validation):raise FloatingPointError(name)
            curve.append({'step':step,'train_loss':float(loss.detach().cpu()),'train_cfm':float(main_loss.detach().cpu()),
                          'train_aux':float(aux.detach().cpu()),'validation_objective':validation,'validation_mse':raw,
                          'seconds':time.perf_counter()-start})
            if validation<best:
                best,best_step=validation,step
                torch.save({'state_dict':cpu_state(model),'options':options,'step':step,'seed':seed,
                            'fingerprint':fingerprint,'pca':pca.to_dict()},checkpoint)
            save_json(out/'training'/f'{name}.json',curve)
            print(out.name,name,step,'val',round(validation,6),'raw',round(raw,6),flush=True)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);model.eval()
    if train_only:
        save_json(out/'pilot'/f'{name}.json',{'best_validation_objective':best,'best_step':best_step,
                  'training':curve,'seconds':time.perf_counter()-start,'options':options})
        return
    tx,tu,tt,_=pairs(bundle,'test',source,cfg['evaluation_seed']+100)
    with torch.no_grad():
        prediction=predict(model,tensor(tx,device),tensor(tt,device),tensor(test.context,device)).cpu().numpy()
        # Diagnostics on all validation windows, not just final minibatch.
        model(vx,vt,vh);activity=scalar_stats(model)
    vm=velocity_metrics(prediction.reshape(len(test),-1),tu.reshape(len(test),-1),pca)
    ix=evaluation_indices(test,cfg['n_ensemble_contexts'])
    n,s,f,c=len(ix),cfg['ensemble_size'],test.target.shape[1],test.target.shape[2]
    eps=sample_source(bundle,n*s,source,cfg['evaluation_seed']+200)
    hh=np.repeat(test.context[ix],s,axis=0)
    gen=generate(model,tensor(eps,device),tensor(hh,device),cfg['euler_steps'],cfg['generation_batch_size']).reshape(n,s,f,c)
    scores=forecast_scores(gen,test.target[ix]);solver=None
    if seed==cfg['seeds'][0]:
        ns=min(8,n)
        refined=generate(model,tensor(eps[:ns*s],device),tensor(hh[:ns*s],device),2*cfg['euler_steps'],cfg['generation_batch_size']).reshape(ns,s,f,c)
        solver={'contexts':ns,'euler_steps':2*cfg['euler_steps'],
                'endpoint_rmse':float(np.sqrt(np.mean((refined.astype(float)-gen[:ns])**2))),
                'base':means({k:v[:ns] for k,v in scores.items()}),
                'refined':means(forecast_scores(refined,test.target[ix[:ns]]))}
        (out/'solver').mkdir(exist_ok=True);np.savez_compressed(out/'solver'/f'{name}.npz',generated=refined)
    sample_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(sample_path,generated=gen,prediction=prediction,indices=ix)
    record={'name':name,'status':'complete','fingerprint':fingerprint,'head':head,'source':source,'seed':seed,
            'device':str(device),'model_options':options,'parameters':parameter_counts(model),'steps':cfg['steps'],
            'best_step':best_step,'best_validation_objective':best,'training':curve,'velocity_metrics':vm,
            'forecast_means':means(scores),'per_context':{k:v.tolist() for k,v in scores.items()},
            'solver_sensitivity':solver,'validation_activity':activity,'first_batch_sha256':batch_hash,'last_batch_sha256':last_batch_hash,
            'coordinate_input':model.transform_input,'variance_weighted_objective':model.balanced_loss,
            'pca_floored_components':int((pca.eigenvalues<pca.floor).sum()),
            'seconds':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'samples_sha256':hashlib.sha256(sample_path.read_bytes()).hexdigest()}
    save_json(path,record)
    print('DONE',out.name,name,'CRPS',round(record['forecast_means']['fair_crps'],6),'seconds',round(record['seconds'],1),flush=True)
    del model,optimizer
    if device.type=='mps':torch.mps.empty_cache()


def run(cfg,out,device,dataset_filter=None,head_filter=None,pilot=False):
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent fallback')
    torch.set_num_threads(4);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reference=Path(cfg['reference_results'])
    references={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(reference.glob('*/runs/*.json'))}
    manifest={'config':cfg,'source_sha256':hashes(),'reference_manifest_sha256':hashlib.sha256((reference/'manifest.json').read_bytes()).hexdigest(),
              'reference_run_sha256':references,'dataset_sha256':{d['path']:hashlib.sha256(Path(d['path']).read_bytes()).hexdigest() for d in cfg['datasets'].values() if d['kind']=='real'}}
    fingerprint=config_hash(manifest);manifest['fingerprint']=fingerprint
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise RuntimeError('Changed provenance: use fresh output directory')
    save_json(out/'manifest.json',manifest)
    save_json(out/'environment.json',{'device':str(device),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()})
    for name,dc in cfg['datasets'].items():
        if dataset_filter and dataset_filter!=name:continue
        dest=out/name;dest.mkdir(exist_ok=True)
        bundle=build_data(dc);save_json(dest/'data_metadata.json',bundle.metadata)
        refmeta=json.loads((reference/name/'data_metadata.json').read_text())
        if bundle.metadata!=refmeta:raise RuntimeError('Dataset differs from frozen reference')
        capacities=select_widths(bundle.splits['train'],cfg,json.loads((reference/name/'capacities.json').read_text()))
        save_json(dest/'capacities.json',capacities)
        pcas,geo={},{}
        for source in cfg['sources']:pcas[source],geo[source]=geometry(bundle,source,cfg['evaluation_seed']+300)
        geo['target']=target_geometry(bundle);save_json(dest/'geometry.json',geo)
        # CPU Elastic Net is a separate CLI stage, allowing independent MPS
        # training. Final audit requires its completed baseline artifacts.
        for seed in cfg['seeds']:
            for source in cfg['sources']:
                for head in cfg['heads']:
                    if head_filter and head!=head_filter:continue
                    options=model_options(bundle.splits['train'],head,capacities[head]['width'],cfg,seed)
                    train_cell(bundle,cfg,head,source,seed,options,pcas[source],dest,fingerprint,device,train_only=pilot)
    print('SUITE FINISHED',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/sparse_extension.json');p.add_argument('--output',default='results/sparse-extension')
    p.add_argument('--device',choices=('cpu','mps'),default='mps');p.add_argument('--dataset');p.add_argument('--head');p.add_argument('--pilot',action='store_true')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,torch.device(a.device),a.dataset,a.head,a.pilot)

if __name__=='__main__':main()
