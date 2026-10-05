"""Prospective exploratory identity-encoder, temporal-coordinate objective control."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .data import GPConfig, sample_dataset, sample_source, source_covariance
from .experiment import config_hash, fixed_pairs, generate, load_config, predict, save_json, tensor
from .metrics import fit_pca, spectrum, evaluate, velocity_metrics
from .models import create_model, count_parameters, optimizer_groups

HEADS = ('mlp', 's4')
LOSSES = ('raw', 'balanced')
SOURCE_FILES = ('fm_stress/controlled_followup.py', 'fm_stress/models.py',
                'fm_stress/data.py', 'fm_stress/metrics.py', 'fm_stress/experiment.py')


def hashes():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def make_model(cfg, head, seed, device):
    torch.manual_seed(seed)
    base = create_model(head, horizon=cfg['data']['horizon'], lookback=cfg['data']['lookback'], **cfg['model'])
    if cfg['model']['encoder_dim'] != cfg['data']['lookback']:
        raise ValueError('Identity history requires encoder_dim == lookback')
    base.encoder = torch.nn.Identity()
    return base.to(device)


def objective(error, loss, q, scales):
    if loss == 'raw':
        return error.square().mean()
    if loss == 'balanced':
        return ((error @ q)/scales).square().mean()
    raise ValueError(loss)


def prepare(cfg, out):
    out = Path(out)
    if (out/'config.json').exists() and load_config(out/'config.json') != cfg:
        raise ValueError('Output configuration mismatch')
    manifest = hashes()
    if (out/'source_manifest.json').exists() and load_config(out/'source_manifest.json') != manifest:
        raise ValueError('Source changed; use a fresh output directory')
    save_json(out/'config.json', cfg)
    save_json(out/'source_manifest.json', manifest)
    gp = GPConfig(**cfg['data'])
    # Deliberately do not generate or inspect test data during geometry/pilots.
    data = {s: sample_dataset(gp, cfg['n_'+s], cfg['data_seed']+i)
            for i,s in enumerate(('train','val'))}
    eps = sample_source(gp, cfg['n_train'], 'gp', cfg['data_seed']+101)
    pca = fit_pca(data['train'].target-eps)
    val_eps = sample_source(gp, cfg['n_val'], 'gp', cfg['data_seed']+111)
    vs = spectrum(data['val'].target-val_eps,pca)
    ts = spectrum(data['val'].target,fit_pca(data['train'].target))
    gate = vs['top2_energy']>.9 and ts['mean_within_patch_std']>.02 and ts['roughness']>.002
    save_json(out/'geometry.json', {'gate_pass':bool(gate), 'pca':{'gp':pca.to_dict()},
             'validation_gp_velocity':vs, 'validation_target':ts,
             'test_inspected':False, 'config_sha256':config_hash(cfg)})
    if not gate:
        raise ValueError('Validation geometry gate failed')
    return gp,data,pca


def pilot_name(head, loss, lr):
    return f'{head}_{loss}_lr{lr:g}'


def choose_rates(cfg, out):
    out=Path(out)
    records=[]
    for head in HEADS:
        for loss in LOSSES:
            for lr in cfg['pilot_learning_rates']:
                path=out/'pilots'/'runs'/(pilot_name(head,loss,lr)+'.json')
                if not path.exists():
                    raise ValueError('All validation-only pilots must complete before selection/finals')
                r=load_config(path)
                if r['status']!='complete' or r['config_sha256']!=config_hash(cfg) or 'metrics' in r:
                    raise ValueError('Invalid or test-evaluated pilot')
                records.append(r)
    chosen={}
    for head in HEADS:
        for loss in LOSSES:
            candidates=[r for r in records if r['head']==head and r['loss']==loss]
            best=min(candidates,key=lambda r:(r['best_validation_objective'],r['learning_rate']))
            chosen[head+'_'+loss]={'learning_rate':best['learning_rate'],
                                 'pilot_validation_objective':best['best_validation_objective'],
                                 'pilot_run':best['run']}
    selection={'config_sha256':config_hash(cfg),'selection_uses':'validation only; separate objective per cell',
               'all_pilots_complete':True,'selected':chosen,
               'pilot_record_sha256':{r['run']:hashlib.sha256((out/'pilots'/'runs'/(r['run']+'.json')).read_bytes()).hexdigest() for r in records}}
    if (out/'selection.json').exists() and load_config(out/'selection.json') != selection:
        raise ValueError('Refusing to alter selection after it has been locked')
    save_json(out/'selection.json',selection)
    return selection


def train_run(cfg,gp,data,pca,out,head,loss,seed,lr,steps,phase,device):
    out=Path(out)
    if phase=='final':
        selected=choose_rates(cfg,out)
        if lr!=selected['selected'][head+'_'+loss]['learning_rate']:
            raise ValueError('Final learning rate differs from locked selection')
    elif phase!='pilot':
        raise ValueError(phase)
    name=pilot_name(head,loss,lr) if phase=='pilot' else f'gp_{head}_{loss}_seed{seed}'
    dest=out/'pilots' if phase=='pilot' else out
    record_path=dest/'runs'/(name+'.json')
    if record_path.exists():
        r=load_config(record_path)
        if r['config_sha256']!=config_hash(cfg) or r['status']!='complete':
            raise ValueError('Cannot reuse mismatched run')
        print('SKIP',name,flush=True)
        return r
    start=time.perf_counter()
    model=make_model(cfg,head,seed,device)
    optimizer=torch.optim.AdamW(optimizer_groups(model,lr,cfg['weight_decay']))
    q,scales=tensor(pca.components.T,device),tensor(pca.scales,device)
    y,h=tensor(data['train'].target,device),tensor(data['train'].context,device)
    chol=tensor(np.linalg.cholesky(source_covariance(gp,'gp')),device)
    vx,vu,vt,_=fixed_pairs(cfg,gp,data['val'],'gp','val')
    vx,vu,vt,vh=[tensor(a,device) for a in (vx,vu,vt,data['val'].context)]
    torch.manual_seed(20000+seed)
    if device.type=='mps':torch.mps.manual_seed(20000+seed)
    curve=[];best=float('inf');best_step=0
    checkpoint=dest/'checkpoints'/(name+'.pt');checkpoint.parent.mkdir(parents=True,exist_ok=True)
    print('START',phase,name,'seed',seed,'params',count_parameters(model),'device',device,flush=True)
    for step in range(1,steps+1):
        model.train();ix=torch.randint(len(y),(cfg['batch_size'],),device=device)
        yy,hh=y[ix],h[ix]
        e=torch.randn(len(ix),gp.horizon,device=device)@chol.T
        t=torch.rand(len(ix),device=device);x=(1-t[:,None])*e+t[:,None]*yy;u=yy-e
        optimizer.zero_grad(set_to_none=True)
        train_loss=objective(model(x,t,hh)-u,loss,q,scales)
        train_loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad']);optimizer.step()
        if step%cfg['eval_every']==0 or step==steps:
            model.eval()
            with torch.no_grad():
                err=predict(model,vx,vt,vh)-vu
                val=objective(err,loss,q,scales).item();raw=err.square().mean().item()
            if not np.isfinite(val):raise FloatingPointError(name)
            curve.append({'step':step,'train_objective':train_loss.item(),'validation_objective':val,
                          'validation_raw_mse':raw,'seconds':time.perf_counter()-start})
            if val<best:
                best=val;best_step=step
                torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            'config':cfg,'head':head,'loss':loss,'seed':seed,'step':step,
                            'learning_rate':lr,'pca':pca.to_dict(),'encoder':'identity'},checkpoint)
            save_json(dest/'training'/(name+'.json'),curve)
            print(name,step,'val',round(val,6),'raw',round(raw,6),flush=True)
    record={'run':name,'source':'gp','head':head,'loss':loss,'seed':seed,'phase':phase,
            'status':'complete','config_sha256':config_hash(cfg),'learning_rate':lr,'steps':steps,
            'parameters':count_parameters(model),'device':str(device),'encoder':'identity',
            'best_step':best_step,'best_validation_objective':best,'curve':curve}
    if phase=='final':
        # This is the first test-data access, after ALL pilots and locked selection.
        test=sample_dataset(gp,cfg['n_test'],cfg['data_seed']+2)
        tx,tu,tt,teps=fixed_pairs(cfg,gp,test,'gp','test')
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['state_dict']);model.eval()
        pred=predict(model,tensor(tx,device),tensor(tt,device),tensor(test.context,device)).cpu().numpy()
        n=min(cfg['n_paths'],len(test))
        gen=generate(model,tensor(teps[:n],device),tensor(test.context[:n],device),cfg['euler_steps'])
        record['metrics']=evaluate(pred,tu,gen,test.target[:n],pca)
        bins=[]
        for i in range(5):
            keep=(tt>=i/5)&(tt<(i+1)/5)
            if keep.any():bins.append({'lower':i/5,'upper':(i+1)/5,'n':int(keep.sum()),**velocity_metrics(pred[keep],tu[keep],pca)})
        record['time_bin_metrics']=bins
        samples=dest/'samples';samples.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(samples/(name+'.npz'),generated=gen,target=test.target[:n],history=test.context[:n],
                            prediction=pred,velocity=tu,time=tt,source=teps[:n])
    record['seconds']=time.perf_counter()-start
    save_json(record_path,record)
    print('DONE',name,'seconds',round(record['seconds'],1),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/controlled_followup.json');p.add_argument('--output',default='results/controlled_followup')
    p.add_argument('--phase',choices=('pilot','final','all'),default='all');p.add_argument('--device',choices=('cpu','mps'),default='mps')
    a=p.parse_args();cfg=load_config(a.config);device=torch.device(a.device)
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
    torch.set_num_threads(4)
    gp,data,pca=prepare(cfg,a.output)
    save_json(Path(a.output)/'environment.json',{'device':a.device,'torch':torch.__version__,'platform':platform.platform()})
    if a.phase in ('pilot','all'):
        for head in HEADS:
            for loss in LOSSES:
                for lr in cfg['pilot_learning_rates']:
                    train_run(cfg,gp,data,pca,a.output,head,loss,cfg['pilot_seed'],lr,cfg['pilot_steps'],'pilot',device)
    if a.phase in ('final','all'):
        selection=choose_rates(cfg,a.output)
        for seed in cfg['seeds']:
            for head in HEADS:
                for loss in LOSSES:
                    lr=selection['selected'][head+'_'+loss]['learning_rate']
                    train_run(cfg,gp,data,pca,a.output,head,loss,seed,lr,cfg['steps'],'final',device)

if __name__=='__main__':main()
