#!/usr/bin/env python3
"""Independent input/RNG/checkpoint replay and predictive Bayes baselines.

Oracle-label replay calls the separately independently derived/audited oracle;
input construction, score arithmetic and checkpoint selection are rebuilt here.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fm_stress.data import GPConfig,sample_dataset,kernel_covariance,source_covariance,conditional_target
from fm_stress.normalized_history_oracle import NormalizedHistoryOracle
from fm_stress.models import create_model
from fm_stress.experiment import CoordinateModel,load_config,save_json,predict,tensor,generate
from fm_stress.followup_label_noise import sha,folder
from fm_stress.ensemble_evaluation import context_scores

def rms(a):return float(np.sqrt(np.mean(np.square(a))))
def maxabs(a):return float(np.max(np.abs(a)))
def direct_crps(ens,target):
 S=ens.shape[1];F=ens.shape[2]
 return np.array([np.abs(x-y).mean()-np.abs(x[:,None]-x[None,:]).sum()/(2*S*(S-1)*F) for x,y in zip(ens,target)])

def audit_streams(cfg,root,fp):
 records=[];g=GPConfig(**cfg['data'])
 for rep in cfg['replicates']:
  ds=rep['data_seed'];train=sample_dataset(g,cfg['n_train'],ds)
  for src in cfg['sources']:
   start=time.perf_counter();p=folder(root,ds,src);prep=load_config(p/'prepared.json')
   assert prep['fingerprint']==fp
   for f,d in prep['files'].items():assert sha(p/f)==d,f
   tr=np.load(p/'train.npz');assert np.array_equal(tr['history'],train.context.astype('float32'));assert np.array_equal(tr['target'],train.target.astype('float32'))
   arrays={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ['index','time','x','stochastic','oracle']}
   ri=np.random.default_rng(ds+2000);rz=np.random.default_rng(ds+2001);rt=np.random.default_rng(ds+2002)
   chol=np.linalg.cholesky(source_covariance(g,src));o=NormalizedHistoryOracle(kernel_covariance(g),g.lookback,chol@chol.T,g.normalization_threshold)
   total=cfg['steps']*cfg['batch_size'];worst=0.;checks=0
   for i in range(0,total,cfg['precompute_chunk']):
    n=min(cfg['precompute_chunk'],total-i);sl=slice(i,i+n)
    ix=ri.integers(len(train),size=n,dtype=np.int32);z=rz.standard_normal((n,g.horizon));eps=z@chol.T;t=rt.random(n)
    x=(1-t[:,None])*eps+t[:,None]*train.target[ix];u=train.target[ix]-eps
    for key,expected in [('index',ix),('time',t),('x',x),('stochastic',u)]:assert np.array_equal(arrays[key][sl],expected.astype(arrays[key].dtype)),(src,key,i)
    v=o.flow(x,t,train.context[ix],order=cfg['quadrature_order'],jacobian=False)['velocity'].astype('float32')
    err=maxabs(v-arrays['oracle'][sl]);worst=max(worst,err);assert err<2e-5,(src,i,err);checks+=n
   splitchecks=[]
   for split,offset in [('val',1),('test',2)]:
    a=np.load(p/(split+'.npz'));data=sample_dataset(g,cfg['n_'+split],ds+offset);seed=ds+800+100*(split=='test')
    eps=np.random.default_rng(seed).standard_normal((len(data),g.horizon))@chol.T;t=np.random.default_rng(seed+1).random(len(data))
    x=(1-t[:,None])*eps+t[:,None]*data.target;u=data.target-eps
    for key,v in [('history',data.context),('target',data.target),('source',eps),('time',t),('x',x),('stochastic',u)]:assert np.array_equal(a[key],v),(split,key)
    vx=[];cv=[]
    for i in range(0,len(data),512):
     v=o.flow(x[i:i+512],t[i:i+512],data.context[i:i+512],order=cfg['quadrature_check_order'],jacobian=False,noise_covariance=True)
     vx.append(v['velocity']);cv.append(v['noise_covariance'])
    vx=np.concatenate(vx);cv=np.concatenate(cv);rel=rms(vx-a['oracle'])/max(rms(vx),1e-12);nrel=rms(cv-a['noise_covariance'])/max(rms(cv),1e-12)
    assert rel<1e-5 and nrel<1e-5
    splitchecks.append({'split':split,'velocity_relative_rms':rel,'noise_covariance_relative_rms':nrel})
   records.append({'data_seed':ds,'source':src,'replayed_training_pairs':checks,'max_stored_teacher_abs_error':worst,'splits':splitchecks,'seconds':time.perf_counter()-start,'prepared_sha256':sha(p/'prepared.json')})
   print('STREAM_AUDITED',ds,src,checks,flush=True)
  w=load_config(folder(root,ds,'white')/'prepared.json')['files'];gfiles=load_config(folder(root,ds,'gp')/'prepared.json')['files']
  assert w['index.npy']==gfiles['index.npy'];assert w['time.npy']==gfiles['time.npy']
 result={'complete':len(records)==len(cfg['replicates'])*len(cfg['sources']),'fingerprint':fp,'records':records,'teacher_replay_dependency':'normalized_history_oracle.py, independently checked by centered-basis derivation, direct adaptive mixture integration and Monte Carlo','status':'passed'}
 save_json(root/'stream_verification.json',result);return result

def audit_threshold_cases(cfg,root):
 g=GPConfig(**cfg['data']);records=[]
 for rep in cfg['replicates']:
  ds=rep['data_seed'];train=sample_dataset(g,cfg['n_train'],ds)
  probe=NormalizedHistoryOracle(kernel_covariance(g),g.lookback,np.eye(g.horizon),g.normalization_threshold)
  posterior=probe.posterior(train.context);rare=np.flatnonzero(posterior.omitted_scale_probability>1e-12)
  for src in cfg['sources']:
   group=folder(root,ds,src);idx=np.load(group/'index.npy',mmap_mode='r');positions=np.flatnonzero(np.isin(idx,rare))
   r={'data_seed':ds,'source':src,'rare_history_count':len(rare),'affected_training_pairs':len(positions)}
   if len(positions):
    x=np.load(group/'x.npy',mmap_mode='r')[positions].astype(float);t=np.load(group/'time.npy',mmap_mode='r')[positions].astype(float);h=train.context[idx[positions]]
    o=NormalizedHistoryOracle(kernel_covariance(g),g.lookback,source_covariance(g,src),g.normalization_threshold)
    v=[o.flow(x,t,h,order=q,jacobian=False)['velocity'] for q in [64,128,256]]
    r['64_vs_128_relative_rms']=rms(v[0]-v[1])/max(rms(v[1]),1e-12)
    r['128_vs_256_relative_rms']=rms(v[1]-v[2])/max(rms(v[2]),1e-12)
    r['64_vs_128_max_abs']=maxabs(v[0]-v[1]);assert r['64_vs_128_relative_rms']<1e-5
   records.append(r)
 result={'status':'passed','records':records};save_json(root/'threshold_verification.json',result);return result


def baselines(cfg,root):
 g=GPConfig(**cfg['data']);rows=[]
 for rep in cfg['replicates']:
  ds=rep['data_seed'];test=sample_dataset(g,cfg['n_test'],ds+2);N=cfg['n_ensemble_contexts'];S=64;seed=ds+44000
  o=NormalizedHistoryOracle(kernel_covariance(g),g.lookback,np.eye(g.horizon),g.normalization_threshold)
  matched=o.sample_target(test.context[:N],S,seed)
  mu,cov=conditional_target(g,test);z=np.random.default_rng(seed+1).normal(size=(N,S,g.horizon));priv=mu[:N,None]+np.einsum('nsj,nij->nsi',z,np.linalg.cholesky(cov[:N]))
  path=root/f'dataset_{ds}'/'predictive_baselines.npz';np.savez_compressed(path,matched=matched,privileged=priv,target=test.target[:N])
  rr={'data_seed':ds,'contexts':N,'ensemble_size':S,'matched_seed':seed,'privileged_seed':seed+1,'samples_sha256':sha(path),'sources':'same future law for both sources; sampled directly, no Euler error','scores':{}}
  for name,ens in [('matched',matched),('privileged',priv)]:
   score=context_scores(ens,test.target[:N]);direct=direct_crps(ens,test.target[:N]);assert maxabs(direct-score['fair_crps'])<1e-10
   rr['scores'][name]={'mean':{k:float(v.mean()) for k,v in score.items()},'per_context':{k:v.tolist() for k,v in score.items()},'direct_crps_max_error':maxabs(direct-score['fair_crps'])}
  rows.append(rr)
 result={'rows':rows,'status':'passed direct CRPS arithmetic','note':'Matched baseline conditions on full normalized history; privileged baseline additionally uses raw history location/scale. Tiny encoders may discard information. 64-member fair scores vs 32-member network scores have different Monte Carlo variance but same population estimand.'}
 save_json(root/'predictive_baselines.json',result);print('BASELINES',len(rows),flush=True);return result

def audit_models(cfg,root,fp):
 g=GPConfig(**cfg['data']);rows=[];missing=[]
 for rep in cfg['replicates']:
  ds,seed=rep['data_seed'],rep['seed']
  for src in cfg['sources']:
   group=folder(root,ds,src);test=np.load(group/'test.npz');val=np.load(group/'val.npz');Q=np.array(load_config(group/'pca.json')['components'])
   for head in cfg['heads']:
    for label in cfg['labels']:
     name=f'{head}_{label}_seed{seed}';rp=group/'runs'/f'{name}.json';cp=group/'checkpoints'/f'{name}.pt'
     if not rp.exists():missing.append(str(rp));continue
     r=load_config(rp);ck=torch.load(cp,map_location='cpu',weights_only=True);saved=np.load(group/'samples'/f'{name}.npz')
     if cfg.get('required_device'):assert r['device']==cfg['required_device']
     assert r['fingerprint']==fp and ck['fingerprint']==fp
     assert sha(cp)==r['checkpoint_sha256'] and sha(group/'samples'/f'{name}.npz')==r['samples_sha256']
     curve=r['training']['curve'];best=min(curve,key=lambda x:x['validation_oracle_mse'])
     assert best['step']==ck['step']==r['best_step'];assert best['validation_oracle_mse']==ck['validation']==r['best_validation_oracle_mse']
     m=CoordinateModel(create_model(head,horizon=g.horizon,lookback=g.lookback,**cfg['model']));m.load_state_dict(ck['state_dict']);m.eval()
     pp=predict(m,tensor(test['x'],'cpu'),tensor(test['time'],'cpu'),tensor(test['history'],'cpu')).numpy()
     vv=predict(m,tensor(val['x'],'cpu'),tensor(val['time'],'cpu'),tensor(val['history'],'cpu')).numpy()
     # Replay float32 validation objective, as used for checkpoint selection.
     vl=float(np.mean(np.square(vv-val['oracle'].astype('float32')),dtype=np.float32))
     prel=rms(pp-saved['prediction'])/max(rms(pp),1e-12);vrel=abs(vl-ck['validation'])/max(abs(vl),1e-12)
     assert prel<1e-4 and vrel<1e-4,(name,prel,vrel)
     pc=np.square((pp-test['stochastic'])@Q.T).mean(0);assert np.max(abs(pc-r['metrics']['pc_mse']))<3e-4
     n=8;gen=generate(m,tensor(test['source'][:n],'cpu'),tensor(test['history'][:n],'cpu'),cfg['euler_steps'],batch_size=32)
     grel=rms(gen-saved['generated'][:n])/max(rms(gen),1e-12);assert grel<2e-4,(name,grel)
     jacref=np.load(group/'jacobian_reference.npz');jerrs=[]
     for i,t in enumerate(jacref['time']):
      x=tensor(jacref['x'][i,:4],'cpu').requires_grad_(True);v=m(x,torch.full((len(x),),float(t)),tensor(jacref['history'][:4],'cpu'))
      J=torch.stack([torch.autograd.grad(v[:,j].sum(),x,retain_graph=True)[0] for j in range(g.horizon)],1).detach().numpy()
      jerr=rms(J-saved['jacobian'][i,:4])/max(rms(J),1e-12);assert jerr<1e-5;jerrs.append(jerr)
     for i,reported in enumerate(r['jacobian']):
      A=Q@saved['jacobian'][i]@Q.T;B=Q@jacref['jacobian'][i]@Q.T;C=Q@jacref['input_covariance'][i]@Q.T
      for block,sl in [('full',slice(None)),('residual',slice(2,None))]:
       D=(A-B)[:,sl,sl];R=B[:,sl,sl];Cov=C[:,sl,sl]
       v=float(np.sqrt(max(float(np.einsum('nij,njk,nik->n',D,Cov,D).mean()),0)/max(float(np.einsum('nij,njk,nik->n',R,Cov,R).mean()),1e-24)))
       assert abs(v-reported[block]['covariance_weighted_relative_error'])<1e-10
     rows.append({'data_seed':ds,'source':src,'head':head,'label':label,'checkpoint_selection_passed':True,'prediction_relative_rms':prel,'validation_loss_relative_error':vrel,'bounded_Euler_replay_relative_rms':grel,'jacobian_relative_rms':jerrs})
     print('MODEL_AUDITED',ds,src,head,label,flush=True)
 result={'complete':len(rows)==len(cfg['replicates'])*len(cfg['sources'])*len(cfg['heads'])*len(cfg['labels']),'records':rows,'missing':missing,'fingerprint':fp,'status':'passed available runs'}
 save_json(root/'independent_verification.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='results/followup-label-noise');p.add_argument('--streams-only',action='store_true');p.add_argument('--baselines-only',action='store_true');p.add_argument('--skip-streams',action='store_true');a=p.parse_args();root=Path(a.input);cfg=load_config(root/'config.json');fp=load_config(root/'manifest.json')['fingerprint'];torch.set_num_threads(2)
 if a.baselines_only:baselines(cfg,root);return
 if not a.skip_streams:audit_streams(cfg,root,fp)
 if a.streams_only:return
 baselines(cfg,root);audit_threshold_cases(cfg,root);result=audit_models(cfg,root,fp)
 print(json.dumps({'complete':result['complete'],'models':len(result['records']),'missing':len(result['missing'])},indent=2))
if __name__=='__main__':main()
