"""Paired ordinary-CFM versus information-matched conditional-mean supervision."""
from __future__ import annotations
import argparse,copy,hashlib,json,time,platform,sys
from pathlib import Path
import numpy as np
import torch
from numpy.lib.format import open_memmap
from .data import GPConfig,kernel_covariance,source_covariance,sample_dataset,sample_source
from .normalized_history_oracle import NormalizedHistoryOracle
from .experiment import CoordinateModel,config_hash,save_json,load_config,tensor,predict,generate
from .models import create_model,optimizer_groups,count_parameters
from .metrics import PCA,fit_pca,spectrum,evaluate,velocity_metrics
from .ensemble_evaluation import context_scores
ROOT=Path(__file__).resolve().parents[1]
VERIFIED_GROUPS=set()
FILES=['fm_stress/followup_label_noise.py','fm_stress/normalized_history_oracle.py','fm_stress/data.py','fm_stress/models.py','fm_stress/metrics.py','fm_stress/experiment.py','fm_stress/ensemble_evaluation.py','docs/followup_label_noise_protocol.md']

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rms(a):return float(np.sqrt(np.square(a).mean()))
def folder(out,ds,source):return Path(out)/f'dataset_{ds}'/source

def manifest(cfg,out):
 p=Path(out)/'manifest.json';p.parent.mkdir(parents=True,exist_ok=True)
 m={'config_sha256':config_hash(cfg),'sources':{f:sha(ROOT/f) for f in FILES}}
 fp=config_hash(m);m['fingerprint']=fp
 if p.exists() and load_config(p)!=m:raise ValueError('Changed source/config; use a separate output')
 save_json(p,m);save_json(Path(out)/'config.json',cfg);return fp

def oracle_for(g,source):return NormalizedHistoryOracle(kernel_covariance(g),g.lookback,source_covariance(g,source),g.normalization_threshold)

def paired_inputs(g,data,source,seed):
 e=sample_source(g,len(data),source,seed)
 t=np.random.default_rng(seed+1).uniform(0,1,len(data))
 return (1-t[:,None])*e+t[:,None]*data.target,data.target-e,t,e

def quadrature_check(o,x,t,h,cfg):
 a=np.concatenate([o.flow(x[i:i+1024],t[i:i+1024],h[i:i+1024],order=cfg['quadrature_order'],jacobian=False)['velocity'] for i in range(0,len(x),1024)])
 b=np.concatenate([o.flow(x[i:i+1024],t[i:i+1024],h[i:i+1024],order=cfg['quadrature_check_order'],jacobian=False)['velocity'] for i in range(0,len(x),1024)])
 d={'relative_rms':rms(a-b)/max(rms(b),1e-12),'max_abs':float(np.max(np.abs(a-b)))}
 if d['relative_rms']>1e-5:raise ValueError('Oracle quadrature not converged: '+str(d))
 return d

def prepare_group(cfg,out,ds,source,fp):
 root=folder(out,ds,source);done=root/'prepared.json'
 if done.exists():
  old=load_config(done)
  if old['fingerprint']!=fp:raise ValueError('Wrong prepared fingerprint')
  for f,d in old['files'].items():
   if sha(root/f)!=d:raise ValueError('Prepared artifact changed: '+f)
  print('PREPARED_REUSE',ds,source,flush=True);return old
 started=time.perf_counter();root.mkdir(parents=True,exist_ok=True)
 g=GPConfig(**cfg['data']);train=sample_dataset(g,cfg['n_train'],ds);o=oracle_for(g,source)
 pca=fit_pca(train.target-sample_source(g,len(train),source,ds+100))
 target_pca=fit_pca(train.target)
 save_json(root/'pca.json',pca.to_dict());save_json(root/'target_pca.json',target_pca.to_dict())
 np.savez_compressed(root/'train.npz',history=train.context.astype('float32'),target=train.target.astype('float32'))
 total=cfg['steps']*cfg['batch_size'];F=g.horizon
 arrays={k:open_memmap(root/(k+'.npy'),mode='w+',dtype='int32' if k=='index' else 'float32',shape=(total,) if k in ['index','time'] else (total,F)) for k in ['index','time','x','stochastic','oracle']}
 ri=np.random.default_rng(ds+2000);rz=np.random.default_rng(ds+2001);rt=np.random.default_rng(ds+2002)
 chol=np.linalg.cholesky(source_covariance(g,source));checks=[];rounding_checks=[];qtail=0.
 for start in range(0,total,cfg['precompute_chunk']):
  n=min(cfg['precompute_chunk'],total-start);end=start+n
  ix=ri.integers(len(train),size=n,dtype=np.int32);e=rz.normal(size=(n,F))@chol.T;t=rt.uniform(0,1,n)
  x=(1-t[:,None])*e+t[:,None]*train.target[ix];u=train.target[ix]-e
  got=o.flow(x,t,train.context[ix],order=cfg['quadrature_order'],jacobian=False)
  qtail=max(qtail,float(got['threshold_tail_probability'].max()))
  for key,val in [('index',ix),('time',t),('x',x),('stochastic',u),('oracle',got['velocity'])]:arrays[key][start:end]=val
  if start==0 or end==total:
   checks.append(quadrature_check(o,x[:128],t[:128],train.context[ix[:128]],cfg))
   xx=x[:128].astype('float32').astype(float);tt=t[:128].astype('float32').astype(float);hh=train.context[ix[:128]].astype('float32').astype(float)
   rounded=o.flow(xx,tt,hh,order=cfg['quadrature_check_order'],jacobian=False)['velocity']
   stored=got['velocity'][:128].astype('float32').astype(float)
   rounding_checks.append({'relative_rms':rms(rounded-stored)/max(rms(stored),1e-12),'max_abs':float(np.max(np.abs(rounded-stored))),'residual_pc_rms':rms((rounded-stored)@pca.components[2:].T),'tail_pc7_16_rms':rms((rounded-stored)@pca.components[6:].T)})
   if rounding_checks[-1]['relative_rms']>1e-4:raise ValueError('Float32 teacher rounding too large')
  if start%(cfg['precompute_chunk']*64)==0:print('PRECOMPUTE',ds,source,end,total,flush=True)
 for a in arrays.values():a.flush()
 del arrays
 split_checks={};geometry={}
 for split,offset in [('val',1),('test',2)]:
  data=sample_dataset(g,cfg['n_'+split],ds+offset);x,u,t,e=paired_inputs(g,data,source,ds+800+100*(split=='test'))
  parts=[o.flow(x[i:i+1024],t[i:i+1024],data.context[i:i+1024],order=cfg['quadrature_order'],jacobian=False,noise_covariance=True) for i in range(0,len(x),1024)]
  got={k:np.concatenate([r[k] for r in parts]) for k in ['velocity','noise_covariance']}
  split_checks[split]=quadrature_check(o,x,t,data.context,cfg)
  np.savez_compressed(root/f'{split}.npz',history=data.context,target=data.target,x=x,stochastic=u,oracle=got['velocity'],time=t,source=e,noise_covariance=got['noise_covariance'])
  geometry[split]={'velocity':spectrum(u,pca),'target':spectrum(data.target,target_pca)}
  if split=='test':
   n=cfg['n_jacobian'];h=data.context[:n];y=data.target[:n];eps=e[:n]
   _,tcov=o.target_moments(h);xs=[];js=[];cxs=[]
   for tt in [.1,.5,.9]:
    xx=(1-tt)*eps+tt*y;r=o.flow(xx,tt,h,order=cfg['quadrature_check_order'])
    xs.append(xx);js.append(r['jacobian']);cxs.append(tt*tt*tcov+(1-tt)**2*o.C)
   np.savez_compressed(root/'jacobian_reference.npz',history=h,x=np.array(xs),time=np.array([.1,.5,.9]),jacobian=np.array(js),input_covariance=np.array(cxs))
 if source=='gp' and geometry['val']['velocity']['top2_energy']<=.9:raise ValueError('Original geometry gate failed')
 save_json(root/'geometry.json',geometry)
 record={'fingerprint':fp,'data_seed':ds,'source':source,'complete':True,'training_pairs':total,
         'stream_quadrature_checks':checks,'float32_teacher_rounding_checks':rounding_checks,'split_quadrature_checks':split_checks,'max_threshold_tail':qtail,
         'seconds':time.perf_counter()-started,'files':{p.name:sha(p) for p in root.iterdir() if p.suffix in ['.npy','.npz','.json'] and p.name!='prepared.json'}}
 save_json(done,record);print('PREPARED',ds,source,round(record['seconds'],1),flush=True);return record

def pca_load(path):
 d=load_config(path);return PCA(np.array(d['mean']),np.array(d['components']),np.array(d['eigenvalues']),d['floor'])

def build(cfg,g,head,seed,device):
 torch.manual_seed(seed)
 if device.type=='mps':torch.mps.manual_seed(seed)
 m=create_model(head,horizon=g.horizon,lookback=g.lookback,**cfg['model'])
 torch.manual_seed(10000+seed)
 for sub in m.encoder.modules():
  if hasattr(sub,'reset_parameters'):sub.reset_parameters()
 return CoordinateModel(m).to(device)

def jacobian_score(model,root,pca):
 ref=np.load(root/'jacobian_reference.npz');probe=copy.deepcopy(model).cpu().eval();rows=[];saved=[];Q=pca.components
 for i,t in enumerate(ref['time']):
  x=tensor(ref['x'][i],'cpu').requires_grad_(True);h=tensor(ref['history'],'cpu');tt=torch.full((len(x),),float(t))
  v=probe(x,tt,h)
  J=torch.stack([torch.autograd.grad(v[:,j].sum(),x,retain_graph=True)[0] for j in range(v.shape[1])],1).detach().numpy()
  saved.append(J);A=Q@J@Q.T;B=Q@ref['jacobian'][i]@Q.T;C=Q@ref['input_covariance'][i]@Q.T
  r={'time':float(t)}
  for block,sl in [('full',slice(None)),('residual',slice(2,None))]:
   d=(A-B)[:,sl,sl];b=B[:,sl,sl];c=C[:,sl,sl]
   en=float(np.einsum('nij,njk,nik->n',d,c,d).mean());bn=float(np.einsum('nij,njk,nik->n',b,c,b).mean())
   r[block]={'mean_relative_error':float(np.mean(np.linalg.norm(d,axis=(1,2))/np.maximum(np.linalg.norm(b,axis=(1,2)),1e-12))),
             'covariance_weighted_relative_error':float(np.sqrt(max(en,0)/max(bn,1e-24))), 'error_energy':en,'reference_energy':bn}
  rows.append(r)
 return rows,np.array(saved)

def evaluate_model(cfg,root,model,ds,source,head,label,seed,best,step,checkpoint,fp,training,device):
 g=GPConfig(**cfg['data']);pca=pca_load(root/'pca.json');a=np.load(root/'test.npz')
 x,u,t,h=[a[k] for k in ['x','stochastic','time','history']]
 pred=predict(model,tensor(x,device),tensor(t,device),tensor(h,device)).cpu().numpy()
 n=cfg['n_paths'];generated=generate(model,tensor(a['source'][:n],device),tensor(h[:n],device),cfg['euler_steps'],cfg['generation_batch_size'])
 nc,ns=cfg['n_ensemble_contexts'],cfg['ensemble_size'];es=sample_source(g,nc*ns,source,ds+3000)
 eh=np.repeat(h[:nc],ns,axis=0)
 ens=generate(model,tensor(es,device),tensor(eh,device),cfg['euler_steps'],cfg['generation_batch_size']).reshape(nc,ns,g.horizon)
 scores=context_scores(ens,a['target'][:nc]);metrics=evaluate(pred,u,generated,a['target'][:n],pca)
 jac,js=jacobian_score(model,root,pca)
 extra={};solver={}
 if ds==cfg['replicates'][0]['data_seed']:
  nf=8;fine=generate(model,tensor(es[:nf*ns],device),tensor(eh[:nf*ns],device),128,cfg['generation_batch_size']).reshape(nf,ns,g.horizon)
  sf=context_scores(fine,a['target'][:nf]);sb=context_scores(ens[:nf],a['target'][:nf]);extra['ensemble128']=fine
  solver={'contexts':nf,'endpoint_rmse':rms(fine-ens[:nf]),'score_change':{k:float(sf[k].mean()-sb[k].mean()) for k in sf}}
 name=f'{head}_{label}_seed{seed}';sample=root/'samples'/f'{name}.npz';sample.parent.mkdir(exist_ok=True)
 np.savez_compressed(sample,prediction=pred,generated=generated,ensemble=ens,ensemble_source=es,jacobian=js,**extra)
 decomposition=velocity_metrics(pred,u,pca);noise=pca.components@a['noise_covariance']@pca.components.T
 record={'complete':True,'data_seed':ds,'seed':seed,'source':source,'head':head,'label':label,'fingerprint':fp,
         'parameters':count_parameters(model),'best_validation_oracle_mse':best,'best_step':step,'device':str(device),
         'metrics':metrics,'oracle_discrepancy':velocity_metrics(pred,a['oracle'],pca),
         'oracle_label_mse':velocity_metrics(a['oracle'],u,pca),
         'expected_noise_per_pc':np.diagonal(noise,axis1=1,axis2=2).mean(0).tolist(),
         'forecast_means':{k:float(v.mean()) for k,v in scores.items()},'per_context':{k:v.tolist() for k,v in scores.items()},
         'jacobian':jac,'solver_sensitivity':solver,'training':training,'checkpoint_sha256':sha(checkpoint),'samples_sha256':sha(sample)}
 save_json(root/'runs'/f'{name}.json',record)
 print('COMPLETE',ds,source,head,label,'residual',metrics['residual_mse'],'CRPS',record['forecast_means']['fair_crps'],flush=True)
 return record

def train_run(cfg,out,ds,seed,source,head,label,fp,device):
 root=folder(out,ds,source);prep=load_config(root/'prepared.json')
 if prep['fingerprint']!=fp:raise ValueError('Preparation fingerprint mismatch')
 if (str(root),fp) not in VERIFIED_GROUPS:
  for file,digest in prep['files'].items():
   if sha(root/file)!=digest:raise ValueError('Changed prepared artifact: '+file)
  VERIFIED_GROUPS.add((str(root),fp))
 if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS requested but unavailable; no CPU fallback')
 if cfg.get('required_device') and cfg['required_device']!=device.type:raise ValueError('Device does not match frozen study configuration')
 name=f'{head}_{label}_seed{seed}';rp=root/'runs'/f'{name}.json';cp=root/'checkpoints'/f'{name}.pt'
 if rp.exists():
  old=load_config(rp)
  if old['fingerprint']!=fp or old['checkpoint_sha256']!=sha(cp) or old['samples_sha256']!=sha(root/'samples'/f'{name}.npz'):raise ValueError('Changed completed run')
  return old
 started=time.perf_counter();g=GPConfig(**cfg['data']);m=build(cfg,g,head,seed,device)
 encoder_sha=ah(np.concatenate([p.detach().cpu().numpy().ravel() for p in m.base.encoder.parameters()]))
 opt=torch.optim.AdamW(optimizer_groups(m,cfg['learning_rate'],cfg['weight_decay']))
 train=np.load(root/'train.npz');history=tensor(train['history'],device)
 stream={k:np.load(root/(k+'.npy'),mmap_mode='r') for k in ['index','time','x',label]}
 val=np.load(root/'val.npz');vx,vt,vh,vy=[tensor(val[k],device) for k in ['x','time','history','oracle']]
 best=float('inf');best_step=0;curve=[];cp.parent.mkdir(exist_ok=True);bs=cfg['batch_size']
 for step in range(1,cfg['steps']+1):
  sl=slice((step-1)*bs,step*bs);ix=torch.as_tensor(np.array(stream['index'][sl],dtype=np.int64),device=device)
  x=tensor(np.array(stream['x'][sl]),device);t=tensor(np.array(stream['time'][sl]),device);u=tensor(np.array(stream[label][sl]),device)
  m.train();opt.zero_grad(set_to_none=True);loss=(m(x,t,history[ix])-u).square().mean()
  if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
  loss.backward()
  torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['clip_grad'],error_if_nonfinite=True);opt.step()
  if step%cfg['eval_every']==0 or step==cfg['steps']:
   m.eval()
   with torch.no_grad():vl=float((predict(m,vx,vt,vh)-vy).square().mean().cpu())
   if not np.isfinite(vl):raise FloatingPointError('Nonfinite validation loss')
   curve.append({'step':step,'training_loss':float(loss.detach().cpu()),'validation_oracle_mse':vl})
   if vl<best:
    best=vl;best_step=step;torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'fingerprint':fp,'step':step,'validation':vl},cp)
 m.load_state_dict(torch.load(cp,map_location=device,weights_only=True)['state_dict']);m.eval()
 training={'seconds':time.perf_counter()-started,'curve':curve,'encoder_initialization_sha256':encoder_sha,
           'stream_files_sha256':{k:prep['files'][k+'.npy'] for k in ['index','time','x','stochastic','oracle']}}
 return evaluate_model(cfg,root,m,ds,source,head,label,seed,best,best_step,cp,fp,training,device)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/followup_label_noise.json');p.add_argument('--output',default='results/followup-label-noise');p.add_argument('--prepare-only',action='store_true');p.add_argument('--device',default='mps');p.add_argument('--data-seed',type=int);p.add_argument('--source');p.add_argument('--head');p.add_argument('--label');p.add_argument('--limit',type=int)
 a=p.parse_args();cfg=load_config(a.config);torch.set_num_threads(2);fp=manifest(cfg,a.output);count=0
 if not a.prepare_only:
  device=torch.device(a.device)
  if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS requested but unavailable; no CPU fallback')
  if cfg.get('required_device') and device.type!=cfg['required_device']:raise ValueError('Device does not match frozen configuration')
  save_json(Path(a.output)/('environment_'+device.type+'.json'),{'device':str(device),'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),'mps_available':torch.backends.mps.is_available(),'mps_built':torch.backends.mps.is_built(),'torch_threads':torch.get_num_threads(),'dtype':'float32 training, float64 analytic oracle','config_sha256':config_hash(cfg)})
 for rep in cfg['replicates']:
  ds,seed=rep['data_seed'],rep['seed']
  if a.data_seed is not None and ds!=a.data_seed:continue
  for source in cfg['sources']:
   if a.source and source!=a.source:continue
   if a.prepare_only:prepare_group(cfg,a.output,ds,source,fp);continue
   if not (folder(a.output,ds,source)/'prepared.json').exists():raise ValueError('Run --prepare-only before training')
   for head in cfg['heads']:
    if a.head and head!=a.head:continue
    for label in cfg['labels']:
     if a.label and label!=a.label:continue
     train_run(cfg,a.output,ds,seed,source,head,label,fp,torch.device(a.device));count+=1
     if a.limit is not None and count>=a.limit:return

if __name__=='__main__':main()
