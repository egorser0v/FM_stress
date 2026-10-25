#!/usr/bin/env python3
"""Frozen post-hoc full-ensemble Euler audit for all 24 label-study checkpoints."""
from pathlib import Path
import json,hashlib,sys,time,platform
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fm_stress.data import GPConfig,sample_source
from fm_stress.models import create_model
from fm_stress.experiment import CoordinateModel,generate,tensor,save_json,config_hash
from fm_stress.ensemble_evaluation import context_scores
IN=ROOT/'results/followup-label-noise';OUT=ROOT/'results/followup-label-solver'
CONFIG={'status':'post-hoc numerical integration sensitivity, no training or checkpoint selection',
        'steps':[64,128],'contexts':256,'members':32,'batch_size':1024,'device':'mps','torch_threads':2,
        'replay_relative_rms_tolerance':1e-5,'independent_score_tolerance':1e-9,
        'design':'All 24 existing checkpoints; same full ensemble initial states, histories and targets as the frozen label study.'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(x):return float(np.sqrt(np.square(x).mean()))
def score_audit(samples,target,scores):
 s=samples.shape[1];vals=[]
 for a,y in zip(samples.astype(float),target.astype(float)):
  vals.append(np.abs(a-y).mean()-np.abs(a[:,None]-a[None,:]).sum()/(2*s*(s-1)*a.shape[1]))
 err=float(np.max(np.abs(np.array(vals)-scores['fair_crps'])))
 order=np.sort(samples.astype(float),axis=1);bounds=[]
 for p in [.05,.95]:
  position=p*(s-1);i=int(np.floor(position));fraction=position-i
  bounds.append((1-fraction)*order[:,i]+fraction*order[:,min(i+1,s-1)])
 coverage=((target>=bounds[0])&(target<=bounds[1])).mean(1)
 coverage_error=float(np.max(np.abs(coverage-scores['coverage_90'])))
 assert max(err,coverage_error)<CONFIG['independent_score_tolerance']
 return {'direct_pairwise_crps_max_abs_error':err,'independent_coverage_max_abs_error':coverage_error}
def main():
 started=time.time();torch.set_num_threads(2)
 if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS is required; no CPU fallback.')
 cfg=json.loads((IN/'config.json').read_text());g=GPConfig(**cfg['data']);device=torch.device('mps')
 audit=json.loads((IN/'independent_verification.json').read_text())
 if not audit.get('complete') or audit.get('status')!='passed available runs' or len(audit.get('records',[]))!=24:raise RuntimeError('Full independent training-result audit must pass before solver study.')
 paths=sorted(IN.glob('dataset_*/*/runs/*.json'));assert len(paths)==24
 files=[Path(__file__),ROOT/'docs/followup_label_solver_protocol.md',IN/'config.json',IN/'manifest.json',IN/'independent_verification.json',ROOT/'fm_stress/data.py',ROOT/'fm_stress/models.py',ROOT/'fm_stress/experiment.py',ROOT/'fm_stress/ensemble_evaluation.py'];records=[]
 for p in paths:
  r=json.loads(p.read_text());group=p.parent.parent;name=p.stem
  checkpoint=group/'checkpoints'/f'{name}.pt';samples=group/'samples'/f'{name}.npz'
  assert sha(checkpoint)==r['checkpoint_sha256'] and sha(samples)==r['samples_sha256']
  files.extend([p,checkpoint,samples,group/'test.npz']);records.append((r,group,name,checkpoint,samples))
 manifest={'config':CONFIG,'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},'frozen_before_solver_outcomes':True}
 manifest['fingerprint']=config_hash(manifest);OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
 save_json(OUT/'manifest.json',manifest);save_json(OUT/'config.json',CONFIG)
 save_json(OUT/'environment.json',{'device':str(device),'mps_available':torch.backends.mps.is_available(),'torch':torch.__version__,'platform':platform.platform(),'threads':torch.get_num_threads()})
 results=[]
 for r,group,name,checkpoint,samples in records:
  key=f"{r['data_seed']}_{r['source']}_{name}";path=OUT/f'{key}.json';arraypath=OUT/f'{key}.npz'
  if path.exists():
   previous=json.loads(path.read_text());assert previous['fingerprint']==manifest['fingerprint'] and sha(arraypath)==previous['arrays_sha256'];results.append(previous);continue
  begin=time.time();a=np.load(samples);test=np.load(group/'test.npz');nc,ns=CONFIG['contexts'],CONFIG['members']
  eps=a['ensemble_source'];target=test['target'][:nc];history=test['history'][:nc]
  rebuilt=sample_source(g,nc*ns,r['source'],r['data_seed']+3000);assert np.array_equal(eps,rebuilt)
  model=CoordinateModel(create_model(r['head'],horizon=g.horizon,lookback=g.lookback,**cfg['model'])).to(device)
  state=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(state['state_dict']);model.eval()
  ee=tensor(eps,device);hh=tensor(np.repeat(history,ns,axis=0),device)
  start64=time.time();base=generate(model,ee,hh,64,CONFIG['batch_size']).reshape(nc,ns,g.horizon);seconds64=time.time()-start64
  relative=rms(base-a['ensemble'])/max(rms(a['ensemble']),1e-12);assert relative<CONFIG['replay_relative_rms_tolerance']
  start128=time.time();fine=generate(model,ee,hh,128,CONFIG['batch_size']).reshape(nc,ns,g.horizon);seconds128=time.time()-start128
  scores64=context_scores(base,target);scores128=context_scores(fine,target)
  original_error=max(abs(float(v.mean())-r['forecast_means'][k]) for k,v in scores64.items());assert abs(float(scores64['fair_crps'].mean())-r['forecast_means']['fair_crps'])<1e-5
  numerical={'steps64':score_audit(base,target,scores64),'steps128':score_audit(fine,target,scores128)}
  np.savez_compressed(arraypath,ensemble128=fine,initial=eps,history=history,target=target)
  got={k:r[k] for k in ['data_seed','seed','source','head','label','best_step']}
  got.update({'fingerprint':manifest['fingerprint'],'device':str(device),'checkpoint_sha256':sha(checkpoint),'original_samples_sha256':sha(samples),'arrays_sha256':sha(arraypath),
              'replay64_relative_rms':relative,'replay64_original_score_max_abs_error':original_error,'endpoint64_128_rms':rms(fine-base),
              'means64':{k:float(v.mean()) for k,v in scores64.items()},'means128':{k:float(v.mean()) for k,v in scores128.items()},
              'per_context64':{k:v.tolist() for k,v in scores64.items()},'per_context128':{k:v.tolist() for k,v in scores128.items()},
              'score_change':{k:float(scores128[k].mean()-scores64[k].mean()) for k in scores64},'independent_score_checks':numerical,
              'seconds64':seconds64,'seconds128':seconds128,'seconds_total':time.time()-begin})
  save_json(path,got);results.append(got);print('COMPLETE',key,'seconds',round(got['seconds_total'],2),'CRPS change',got['score_change']['fair_crps'],flush=True)
 for p,h in manifest['inputs_sha256'].items():assert sha(ROOT/p)==h
 save_json(OUT/'summary.json',{'complete':len(results)==24,'expected':24,'observed':len(results),'rows':results,'fingerprint':manifest['fingerprint'],'frozen_inputs_unchanged':True,'seconds':time.time()-started})
 print('ALL COMPLETE',time.time()-started,flush=True)
if __name__=='__main__':main()
