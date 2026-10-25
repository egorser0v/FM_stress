#!/usr/bin/env python3
"""Independent NumPy metric/data audit plus CPU checkpoint replay for capacity study.

Uses independently implemented definitions from check_kernel_scan, not the
experiment's sampling/PCA/scoring functions. Model architecture itself is reused
for checkpoint replay. Existing partial result sets can be checked honestly.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import check_kernel_scan as audit
from fm_stress.models import create_model
from fm_stress.experiment import CoordinateModel

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',default='results/followup-capacity');p.add_argument('--replay-n',type=int,default=1024);p.add_argument('--threads',type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads)
 root=Path(a.results);cfg=audit.read(root/'config.json');manifest=audit.read(root/'manifest.json');L=audit.Ledger();rows=[];cache={};paired={};train_paired={};pca_checked=set();observed=set()
 expected_ids={(k,rep['data_seed'],source,head,budget) for k,heads in cfg['kernel_heads'].items() for rep in cfg['replicates'] for source in cfg['sources'] for head in heads for budget in cfg['budgets']}
 expected=len(expected_ids)
 L.true('manifest',audit.stable_hash(cfg)==manifest['config_sha256'],'config hash')
 L.true('manifest',audit.stable_hash({k:v for k,v in manifest.items() if k!='fingerprint'})==manifest['fingerprint'],'manifest fingerprint')
 for name,digest in manifest['source_hashes'].items():
  archived_protocol=root/'frozen_protocol.md'
  source_path=archived_protocol if name=='docs/followup_capacity_protocol.md' and archived_protocol.exists() else ROOT/name
  L.true('manifest',audit.sha(source_path)==digest,'source hash '+name)
 # Only the human-readable protocol may have a documented post-run correction.
 # Its original bytes remain pinned by the manifest and checked above.
 protocol_provenance={'frozen_sha256':manifest['source_hashes']['docs/followup_capacity_protocol.md'],'public_sha256':audit.sha(ROOT/'docs/followup_capacity_protocol.md'),'frozen_copy':str(root/'frozen_protocol.md')}
 for rp in sorted(root.glob('*/*/draw*/runs/*.json')):
  r=audit.read(rp);label=str(rp.relative_to(root));identity=(r['kernel'],r['data_seed'],r['source'],r['head'],r['budget']);L.true(label,identity in expected_ids and identity not in observed,'unique expected identity');observed.add(identity);g=r['geometry'];spec={**g,'threshold':g['normalization_threshold']};ds=r['data_seed'];source=r['source'];key=(r['kernel'],ds)
  if key not in cache:
   train=audit.windows(spec,cfg['n_train'],ds);test=audit.windows(spec,cfg['n_test'],ds+2);cache[key]=(train,test)
  train,test=cache[key];gx=rp.parent.parent;ckpath=gx/'checkpoints'/(rp.stem+'.pt');samp=gx/'samples'/(rp.stem+'.npz')
  L.true(label,audit.sha(ckpath)==r['checkpoint_sha256'],'checkpoint hash');L.true(label,audit.sha(samp)==r['samples_sha256'],'samples hash');L.true(label,r['fingerprint']==manifest['fingerprint'],'fingerprint')
  z=np.load(samp);x,u,t,e=audit.pairs(spec,test.target,source,ds,'test');georecord=audit.read(gx/'geometry.json');q=np.array(georecord['sources'][source]['pca']['components']);qt=np.array(georecord['target_pca']['components'])
  if (key,source) not in pca_checked:
   for basis,vals,raw,labelbasis in [(q,np.array(georecord['sources'][source]['pca']['eigenvalues']),train.target-audit.source_draw(spec,len(train.target),source,ds+100),'velocity'),(qt,np.array(georecord['target_pca']['eigenvalues']),train.target,'target')]:
    centered=raw-raw.mean(0);cov=centered.T@centered/(len(raw)-1)
    L.close(label,'train_'+labelbasis+'_orthogonal',basis@basis.T,np.eye(len(basis)),1e-7,1e-9)
    L.close(label,'train_'+labelbasis+'_PCA',basis@cov@basis.T,np.diag(vals),1e-6,1e-8)
   pca_checked.add((key,source))
  ee=audit.source_draw(spec,len(z['ensemble'])*cfg['ensemble_size'],source,ds+1000).reshape(z['ensemble'].shape)
  hashes={k:audit.array_sha(v) for k,v in [('raw_history',test.raw_context),('raw_target',test.raw_target),('source',e),('times',t),('ensemble_source',ee)]}
  L.true(label,hashes==r['test_input_hashes'],'independent rebuilt test-input hashes')
  pe=np.mean(((z['prediction']-u)@q.T)**2,axis=0)
  L.close(label,'pc_mse',pe,r['metrics']['pc_mse'],1e-8,1e-10);L.close(label,'pc1',pe[0],r['metrics']['pc1_mse'],1e-8);L.close(label,'residual',pe[2:].mean(),r['metrics']['residual_mse'],1e-8)
  L.close(label,'target_basis_pc_mse',np.mean(((z['prediction']-u)@qt.T)**2,axis=0),r['fixed_target_basis_metrics']['pc_mse'],1e-8,1e-10)
  L.close(label,'roughness_generated',audit.roughness(z['generated']),r['metrics']['generated_roughness'],1e-8);L.close(label,'roughness_target',audit.roughness(test.target[:len(z['generated'])]),r['metrics']['target_roughness'],1e-8)
  scaled=pe/np.maximum(np.array(georecord['sources'][source]['pca']['eigenvalues']),georecord['sources'][source]['pca']['floor'])
  L.close(label,'normalized_PC_error',scaled,r['metrics']['pc_normalized_mse'],1e-7,1e-9)
  for name,errors in [('success',pe),('normalized_success_diagnostic',scaled)]:
   flags=audit.literal_rule(errors[0],errors[2:].mean(),audit.roughness(z['generated']),audit.roughness(test.target[:len(z['generated'])]))
   observed_flags=tuple(r['metrics'][name][k] for k in ['velocity_rule_pass','roughness_rule_pass','works'])
   L.true(label,flags==observed_flags,'literal rule '+name)
  scores=audit.forecast_scores(z['ensemble'],test.target[:len(z['ensemble'])])
  for name,sc in scores.items():L.close(label,'forecast_'+name,sc,r['per_context'][name],1e-6,1e-9);L.close(label,'forecast_mean_'+name,sc.mean(),r['forecast_means'][name],1e-6,1e-9)
  a_spec=r['architecture'];model=CoordinateModel(create_model(a_spec['family'],horizon=g['horizon'],lookback=g['lookback'],**{**cfg['model'],'width':a_spec['width'],'depth':a_spec['depth']})).eval()
  ck=torch.load(ckpath,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);N=min(a.replay_n,len(x));pred=[]
  with torch.no_grad():
   for st in range(0,N,128):
    sl=slice(st,min(st+128,N));pred.append(model(torch.tensor(x[sl],dtype=torch.float32),torch.tensor(t[sl],dtype=torch.float32),torch.tensor(test.context[sl],dtype=torch.float32)).numpy())
  pred=np.concatenate(pred);rms=float(np.sqrt(np.mean((pred-z['prediction'][:N])**2)));scale=float(np.sqrt(np.mean(z['prediction'][:N]**2)))
  L.true(label,rms<=1e-4*max(scale,1e-3),'CPU replay RMS relative threshold');L.true(label,sum(p.numel() for p in model.parameters())==r['parameters'],'parameter count')
  with torch.no_grad():
   paths=torch.tensor(e[:2],dtype=torch.float32);history=torch.tensor(test.context[:2],dtype=torch.float32)
   for step in range(cfg['euler_steps']):paths=paths+model(paths,torch.full((len(paths),),step/cfg['euler_steps']),history)/cfg['euler_steps']
  path_rms=float(np.sqrt(np.mean((paths.numpy()-z['generated'][:2])**2)))
  path_scale=float(np.sqrt(np.mean(z['generated'][:2]**2)))
  L.true(label,path_rms<=2e-4*max(path_scale,1e-3),'CPU Euler replay RMS relative threshold')
  tr=r['training'];eligible=[c for c in tr['curve'] if c['step']<=r['budget']];best=min(eligible,key=lambda c:c['validation_mse'])
  L.true(label,r['best_step']==best['step']==ck['step'],'prefix checkpoint selected minimum');L.close(label,'best_val',best['validation_mse'],r['best_validation_mse'],1e-12)
  trainkey=(r['kernel'],ds);trainidentity={'encoder':tr['encoder_initial_sha256'],'batches':tr['batch_hashes']}
  if trainkey in train_paired:L.true(label,train_paired[trainkey]==trainidentity,'cross-source training random-number pairing')
  else:train_paired[trainkey]=trainidentity
  pairkey=(r['kernel'],ds,source);identity={'encoder':tr['encoder_initial_sha256'],'batches':tr['batch_hashes'],'inputs':r['test_input_hashes']}
  if pairkey in paired:L.true(label,paired[pairkey]==identity,'head/source RNG pairing')
  else:paired[pairkey]=identity
  null=r['output_null_diagnostic']
  if null['applicable']:
   W=model.base.output_projection.weight.detach().double().numpy();b=model.base.output_projection.bias.detach().double().numpy();Q,S,_=np.linalg.svd(W,full_matrices=True);rank=np.linalg.matrix_rank(W);Q=Q[:,rank:];f=W.shape[0]
   L.true(label,rank==null['output_weight_rank'],'output rank');L.close(label,'null_projection',Q@Q.T,null['projection'],1e-7,1e-9)
   forced=float(np.square((u-b)@Q).sum(-1).mean()/f);L.close(label,'null_forced_error',forced,null['forced_velocity_mse_per_coordinate'],1e-7,1e-10)
   diff=(z['generated']-e[:len(z['generated'])]-b)@Q
   L.true(label,np.max(np.abs(diff),initial=0)<2e-5,'Euler nullspace invariant')
  rows.append({'run':label,'cpu_replay_n':N,'cpu_replay_rms':rms,'cpu_replay_relative_rms':rms/max(scale,1e-3),'cpu_euler_n':2,'cpu_euler_rms':path_rms})
  print('VERIFIED',len(rows),label,flush=True)
 result={'protocol_provenance':protocol_provenance,'complete':observed==expected_ids and len(rows)==expected and not L.failures,'expected_records':expected,'verified_records':len(rows),'missing_identities':[list(t) for t in sorted(expected_ids-observed)],'checks':L.checks,'failures':L.failures,'max_relative_deviations':L.max_dev,'records':rows,'scope':'Independent data/score formulas; CPU checkpoint replay first N pairs. No retraining. Partial sets remain complete=false.'}
 (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['complete','expected_records','verified_records','checks','failures']}))
 if L.failures:raise SystemExit(1)
if __name__=='__main__':main()
