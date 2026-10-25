import json
from pathlib import Path
import numpy as np
import torch
from fm_stress.data import GPConfig,kernel_covariance,sample_dataset,source_covariance
from fm_stress.followup_label_noise import prepare_group,manifest,folder,build,paired_inputs,oracle_for,ah


def small_config():
 c=json.loads(Path('configs/followup_label_noise.json').read_text())
 c.update(n_train=64,n_val=32,n_test=32,steps=4,batch_size=8,eval_every=2,precompute_chunk=16,n_jacobian=4,n_paths=8,n_ensemble_contexts=8,ensemble_size=4)
 c['replicates']=[{'data_seed':9917041,'seed':997}]
 return c


def test_stream_pairing_and_oracle_computation(tmp_path):
 c=small_config();fp=manifest(c,tmp_path);ds=c['replicates'][0]['data_seed'];g=GPConfig(**c['data'])
 for s in ['white','gp']:prepare_group(c,tmp_path,ds,s,fp)
 rw=folder(tmp_path,ds,'white');rg=folder(tmp_path,ds,'gp')
 np.testing.assert_array_equal(np.load(rw/'index.npy'),np.load(rg/'index.npy'))
 np.testing.assert_array_equal(np.load(rw/'time.npy'),np.load(rg/'time.npy'))
 tr=sample_dataset(g,c['n_train'],ds);idx=np.load(rg/'index.npy');t=np.load(rg/'time.npy')
 epsw=tr.target[idx]-np.load(rw/'stochastic.npy');epsg=tr.target[idx]-np.load(rg/'stochastic.npy')
 np.testing.assert_allclose(epsw@np.linalg.cholesky(source_covariance(g,'gp')).T,epsg,atol=2e-6)
 np.testing.assert_allclose(np.load(rg/'x.npy'),t[:,None]*tr.target[idx]+(1-t[:,None])*epsg,atol=1e-6)
 got=oracle_for(g,'gp').flow(np.load(rg/'x.npy').astype(float),t.astype(float),tr.context[idx],order=128,jacobian=False)['velocity']
 # Saved inputs are rounded after oracle computation; GP ill-conditioning can
 # amplify tiny rounding, so use a tolerance appropriate to float32 storage.
 np.testing.assert_allclose(np.load(rg/'oracle.npy'),got,atol=2e-4,rtol=2e-4)
 assert np.mean((np.load(rg/'oracle.npy')-np.load(rg/'stochastic.npy'))**2)>0


def test_initialization_paired_across_labels_and_heads():
 c=small_config();g=GPConfig(**c['data']);dev=torch.device('cpu')
 a=build(c,g,'mlp',997,dev);b=build(c,g,'mlp',997,dev);s=build(c,g,'s4',997,dev)
 assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
 assert all(torch.equal(x,y) for x,y in zip(a.base.encoder.parameters(),s.base.encoder.parameters()))


def test_conditional_mean_label_noise_identity():
 # Fresh full Gaussian windows independently test noise covariance calibration.
 g=GPConfig();d=sample_dataset(g,4096,99417041)
 for source in ['white','gp']:
  x,u,t,_=paired_inputs(g,d,source,99417841);o=oracle_for(g,source)
  vals=[];variances=[]
  for i in range(0,len(x),512):
   r=o.flow(x[i:i+512],t[i:i+512],d.context[i:i+512],order=64,jacobian=False,noise_covariance=True)
   vals.extend(np.square(u[i:i+512]-r['velocity']).mean(1))
   variances.extend(np.trace(r['noise_covariance'],axis1=1,axis2=2)/g.horizon)
  delta=np.array(vals)-variances
  assert abs(delta.mean())<5*delta.std(ddof=1)/np.sqrt(len(delta))
