import copy,json
from pathlib import Path
import numpy as np
import torch
from fm_stress import followup_capacity as fc
from fm_stress import kernel_scan as ks
from fm_stress.models import count_parameters

def cfg():return json.loads(Path('configs/followup_capacity.json').read_text())
def test_parameter_and_encoder_controls():
 c=cfg();g=ks.make_geometry(c,'rbf','F64_l8'); states=[]
 expected={'mlp_original':73960,'mlp_rank_clear':73953,'mlp_wide':114328,'s4':74038}
 for name,n in expected.items():
  m=fc.build_model(c,g,name,61,torch.device('cpu'))
  assert count_parameters(m)==n
  states.append(torch.cat([p.flatten() for p in m.base.encoder.parameters()]))
 assert all(torch.equal(states[0],s) for s in states)

def test_exact_null_identity():
 c=cfg();g=ks.make_geometry(c,'rbf','F64_l8');m=fc.build_model(c,g,'mlp_original',61,torch.device('cpu'))
 with torch.no_grad():
  torch.nn.init.normal_(m.base.output_projection.weight)
  torch.nn.init.normal_(m.base.output_projection.bias)
 x=np.random.default_rng(1).normal(size=(4,64)).astype('float32');h=np.zeros((4,128),'float32')
 pred=ks.predict(m,torch.tensor(x),torch.ones(4)*.3,torch.tensor(h)).numpy()
 gen=ks.generate(m,torch.tensor(x),torch.tensor(h),4,4)
 r=fc.output_null_diagnostic(m,np.zeros_like(x),np.eye(64),np.zeros_like(x),x,gen,pred)
 assert r['null_dimension']==8
 assert abs(r['source_null_variance_per_coordinate']-.125)<1e-12
 assert r['prediction_null_identity_max_abs']<1e-5
 assert r['generated_null_identity_max_abs']<1e-5

def test_prefix_selection_and_pairing():
 assert fc.prefix_eligible(6000,6000)
 assert not fc.prefix_eligible(6500,6000)
 c=cfg();g=ks.make_geometry(c,'rbf','F64_l8');ds=ks.sample_windows(g,8,907041)
 a=ks.fixed_pairs(g,ds,'white',907041,'val');b=ks.fixed_pairs(g,ds,'gp',907041,'val')
 assert np.array_equal(a[2],b[2])
 assert not np.array_equal(a[3],b[3])
 assert np.allclose(a[3]@np.linalg.cholesky(ks.source_covariance(g,'gp')).T,b[3])

def test_cpu_smoke_both_prefixes(tmp_path):
 c=cfg();c.update(n_train=128,n_val=32,n_test=16,n_paths=8,n_ensemble_contexts=2,ensemble_size=2,steps=4,budgets=[2,4],eval_every=2,batch_size=8,euler_steps=4,generation_batch_size=8)
 c['replicates']=[{'data_seed':907041,'seed':71}]
 fp=fc.prepare(c,tmp_path);bundle=ks.prepare_group(c,tmp_path,'rbf','F64_l8',907041,fp)
 records=fc.train_one(c,bundle,tmp_path,'rbf',907041,71,'white','mlp_original',torch.device('cpu'),fp)
 assert len(records)==2
 assert records[0]['best_step']<=2 and records[1]['best_step']<=4
 assert all(r['total_training_steps']==4 for r in records)
 assert records[0]['test_input_hashes']==records[1]['test_input_hashes']
 assert records[0]['training']['batch_hashes']==records[1]['training']['batch_hashes']
 assert fc.train_one(c,bundle,tmp_path,'rbf',907041,71,'white','mlp_original',torch.device('cpu'),fp)==records
