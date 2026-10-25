#!/usr/bin/env python3
"""Independent matrix-derivation and artifact audit of the matched oracle study."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.linalg import null_space
from scipy.integrate import quad_vec
from scipy.special import gammaln
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fm_stress.data import GPConfig,kernel_covariance,source_covariance,sample_dataset
from fm_stress.normalized_history_oracle import NormalizedHistoryOracle
from fm_stress.experiment import save_json


def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def main():
 out=ROOT/'results/followup-oracle';summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
 cfg=json.loads((ROOT/'results/main/config.json').read_text());geo=json.loads((ROOT/'results/main/geometry.json').read_text());g=GPConfig(**cfg['data'])
 K=kernel_covariance(g);L=g.lookback;F=g.horizon
 # Independent Gaussian construction: U=B'H and D=Y-mean(H).
 B=null_space(np.ones((1,L)))
 Ch=B.T@K[:L,:L]@B
 Cross=K[L:,:L]@B-np.ones((F,1))@(np.ones((1,L))@K[:L,:L]@B/L)
 vmean=K[:L,:L].sum()/L**2
 crossmean=K[L:,:L].mean(1)
 Cd=K[L:,L:]+vmean-np.outer(crossmean,np.ones(F))-np.outer(np.ones(F),crossmean)
 M=np.linalg.solve(Ch,Cross.T).T
 V=Cd-M@Cross.T; V=(V+V.T)/2
 data=sample_dataset(g,4096,8811041);z=data.context;u=z@B
 means=u@M.T;rates=np.einsum('ni,ij,nj->n',u,np.linalg.inv(Ch),u)
 o=NormalizedHistoryOracle(K,L,np.eye(F));p=o.posterior(z)
 checks={'centered_basis_mean_max_abs':float(np.max(np.abs(means-p.mean))),
         'centered_basis_rate_relative_rms':rms(rates-p.rate_twice)/rms(p.rate_twice),
         'centered_basis_covariance_relative_rms':rms(V-o.V)/rms(o.V)}
 assert checks['centered_basis_mean_max_abs']<3e-7
 assert checks['centered_basis_rate_relative_rms']<1e-7
 assert checks['centered_basis_covariance_relative_rms']<1e-7
 # Predictive-law calibration using independently sampled complete raw GP windows.
 # Mean Mahalanobis residual/a must be F/(L-3) under the conditional t law.
 e=data.target-means
 stat=np.einsum('ni,ij,nj->n',e,np.linalg.inv(V),e)/rates
 expected=F/(L-3)
 checks['fresh_raw_gp_calibration']={'sample_mean':float(stat.mean()),'theory':expected,'standard_error':float(stat.std(ddof=1)/np.sqrt(len(stat))),
                                      'z_score':float((stat.mean()-expected)/(stat.std(ddof=1)/np.sqrt(len(stat))))}
 assert abs(checks['fresh_raw_gp_calibration']['z_score'])<5
 probes=[]; support=[]
 for source in ['white','gp']:
  arr=np.load(out/f'oracle_{source}.npz');cs=source_covariance(g,source);o=NormalizedHistoryOracle(K,L,cs)
  h=arr['history'];p=o.posterior(h)
  Q=np.asarray(geo['pca'][source]['components']);_,tcov=o.target_moments(h)
  for i,t in enumerate(arr['times']):
   x=(1-t)*arr['source']+t*arr['target']
   cov=t*t*tcov+(1-t)**2*cs
   inv=np.linalg.inv(cov);resinv=np.einsum('ij,njk,ik->ni',Q[2:],inv,Q[2:]).sum(1)
   mahal=np.sqrt(.1**2/(F-2)*resinv)
   support.append({'source':source,'time':float(t),'perturbation_norm':.1,'direction':'uniform sphere in velocity-PC3..F',
                   'rms_mahalanobis_radius_mean':float(mahal.mean()),'p10':float(np.quantile(mahal,.1)),'p90':float(np.quantile(mahal,.9)),
                   'typical_unconditional_radius_sqrtF':float(np.sqrt(F)),
                   'interpretation':'Geometry diagnostic: conditional covariance radius; mixture is not Gaussian. Does not prove every direction is off support.'})
  # Independent adaptive scalar integration of a Gaussian mixture, without
  # generalized eigenvectors or Gauss-Laguerre weights. Three histories at t=.5.
  for n in range(3):
   t=.5;x=.5*(arr['source'][n]+arr['target'][n]);mu=p.mean[n];a=p.rate_twice[n]
   def direct(q,shift=0):
    if q<=0:return np.zeros(F+1)
    cy=o.V*a/(2*q);S=t*t*cy+(1-t)**2*cs;U=t*cy-(1-t)*cs
    diff=x-t*mu;sol=np.linalg.solve(S,diff)
    logf=(o.shape-1)*np.log(q)-q-gammaln(o.shape)-.5*(np.linalg.slogdet(S)[1]+diff@sol+F*np.log(2*np.pi))
    velocity=mu+U@sol
    return np.concatenate(([1.],velocity))*np.exp(logf-shift)
   q0=o.shape
   cy=o.V*a/(2*q0);S=t*t*cy+(1-t)**2*cs;diff=x-t*mu
   shift=(o.shape-1)*np.log(q0)-q0-gammaln(o.shape)-.5*(np.linalg.slogdet(S)[1]+diff@np.linalg.solve(S,diff)+F*np.log(2*np.pi))
   value,err=quad_vec(lambda q:direct(q,shift),0,np.inf,epsabs=1e-9,epsrel=1e-9)
   val=value[1:]/value[0];got=o.flow(x[None],t,h[n:n+1],order=128)['velocity'][0]
   rel=rms(val-got)/max(rms(got),1e-12)
   probes.append({'source':source,'history_index':n,'relative_velocity_rms':rel,'absolute_integrator_error':float(err)})
   assert rel<3e-7
 checks['independent_adaptive_integral']=probes
 # Recompute every saved model discrepancy using arrays, avoiding study helper.
 count=0;maxerr=0.
 for r in summary['rows']:
  a=np.load(out/f"model_{r['source']}_{r['kind']}_seed{r['seed']}.npz")
  ref=np.load(out/f"oracle_{r['source']}.npz")
  i=summary['config']['times'].index(r['time']);Q=np.asarray(geo['pca'][r['source']]['components'])
  J=a['jacobian'][i];A=(Q@J@Q.T)[:,2:,2:]
  for kind in ['matched','privileged']:
   R=(Q@ref[f'{kind}_jacobian'][i]@Q.T)[:,2:,2:]
   v=float(np.mean(np.linalg.norm(A-R,axis=(1,2))/np.linalg.norm(R,axis=(1,2))))
   err=abs(v-r[kind]['residual']['mean_relative_error']);maxerr=max(maxerr,err);assert err<1e-10;count+=1
 checks['replayed_residual_jacobian_rows']=count;checks['max_summary_replay_error']=maxerr
 checks['source_hashes_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==s for p,s in manifest['sha256'].items())
 assert checks['source_hashes_unchanged']
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*')) if p.suffix in ['.json','.npz'] and p.name!='independent_verification.json'}
 save_json(out/'independent_verification.json',{'status':'passed','checks':checks,'perturbation_support':support,'artifact_sha256':hashes})
 print(json.dumps({'status':'passed','checks':checks,'perturbation_support':support},indent=2))

if __name__=='__main__':main()
