#!/usr/bin/env python3
"""Independent metric replay and presentation for finite perturbation diagnostics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fm_stress.experiment import save_json
IN=ROOT/'results/followup-oracle/finite_perturbation';OUT=ROOT/'output/followup-oracle/finite_perturbation'
def main():
 OUT.mkdir(parents=True,exist_ok=True);s=json.loads((IN/'results.json').read_text());m=json.loads((IN/'manifest.json').read_text());g=json.loads((ROOT/'results/main/geometry.json').read_text())
 for f,h in m['sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
 maxerr=0
 for r in s['rows']:
  src=r['source'];R=np.asarray(g['pca'][src]['components'])[2:];ref=np.load(IN/f'reference_{src}_t{r["time"]}_s{r["scale"]}.npz');pred=np.load(IN/f'model_{src}_{r["kind"]}_seed{r["seed"]}.npz')[f'change_t{r["time"]}_s{r["scale"]}']
  a=pred@R.T;b=ref['oracle_change']@R.T;err=float(np.sqrt(np.mean((a-b)**2)));den=float(np.sqrt(np.mean(b**2)))
  maxerr=max(maxerr,abs(err-r['residual']['absolute_rms_error']),abs(den-r['residual']['reference_rms']),abs(err/den-r['residual']['relative_rms_error']))
 assert maxerr<1e-12
 save_json(IN/'independent_arithmetic_audit.json',{'status':'passed','rows':len(s['rows']),'max_metric_difference':maxerr,'all_frozen_inputs_unchanged':True,'limitation':'Replays saved finite changes and independent arithmetic, not an independent oracle implementation; oracle derivation and quadrature are separately audited.'})
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10.8,4.6));colors={'mlp':'#236c9b','s4':'#df733b','whitened_mlp':'#198577'}
 for ax,src in zip(axs,['white','gp']):
  kinds=['mlp','s4'] if src=='white' else ['mlp','s4','whitened_mlp']
  for j,kind in enumerate(kinds):
   for i,scale in enumerate([.01,.1]):
    r=next(r for r in s['summary'] if r['source']==src and r['kind']==kind and r['scale']==scale)
    vals=list(r['residual_relative_error_per_seed'].values());x=j+(i-.5)*.3
    ax.bar(x,np.mean(vals),width=.27,color=colors[kind],alpha=1 if i==0 else .45)
    ax.scatter(np.full(5,x),vals,s=10,color='#172733',zorder=5)
  ax.set_xticks(range(len(kinds)),['MLP','S4'] if src=='white' else ['MLP','S4','Whitened MLP']);ax.set_title(('White' if src=='white' else 'GP')+' source',loc='left');ax.set_ylabel('Relative RMS finite-change error');ax.set_ylim(0,.78);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Actual local velocity changes under covariance-scaled perturbations',fontsize=13,ha='left',x=.03)
 fig.text(.03,.01,'Dark bars: scale 0.01. Light bars: 0.1. Dots: five seeds. Lower is better. Residual input and output PCs.',fontsize=10)
 fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(OUT/'finite_change_error.png',dpi=180);plt.close(fig)
 table=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [r['source'],r['kind'],r['scale'],f'{r["residual"]["relative_rms_error"]:.4f}',f'{r["residual"]["absolute_rms_error"]:.6f}',f'{r["residual"]["reference_rms"]:.6f}'])+'</tr>' for r in s['summary'])
 html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Finite perturbation follow-up</title><style>body{font:17px/1.55 system-ui;max-width:1100px;margin:42px auto;padding:0 24px;color:#22333d}h1{font-size:32px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:nth-child(-n+2){text-align:left}</style><h1>Does the derivative ranking describe actual local changes?</h1><p>The 25 original checkpoints were evaluated without retraining. For each source, all heads share 256 histories, five interpolation times, four random directions per history, and two displacement scales. Directions lie in velocity PCs 3–16 and are scaled by their conditional input covariance. The reference is the nonlinear conditional velocity given the full normalized history.</p><img src="finite_change_error.png"><p>The finite-change results reproduce the covariance-weighted derivative ranking: for a GP source, whitened MLP has the smallest error, followed by raw MLP and S4. This differs from the unweighted Jacobian ranking, where S4 was closer than raw MLP. The conclusion depends on the geometry of the perturbations, not solely on whether a derivative or a finite difference is measured.</p><table><tr><th>Source</th><th>Head</th><th>Scale</th><th>Relative error</th><th>Absolute RMS error</th><th>Reference RMS</th></tr>'''+table+'''</table><p>Each table entry averages the per-time, per-seed statistic. In particular, mean relative errors are averages of error/reference ratios, not ratios of the two displayed means. Every underlying numerator and denominator is saved in results.json.</p><p>All 250 finite-change metric rows were independently recalculated from saved arrays; all frozen source and checkpoint hashes remained unchanged. Increasing quadrature from 64 to 128 changes finite velocity differences by at most 1.94 × 10⁻¹² relative RMS. The oracle's linearization error is at most 0.13% at scale 0.01 and 1.29% at scale 0.1. The original models are evaluated in float32; numerical rounding is small at these selected scales but these probes are not a zero-step derivative limit.</p><p>This is an explicitly post-hoc geometric sensitivity check. Covariance-scaled directions and their Mahalanobis radii do not certify likelihood or distributional support for a Student-t mixture. These results do not establish why optimization succeeds or fails, and they do not replace held-out forecast evaluation.</p></html>'''
 (OUT/'report.html').write_text(html)
 save_json(IN/'artifact_hashes.json',{'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(IN.glob('*')) if p.is_file() and p.name!='artifact_hashes.json'},'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 print(json.dumps({'rows':len(s['rows']),'max_error':maxerr,'report':str(OUT/'report.html')}))
if __name__=='__main__':main()
