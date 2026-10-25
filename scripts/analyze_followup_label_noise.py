#!/usr/bin/env python3
"""Report all paired label-noise interventions and audit saved metric arithmetic."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fm_stress.experiment import save_json
from fm_stress.followup_label_noise import folder,sha

KEYS={'residual_mse':('metrics','residual_mse'),'pc1_mse':('metrics','pc1_mse'),
      'oracle_residual_mse':('oracle_discrepancy','residual_mse'),'oracle_total_mse':('oracle_discrepancy','velocity_mse'),
      'fair_crps':('forecast_means','fair_crps'),'energy_score':('forecast_means','fair_energy_score'),
      'coverage_90':('forecast_means','coverage_90')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='results/followup-label-noise');p.add_argument('--output',default='output/followup-label-noise');a=p.parse_args()
 root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
 baselines=json.loads((root/'predictive_baselines.json').read_text()) if (root/'predictive_baselines.json').exists() else None
 baseline_sizes=[]
 if baselines:
  for rep in cfg['replicates']:
   bp=root/f'dataset_{rep["data_seed"]}'/'predictive_baselines.npz';ba=np.load(bp)
   for members in [32,64]:
    lo,hi=np.quantile(ba['matched'][:,:members],[.05,.95],axis=1,method='linear')
    baseline_sizes.append({'data_seed':rep['data_seed'],'members':members,'coverage_90':float(((ba['target']>=lo)&(ba['target']<=hi)).mean()),'samples_sha256':sha(bp),'note':'Post-hoc arithmetic on the first members of the existing exact-reference ensemble; no new samples or training.'})
  save_json(root/'predictive_baseline_size_sensitivity.json',{'complete':len(baseline_sizes)==6,'rows':baseline_sizes})
 records=[];missing=[];audit=[];maxerr=0.;expected=0
 for rep in cfg['replicates']:
  for src in cfg['sources']:
   group=folder(root,rep['data_seed'],src)
   for head in cfg['heads']:
    for label in cfg['labels']:
     expected+=1;name=f"{head}_{label}_seed{rep['seed']}";path=group/'runs'/f'{name}.json'
     if not path.exists():missing.append(str(path));continue
     r=json.loads(path.read_text());sam=np.load(group/'samples'/f'{name}.npz');test=np.load(group/'test.npz');P=np.array(json.loads((group/'pca.json').read_text())['components'])
     if r['fingerprint']!=manifest['fingerprint'] or r['samples_sha256']!=sha(group/'samples'/f'{name}.npz') or r['checkpoint_sha256']!=sha(group/'checkpoints'/f'{name}.pt'):raise AssertionError('Hash mismatch '+name)
     # Independent direct per-PC arithmetic; no production metric helper.
     for field,labelname in [('metrics','stochastic'),('oracle_discrepancy','oracle')]:
      pc=np.square((sam['prediction']-test[labelname])@P.T).mean(0)
      err=float(np.max(np.abs(pc-r[field]['pc_mse'])));maxerr=max(maxerr,err);assert err<1e-9
     ens=sam['ensemble'].astype(float);target=test['target'][:len(ens)];S=ens.shape[1]
     # Direct O(S²) definition, independent of sorted-rank CRPS implementation.
     crps=[]
     for samples,y in zip(ens,target):
      crps.append(float(np.abs(samples-y).mean()-np.abs(samples[:,None]-samples[None,:]).sum()/(2*S*(S-1)*samples.shape[1])))
     err=abs(np.mean(crps)-r['forecast_means']['fair_crps']);maxerr=max(maxerr,err);assert err<1e-9
     # Independent type-7 order-statistic interpolation for the 90% interval.
     ordered=np.sort(ens,axis=1);bounds=[]
     for probability in [.05,.95]:
      position=(S-1)*probability;low=int(np.floor(position));fraction=position-low
      bounds.append((1-fraction)*ordered[:,low]+fraction*ordered[:,min(low+1,S-1)])
     coverage=float(((target>=bounds[0])&(target<=bounds[1])).mean())
     err=abs(coverage-r['forecast_means']['coverage_90']);maxerr=max(maxerr,err);assert err<1e-12
     width=float((bounds[1]-bounds[0]).mean());assert abs(width-r['forecast_means']['width_90'])<1e-12
     oracle_error=test['oracle']-test['stochastic'];d=sam['prediction']-test['oracle']
     cross=2*np.mean((d@P.T)*(oracle_error@P.T),0)
     identity=np.square((sam['prediction']-test['stochastic'])@P.T).mean(0)-np.square(d@P.T).mean(0)-np.square(oracle_error@P.T).mean(0)-cross
     assert np.max(abs(identity))<1e-9
     audit.append({'run':str(path),'max_decomposition_error':float(np.max(abs(identity)))})
     records.append(r)
 rows=[]
 for r in records:
  row={k:r[k] for k in ['data_seed','seed','source','head','label','best_step','parameters']}
  row.update({k:r[parts[0]][parts[1]] for k,parts in KEYS.items()})
  row['weighted_residual_jacobian_error']=float(np.mean([v['residual']['covariance_weighted_relative_error'] for v in r['jacobian']]))
  row['unweighted_residual_jacobian_error']=float(np.mean([v['residual']['mean_relative_error'] for v in r['jacobian']]))
  row['roughness_ratio']=r['metrics']['success']['roughness_ratio'];row['literal_success']=r['metrics']['success']['works']
  if baselines:
   b=next(b for b in baselines['rows'] if b['data_seed']==r['data_seed'])
   row['fair_crps_gap_to_matched_predictive_baseline']=row['fair_crps']-b['scores']['matched']['mean']['fair_crps']
  rows.append(row)
 if rows:
  with (out/'runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 indexed={(r['data_seed'],r['source'],r['head'],r['label']):r for r in rows};contrasts=[]
 metricnames=list(KEYS)+['weighted_residual_jacobian_error','unweighted_residual_jacobian_error','roughness_ratio']
 for src in cfg['sources']:
  for head in cfg['heads']:
   for metric in metricnames:
    paired=[]
    for rep in cfg['replicates']:
     ds=rep['data_seed'];base=indexed.get((ds,src,head,'stochastic'));new=indexed.get((ds,src,head,'oracle'))
     if base and new:paired.append({'data_seed':ds,'stochastic':base[metric],'oracle':new[metric], 'difference':new[metric]-base[metric], 'ratio':new[metric]/base[metric] if base[metric] else None})
    optimum=.9 if metric=='coverage_90' else 1. if metric=='roughness_ratio' else None
    for pair in paired:pair['desirable_change']=(abs(pair['oracle']-optimum)<abs(pair['stochastic']-optimum)) if optimum is not None else pair['difference']<0
    if paired:contrasts.append({'source':src,'head':head,'metric':metric,'per_draw':paired,'mean_difference':float(np.mean([v['difference'] for v in paired])), 'mean_ratio':float(np.mean([v['ratio'] for v in paired])) if all(v['ratio'] is not None for v in paired) else None,'improvement_definition':f'closer to {optimum}' if optimum is not None else 'lower value','consistent_improvement':all(v['desirable_change'] for v in paired),'complete':len(paired)==len(cfg['replicates'])})
 interactions=[]
 for metric in ['residual_mse','oracle_residual_mse','fair_crps']:
  for axis in ['source','head']:
   keep=cfg['heads'] if axis=='source' else cfg['sources']
   for fixed in keep:
    pairs=[]
    for rep in cfg['replicates']:
     ds=rep['data_seed'];vals=[]
     for variable in (cfg['sources'] if axis=='source' else cfg['heads']):
      src,head=(variable,fixed) if axis=='source' else (fixed,variable)
      b=indexed.get((ds,src,head,'stochastic'));n=indexed.get((ds,src,head,'oracle'))
      if b and n and min(b[metric],n[metric])>0:vals.append(np.log(n[metric]/b[metric]))
     if len(vals)==2:pairs.append({'data_seed':ds,'log_ratio_difference':float(vals[1]-vals[0])})
    interactions.append({'axis':axis,'fixed':fixed,'metric':metric,'contrast':'GP minus white' if axis=='source' else 'S4 minus MLP','per_draw':pairs,'mean':float(np.mean([v['log_ratio_difference'] for v in pairs])) if pairs else None})
 # Training stream and encoder initializations must match within a draw.
 pairing=[]
 for rep in cfg['replicates']:
  rr=[r for r in records if r['data_seed']==rep['data_seed']]
  assert len({r['training']['encoder_initialization_sha256'] for r in rr})<=1
  for src in cfg['sources']:
   subset=[r for r in rr if r['source']==src]
   assert len({json.dumps(r['training']['stream_files_sha256'],sort_keys=True) for r in subset})<=1
  pairing.append({'data_seed':rep['data_seed'],'encoder_and_stream_pairing_passed':True})
 noise_geometry=[]
 for rep in cfg['replicates']:
  for src in cfg['sources']:
   group=folder(root,rep['data_seed'],src)
   if not (group/'test.npz').exists():continue
   arr=np.load(group/'test.npz');Q=np.array(json.loads((group/'pca.json').read_text())['components'])
   pc=arr['stochastic']@Q.T;variance=pc.var(0,ddof=1)
   noise=np.diagonal(Q@arr['noise_covariance']@Q.T,axis1=1,axis2=2).mean(0)
   noise_geometry.append({'data_seed':rep['data_seed'],'source':src,'label_variance_per_pc':variance.tolist(),'expected_conditional_noise_per_pc':noise.tolist(),'noise_to_total_variance_per_pc':(noise/variance).tolist(),'residual_noise':float(noise[2:].mean()),'pc1_noise':float(noise[0]),'note':'Finite test estimate; ratio is descriptive, no clipping and no reinterpretation of original success rule.'})
 summary={'complete':len(records)==expected,'expected_runs':expected,'observed_runs':len(records),'missing':missing,'rows':rows,'contrasts':contrasts,'interactions':interactions,
          'best_at_final_step':sum(r['best_step']==cfg['steps'] for r in records),'literal_successes':sum(r['metrics']['success']['works'] for r in records),
          'solver_sensitivity':[{'data_seed':r['data_seed'],'source':r['source'],'head':r['head'],'label':r['label'],**r['solver_sensitivity']} for r in records if r.get('solver_sensitivity')],'label_noise_geometry':noise_geometry,'predictive_baselines':baselines,'predictive_baseline_size_sensitivity':baseline_sizes,'audit':{'status':'passed available runs','max_metric_error':maxerr,'records':audit,'pairing':pairing},'fingerprint':manifest['fingerprint']}
 full_solver_path=ROOT/'output/followup-label-solver/summary.json'
 summary['full_solver_sensitivity']=json.loads(full_solver_path.read_text()) if full_solver_path.exists() else None
 save_json(out/'summary.json',summary)
 if summary['complete']:render(summary,out)
 print(json.dumps({'complete':summary['complete'],'observed':len(records),'expected':expected,'max_metric_error':maxerr,'best_at_final_step':summary['best_at_final_step']},indent=2))

def render(s,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'mlp':'#2867a5','s4':'#d5653e'};fig,axs=plt.subplots(1,3,figsize=(12,4.2))
 for ax,metric,title in zip(axs,['residual_mse','oracle_residual_mse','fair_crps'],['Residual velocity MSE','Matched-oracle residual discrepancy','Forecast CRPS']):
  for i,src in enumerate(['white','gp']):
   for j,head in enumerate(['mlp','s4']):
    c=next(r for r in s['contrasts'] if r['source']==src and r['head']==head and r['metric']==metric)
    vals=[v['ratio'] for v in c['per_draw']];x=i*3+j
    ax.bar(x,np.mean(vals),color=colors[head],width=.7,alpha=.8)
    ax.scatter(np.full(len(vals),x),vals,color='#222222',s=15,zorder=3)
  ax.axhline(1,color='#777777',ls='--',lw=1);ax.set_xticks([0,1,3,4],['White\nMLP','White\nS4','GP\nMLP','GP\nS4']);ax.set_title(title,loc='left',fontsize=11);ax.set_ylabel('Oracle-label / ordinary-label result');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Conditional-mean supervision at a fixed training budget',x=.02,ha='left',fontsize=14)
 fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'label_noise_intervention.png',dpi=180);plt.close(fig)
 table=[]
 for src in ['white','gp']:
  for head in ['mlp','s4']:
   for label in ['stochastic','oracle']:
    rr=[r for r in s['rows'] if r['source']==src and r['head']==head and r['label']==label]
    table.append('<tr><td>'+src+'</td><td>'+head+'</td><td>'+label+'</td>'+''.join(f'<td>{np.mean([r[k] for r in rr]):.5f}</td>' for k in ['residual_mse','oracle_residual_mse','fair_crps','weighted_residual_jacobian_error'])+'</tr>')
 assignment_rows=[]
 for src in ['white','gp']:
  for head in ['mlp','s4']:
   for label in ['stochastic','oracle']:
    rr=[r for r in s['rows'] if r['source']==src and r['head']==head and r['label']==label]
    assignment_rows.append('<tr><td>'+src+'</td><td>'+head+'</td><td>'+label+'</td>'+''.join(f'<td>{np.mean([r[k] for r in rr]):.5f}</td>' for k in ['pc1_mse','residual_mse','roughness_ratio'])+'</tr>')
 html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Conditional label-noise study</title><style>body{font:17px/1.55 system-ui;color:#23323b;max-width:1120px;margin:45px auto;padding:0 24px}h1{font-size:34px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px;border-bottom:1px solid #dde3e6;text-align:right}td:nth-child(-n+3){text-align:left}img{width:100%}</style><h1>Does noisy supervision limit the learned flow?</h1><p>24 fresh fits: two sources, two heads, two target estimators and three independent data/initialization pairs. Every pair of label modes sees exactly the same 1,536,000 training interpolants. Ordinary CFM labels are replaced by their conditional expectation given the normalized history and interpolated point. Both modes use oracle-discrepancy validation selection; this differs from the original baseline's selection policy.</p><img src="label_noise_intervention.png"><p>Bars average three paired replicate-specific ratios; dots show every replicate. Below 1 favours oracle supervision. No parameter or checkpoint was selected by these test scores.</p><table><tr><th>Source</th><th>Head</th><th>Training label</th><th>Residual MSE</th><th>Oracle discrepancy</th><th>CRPS</th><th>Weighted Jacobian error</th></tr>'''+''.join(table)+'''</table><p>Conditional-mean labels remove conditional target noise while preserving the population regression optimum. The analytical generator is known only in this controlled synthetic study; this is not a deployable forecasting method. A finite history pool is reused, so the empirical objective is not exactly unchanged. Improvements show sensitivity to conditional-mean supervision at this training budget and validation policy. The intervention also replaces one realized future per training history with knowledge of the population conditional law, so it does not isolate empirical label variance alone or establish intrinsic architecture failure or a Hessian mechanism.</p><p>Boundary-selected checkpoints: '''+str(s['best_at_final_step'])+'''/24. Literal assignment-rule successes: '''+str(s['literal_successes'])+'''/24. Full per-draw results, source/head interactions, numerical audits and caveats are in summary.json and docs/followup_label_noise_protocol.md.</p></html>'''
 if s.get('solver_sensitivity'):
  maximum=max(abs(r['score_change']['fair_crps']) for r in s['solver_sensitivity'])
  html=html.replace('</html>',f'<p>Euler 64 versus 128 was checked on eight contexts from the first replicate for all eight cells; the largest absolute fair-CRPS shift on that subset is {maximum:.5f}. This is a limited numerical sensitivity check, not a complete solver audit or proof that tiny score differences are resolved.</p></html>')
 html=html.replace('</html>','<h2>Original assignment metrics</h2><p>These are the original raw-coordinate metrics: PC1 velocity MSE, residual-PC velocity MSE and generated-to-target roughness ratio. The roughness target is 1. The original literal pass rule is retained without changing its thresholds. Velocity PCA is fitted separately for each source; use the common-target forecast scores for direct source-quality comparisons. The velocity rule is two-sided: residual MSE must be within 20% of PC1 MSE. Unequal PC-specific label-noise floors complicate that comparison; low residual error alone does not pass it. Smoothness alone also does not establish conditional forecasting quality.</p><table><tr><th>Source</th><th>Head</th><th>Training label</th><th>PC1 MSE</th><th>Residual MSE</th><th>Roughness ratio</th></tr>'+''.join(assignment_rows)+'</table></html>')
 if s.get('predictive_baseline_size_sensitivity'):
  c32=np.mean([r['coverage_90'] for r in s['predictive_baseline_size_sensitivity'] if r['members']==32]);c64=np.mean([r['coverage_90'] for r in s['predictive_baseline_size_sensitivity'] if r['members']==64])
  html=html.replace('</html>',f'<p>Finite ensemble size matters for interval coverage: rescoring the first 32 of the already saved exact-reference samples gives mean nominal-90% coverage {c32:.4f}, versus {c64:.4f} with 64 members. These are finite-test estimates. Coverage below 0.90 by itself therefore does not establish underdispersion in a 32-member forecast; compare the reference at the same ensemble size.</p></html>')
 if s.get('full_solver_sensitivity'):
  ss=s['full_solver_sensitivity']
  html=html.replace('</html>',f'<h2>Full-ensemble numerical follow-up</h2><p>A separate post-hoc audit reran all 24 checkpoints at 64 and 128 Euler steps on all 256 contexts and 32 members. Maximum individual CRPS change: {ss["max_abs_individual_crps_change"]:.6f}; maximum change in an oracle-versus-ordinary label contrast: {ss["max_abs_label_crps_contrast_change"]:.6f}; contrast sign reversals: {len(ss["crps_label_contrast_sign_changes"])}. Original 64-step scores above remain unchanged. See <a href="../followup-label-solver/report.html">the full numerical audit</a>.</p></html>')
 if s.get('predictive_baselines'):
  bb=s['predictive_baselines']['rows'];text='<h2>Predictive Bayes reference with the same normalized history</h2><p>The exact normalized-history Student-t predictive law has mean CRPS '+f"{np.mean([b['scores']['matched']['mean']['fair_crps'] for b in bb]):.5f}"+'. The raw-history Gaussian reference has mean CRPS '+f"{np.mean([b['scores']['privileged']['mean']['fair_crps'] for b in bb]):.5f}"+'. The latter has extra information; it is not the network-input Bayes target. These direct 64-member predictive samples use the same test contexts and no Euler integration; unbiased fair scores are comparable in expectation with the network 32-member scores, with different Monte Carlo variance. Tiny encoders can discard information present in the full normalized history.</p>'
  html=html.replace('</html>',text+'</html>')
 (out/'report.html').write_text(html)

if __name__=='__main__':main()
