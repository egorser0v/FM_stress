#!/usr/bin/env python3
"""Summarize capacity controls; paired independent draws, budgets correlated."""
import argparse,csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def old_scan_bridge(output):
 """Historical RBF white-source comparison, all five geometries, no selection."""
 oldroot=Path('results/kernel-scan');oldcfg=json.loads((oldroot/'config.json').read_text())
 observations=[];cells=[];paired=[]
 for geometry in oldcfg['geometries']:
  gid=geometry['id'];local=[]
  for replicate in oldcfg['replicates']:
   draw=replicate['data_seed'];byhead={}
   for head in ['mlp','s4']:
    path=oldroot/'rbf'/gid/f'draw{draw}'/'runs'/f'white_{head}.json'
    record=json.loads(path.read_text())
    row={'kernel':'rbf','source':'white','geometry':gid,'horizon':geometry['horizon'],'lookback':geometry['lookback'],'lengthscale':geometry['lengthscale'],'F_over_ell':geometry['horizon']/geometry['lengthscale'],'data_seed':draw,'optimization_seed':record['seed'],'head':head,'width':record['width'],'parameters':record['parameters'],'residual_mse':record['metrics']['residual_mse'],'record_path':str(path),'record_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    observations.append(row);local.append(row);byhead[head]=row
   paired.append({'geometry':gid,'data_seed':draw,'mlp_residual_mse':byhead['mlp']['residual_mse'],'s4_residual_mse':byhead['s4']['residual_mse'],'s4_over_mlp':byhead['s4']['residual_mse']/byhead['mlp']['residual_mse']})
  means={h:float(np.mean([r['residual_mse'] for r in local if r['head']==h])) for h in ['mlp','s4']}
  ratios=[r['s4_over_mlp'] for r in paired if r['geometry']==gid]
  cells.append({'geometry':gid,'horizon':geometry['horizon'],'lookback':geometry['lookback'],'lengthscale':geometry['lengthscale'],'F_over_ell':geometry['horizon']/geometry['lengthscale'],'n_paired_draws':len(ratios),'mean_residual_mse':means,'ratio_of_means_s4_over_mlp':means['s4']/means['mlp'],'mean_paired_ratio_s4_over_mlp':float(np.mean(ratios))})
 bridge={'scope':'Historical Vasya kernel scan, RBF targets and white source, all five configured geometries; no new training.','aggregation':'Within each geometry and head, arithmetic mean of residual MSE across the three independent data/optimization pairs. Primary displayed ratio is mean(S4 residual MSE) / mean(MLP residual MSE). Individual paired ratios and their arithmetic mean are also supplied; these are distinct aggregations.','interpretation':'The pattern is consistent with a dimension-dependent architectural restriction. Changing horizon also changes lookback, normalized geometry and sequence length, so this historical comparison alone does not identify F/ell as a cause.','config_sha256':hashlib.sha256((oldroot/'config.json').read_bytes()).hexdigest(),'cells':cells,'paired_draws':paired,'observations':observations}
 (output/'vasya_geometry_bridge.json').write_text(json.dumps(bridge,indent=2)+'\n')
 with (output/'vasya_geometry_bridge.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(observations[0]));w.writeheader();w.writerows(observations)

METRICS=['fair_crps','fair_energy_score','pc1_mse','pc2_mse','residual_mse','target_basis_residual_mse','roughness_ratio','null_forced_mse','null_source_variance','null_target_variance']

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',default='results/followup-capacity');p.add_argument('--output',default='output/followup-capacity');a=p.parse_args();root=Path(a.results);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 old_scan_bridge(out)
 records=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/draw*/runs/*.json'))]
 cfg=json.loads((root/'config.json').read_text()); expected=sum(len(h) for h in cfg['kernel_heads'].values())*len(cfg['sources'])*len(cfg['replicates'])*len(cfg['budgets'])
 rows=[]
 for r in records:
  n=r['output_null_diagnostic'];m=r['metrics']
  rows.append({k:r[k] for k in ['kernel','source','head','data_seed','seed','budget','best_step','parameters']}|{
   'fair_crps':r['forecast_means']['fair_crps'],'fair_energy_score':r['forecast_means']['fair_energy_score'],
   'pc1_mse':m['pc1_mse'],'pc2_mse':m['pc2_mse'],'residual_mse':m['residual_mse'],
   'target_basis_residual_mse':r['fixed_target_basis_metrics']['residual_mse'],
   'roughness_ratio':m['generated_roughness']/m['target_roughness'],
   'literal_rule_pass':m['success']['works'],'roughness_rule_pass':m['success']['roughness_rule_pass'],'null_dimension':n.get('null_dimension'),'null_forced_mse':n.get('forced_velocity_mse_per_coordinate'),
   'null_source_variance':n.get('source_null_variance_per_coordinate'),'null_target_variance':n.get('target_null_variance_per_coordinate')})
 with (out/'runs.csv').open('w') as f:
  if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups=defaultdict(list)
 for row in rows:groups[(row['kernel'],row['source'],row['head'],row['budget'])].append(row)
 summaries=[]
 for (k,s,h,b),rr in groups.items():
  d={'kernel':k,'source':s,'head':h,'budget':b,'n_draws':len(rr),'best_at_budget':sum(r['best_step']==b for r in rr),'literal_rule_passes':sum(r['literal_rule_pass'] for r in rr),'roughness_rule_passes':sum(r['roughness_rule_pass'] for r in rr)}
  d['metrics']={m:{'mean':float(np.mean([r[m] for r in rr])),'min':float(min(r[m] for r in rr)),'max':float(max(r[m] for r in rr))} for m in METRICS if all(r[m] is not None for r in rr)}
  summaries.append(d)
 lookup={(r['kernel'],r['source'],r['head'],r['budget'],r['data_seed']):r for r in rows};contrasts=[]
 for k,heads in cfg['kernel_heads'].items():
  for s in cfg['sources']:
   for b in cfg['budgets']:
    pairs=[('mlp_rank_clear','s4')]+([('mlp_rank_clear','mlp_original'),('mlp_wide','mlp_original'),('s4','mlp_original')] if k=='rbf' else [])
    for candidate,reference in pairs:
     rr=[]
     for rep in cfg['replicates']:
      ds=rep['data_seed'];ca=lookup.get((k,s,candidate,b,ds));re=lookup.get((k,s,reference,b,ds))
      if ca and re:rr.append({'data_seed':ds,**{m:ca[m]/re[m]-1 for m in ['fair_crps','fair_energy_score','residual_mse']}})
     if rr:contrasts.append({'kernel':k,'source':s,'budget':b,'candidate':candidate,'reference':reference,'paired_relative_changes':rr,'mean_relative_change':{m:float(np.mean([r[m] for r in rr])) for m in ['fair_crps','fair_energy_score','residual_mse']}})
 budgets=[]
 for k,s,h,b in groups:
  if b!=cfg['budgets'][0]:continue
  vals=[]
  for rep in cfg['replicates']:
   d=rep['data_seed'];lo=lookup.get((k,s,h,cfg['budgets'][0],d));hi=lookup.get((k,s,h,cfg['budgets'][-1],d))
   if lo and hi:vals.append({'data_seed':d,**{m:hi[m]/lo[m]-1 for m in ['fair_crps','fair_energy_score','residual_mse']}})
  if vals:budgets.append({'kernel':k,'source':s,'head':h,'paired_12000_vs_6000':vals})
 closures=[]
 for source in cfg['sources']:
  for budget in cfg['budgets']:
   for rep in cfg['replicates']:
    ds=rep['data_seed'];base=lookup.get(('rbf',source,'mlp_original',budget,ds));s4=lookup.get(('rbf',source,'s4',budget,ds))
    if not base or not s4:continue
    for head in ['mlp_rank_clear','mlp_wide']:
     candidate=lookup.get(('rbf',source,head,budget,ds))
     if not candidate:continue
     gap=base['residual_mse']-s4['residual_mse'];remaining=candidate['residual_mse']-s4['residual_mse']
     closures.append({'source':source,'budget':budget,'data_seed':ds,'candidate':head,'original_mlp_minus_s4_residual':gap,'candidate_minus_s4_residual':remaining,'fraction_of_positive_original_gap_closed':(gap-remaining)/gap if gap>0 else None})
 summary={'gap_closure_diagnostic':closures,'complete':len(records)==expected,'expected_checkpoint_records':expected,'observed_checkpoint_records':len(records),'independent_training_fits':len(records)//len(cfg['budgets']),'note':'Budget records are correlated prefix selections on the same training trajectory. Three independent draws; no test tuning.','cells':summaries,'contrasts':contrasts,'budget_changes':budgets}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 for kernel in cfg['kernel_heads']:
  fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
  for i,source in enumerate(cfg['sources']):
   for j,metric in enumerate(['residual_mse','fair_crps']):
    ax=axes[i,j]
    for hi,head in enumerate(cfg['kernel_heads'][kernel]):
     for bi,b in enumerate(cfg['budgets']):
      rr=groups.get((kernel,source,head,b),[])
      if not rr:continue
      x=hi+(-.12 if bi==0 else .12);ys=[r[metric] for r in rr]
      ax.scatter([x]*len(ys),ys,color=['#a6bbc7','#087f8c'][bi],s=28)
      ax.plot([x-.09,x+.09],[np.mean(ys)]*2,color=['#627d8a','#064e56'][bi],lw=2.2)
    ax.set_xticks(range(len(cfg['kernel_heads'][kernel])),[{'mlp_original':'MLP 56×3','mlp_rank_clear':'MLP 65×2','mlp_wide':'MLP 72×3','s4':'S4 59×3'}[h] for h in cfg['kernel_heads'][kernel]])
    ax.set_title(f'{source.upper()} source · '+('Residual velocity MSE' if metric=='residual_mse' else 'Fair CRPS'))
    ax.set_ylabel('Lower is better');ax.grid(axis='y',alpha=.18)
  fig.suptitle(f'{kernel.upper()} · F=64 · capacity and training budget\nPale: 6,000 updates; teal: 12,000. Points: independent draws; bars: mean.',fontsize=15)
  fig.savefig(out/f'{kernel}_capacity.png',dpi=170);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
  for ax,source in zip(axes,cfg['sources']):
   for hi,head in enumerate(cfg['kernel_heads'][kernel]):
    for bi,b in enumerate(cfg['budgets']):
     rr=groups.get((kernel,source,head,b),[])
     if not rr:continue
     x=hi+(-.12 if bi==0 else .12);ys=[r['roughness_ratio'] for r in rr]
     ax.scatter([x]*len(ys),ys,color=['#a6bbc7','#087f8c'][bi],s=30)
     ax.plot([x-.09,x+.09],[np.mean(ys)]*2,color=['#627d8a','#064e56'][bi],lw=2.2)
   ax.axhline(1,color='#aa6b32',lw=1.3,ls='--',label='Target roughness')
   ax.set_xticks(range(len(cfg['kernel_heads'][kernel])),[{'mlp_original':'MLP 56×3','mlp_rank_clear':'MLP 65×2','mlp_wide':'MLP 72×3','s4':'S4 59×3'}[h] for h in cfg['kernel_heads'][kernel]])
   ax.set_title(source.upper()+' source');ax.set_ylabel('Generated / target roughness');ax.grid(axis='y',alpha=.18)
  fig.suptitle(f'{kernel.upper()} · F=64 · generated trajectory roughness (target = 1)\nPale: 6,000 updates; teal: 12,000. Width × depth shown below.',fontsize=13)
  fig.savefig(out/f'{kernel}_roughness.png',dpi=170);plt.close(fig)
 # Fixed illustrative selection: first configured data draw, first held-out
 # history, first four ensemble members. Never select attractive trajectories.
 draw=cfg['replicates'][0]['data_seed'];budget=cfg['budgets'][-1];heads=cfg['kernel_heads']['rbf']
 if all(('rbf',source,head,budget,draw) in lookup for source in cfg['sources'] for head in heads):
  fig,axes=plt.subplots(2,len(heads),figsize=(16,7),sharex=True,sharey=True,constrained_layout=True)
  for ri,source in enumerate(cfg['sources']):
   for ci,head in enumerate(heads):
    sample=root/'rbf'/'F64_l8'/f'draw{draw}'/'samples'/f'{source}_{head}_budget{budget}.npz'
    with np.load(sample) as z:
     paths=z['ensemble'][0,:4];target=z['ensemble_target'][0]
    ax=axes[ri,ci]
    for idx,path in enumerate(paths):ax.plot(path,color='#087f8c',alpha=.38,lw=1.1,label='Generated' if idx==0 else None)
    ax.plot(target,color='#222c36',lw=2,label='Held-out target');ax.grid(alpha=.15)
    ax.set_title({'mlp_original':'MLP 56 × 3','mlp_rank_clear':'MLP 65 × 2','mlp_wide':'MLP 72 × 3','s4':'S4 59 × 3'}[head])
    if ci==0:ax.set_ylabel(source.upper()+' source\nNormalized value')
    if ri==1:ax.set_xlabel('Forecast step')
  axes[0,0].legend(frameon=False,fontsize=9)
  fig.suptitle('RBF · F=64 · one fixed held-out context, four matched source draws\n12,000-update budget; first configured data draw and test context, no visual selection.',fontsize=14)
  fig.savefig(out/'rbf_example_paths.png',dpi=170);plt.close(fig)
 print(json.dumps({k:summary[k] for k in ['complete','expected_checkpoint_records','observed_checkpoint_records']}))
if __name__=='__main__':main()
