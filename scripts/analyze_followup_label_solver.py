#!/usr/bin/env python3
"""Independently replay fine-step samples on CPU and summarize paired contrasts."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fm_stress.data import GPConfig
from fm_stress.models import create_model
from fm_stress.experiment import CoordinateModel,generate,tensor,save_json
IN=ROOT/'results/followup-label-solver';OUT=ROOT/'output/followup-label-solver'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);s=json.loads((IN/'summary.json').read_text());assert s['complete'] and len(s['rows'])==24
 manifest=json.loads((IN/'manifest.json').read_text());cfg=json.loads((ROOT/'results/followup-label-noise/config.json').read_text());g=GPConfig(**cfg['data'])
 for p,h in manifest['inputs_sha256'].items():assert sha(ROOT/p)==h
 audits=[]
 for r in s['rows']:
  key=f"{r['data_seed']}_{r['source']}_{r['head']}_{r['label']}_seed{r['seed']}";a=np.load(IN/f'{key}.npz');assert sha(IN/f'{key}.npz')==r['arrays_sha256']
  ens=a['ensemble128'].astype(float);target=a['target'];ns=ens.shape[1];crps=[]
  for row,y in zip(ens,target):crps.append(np.abs(row-y).mean()-np.abs(row[:,None]-row[None,:]).sum()/(2*ns*(ns-1)*row.shape[1]))
  direct_error=float(np.max(np.abs(np.array(crps)-r['per_context128']['fair_crps'])));assert direct_error<1e-9
  name=f"{r['head']}_{r['label']}_seed{r['seed']}";checkpoint=ROOT/f"results/followup-label-noise/dataset_{r['data_seed']}/{r['source']}/checkpoints/{name}.pt";assert sha(checkpoint)==r['checkpoint_sha256']
  model=CoordinateModel(create_model(r['head'],horizon=g.horizon,lookback=g.lookback,**cfg['model']));model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);model.eval()
  n=8;got=generate(model,tensor(a['initial'][:n*ns],'cpu'),tensor(np.repeat(a['history'][:n],ns,axis=0),'cpu'),128,1024).reshape(n,ns,g.horizon)
  rel=float(np.sqrt(np.square(got-ens[:n]).mean())/max(np.sqrt(np.square(ens[:n]).mean()),1e-12));assert rel<2e-4
  audits.append({'run':key,'direct_fair_crps_max_abs_error':direct_error,'cpu_mps_128_replay_relative_rms':rel,'replay_contexts':n})
 indexed={(r['data_seed'],r['source'],r['head'],r['label']):r for r in s['rows']};paired=[]
 for source in ['white','gp']:
  for head in ['mlp','s4']:
   for rep in cfg['replicates']:
    ds=rep['data_seed'];a=indexed[(ds,source,head,'stochastic')];b=indexed[(ds,source,head,'oracle')]
    for metric in ['fair_crps','fair_energy_score','coverage_90','width_90']:
     d64=b['means64'][metric]-a['means64'][metric];d128=b['means128'][metric]-a['means128'][metric]
     paired.append({'data_seed':ds,'source':source,'head':head,'metric':metric,'ordinary64':a['means64'][metric],'oracle64':b['means64'][metric],
                    'ordinary128':a['means128'][metric],'oracle128':b['means128'][metric],'label_difference64':d64,'label_difference128':d128,'difference_change':d128-d64,'sign_preserved':bool(np.sign(d64)==np.sign(d128))})
 cells=[]
 for source in ['white','gp']:
  for head in ['mlp','s4']:
   for label in ['stochastic','oracle']:
    rr=[r for r in s['rows'] if r['source']==source and r['head']==head and r['label']==label]
    cells.append({'source':source,'head':head,'label':label,'means64':{k:float(np.mean([r['means64'][k] for r in rr])) for k in rr[0]['means64']},'means128':{k:float(np.mean([r['means128'][k] for r in rr])) for k in rr[0]['means128']}})
 summary={'complete':True,'cells':cells,'paired_label_contrasts':paired,'max_abs_individual_crps_change':max(abs(r['score_change']['fair_crps']) for r in s['rows']),
          'max_abs_label_crps_contrast_change':max(abs(r['difference_change']) for r in paired if r['metric']=='fair_crps'),
          'crps_label_contrast_sign_changes':[r for r in paired if r['metric']=='fair_crps' and not r['sign_preserved']], 'fingerprint':s['fingerprint']}
 save_json(IN/'independent_verification.json',{'complete':True,'status':'passed','records':audits,'frozen_inputs_unchanged':True});save_json(OUT/'summary.json',summary)
 with (OUT/'paired_contrasts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 table=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [r['source'],r['head'],r['label'],f'{r["means64"]["fair_crps"]:.6f}',f'{r["means128"]["fair_crps"]:.6f}',f'{r["means64"]["coverage_90"]:.4f}',f'{r["means128"]["coverage_90"]:.4f}'])+'</tr>' for r in cells)
 html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Label-study Euler audit</title><style>body{font:17px/1.55 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#23343e}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:nth-child(-n+3){text-align:left}</style><h1>Does the label-study conclusion depend on Euler resolution?</h1><p>All 24 frozen checkpoints were rerun on MPS at 64 and 128 Euler steps, using exactly the same 256 histories and 32 initial paths per history. No model was retrained or reselected. The 64-step replay reproduces the original outputs; every 128-step ensemble has an additional eight-context CPU replay and independent direct CRPS arithmetic.</p><table><tr><th>Source</th><th>Head</th><th>Labels</th><th>CRPS 64</th><th>CRPS 128</th><th>Coverage 64</th><th>Coverage 128</th></tr>'''+table+'''</table><p>Entries average three independent data/initialization replicates. Coverage is nominal 90%. Full paired replicate contrasts are saved in paired_contrasts.csv.</p><p>The largest individual CRPS change is '''+f'{summary["max_abs_individual_crps_change"]:.6f}'+'''; the largest change in a paired oracle-versus-ordinary label contrast is '''+f'{summary["max_abs_label_crps_contrast_change"]:.6f}'+'''. The number of label-contrast sign reversals is '''+str(len(summary['crps_label_contrast_sign_changes']))+'''. This is a post-hoc numerical sensitivity check, not proof of exact ODE integration or additional independent training evidence.</p></html>'''
 (OUT/'report.html').write_text(html);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
