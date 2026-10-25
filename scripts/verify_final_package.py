#!/usr/bin/env python3
"""Check the delivered manifest and paper/slide numerical evidence, stdlib only."""
import hashlib,json,subprocess,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
p=read('FINAL_PACKAGE.json')
for name,value in p['files'].items():
 assert sha(name)==value, f'Delivered file changed: {name}'
subprocess.run(['git','cat-file','-e',p['analysis_revision']+'^{commit}'],cwd=ROOT,check=True)
e=read('output/final-paper/evidence.json')
assert e['trajectory_metrics']==read('results/trajectory-metrics/summary.json')
assert e['pca_audit']==read('results/final-pca-audit/summary.json')
roles=read('docs/contributions.json')
assert e['contributions']==roles and roles['status']=='user_approved'
for name,value in e['source_paths'].items():assert sha(name)==value,name
content=read('output/final-presentation/content.json')
figroot='output/final-presentation/figures/'
fp=read(figroot+'provenance.json')
for name,value in fp['input_sha256'].items():assert sha(name)==value,name
for name,value in fp['output_sha256'].items():assert sha(figroot+name)==value,name
sp=read(figroot+'score_differences.provenance.json')
for key in ('source','script','output'):assert sha(sp[key])==sp[key+'_sha256'],sp[key]
assert len(content['slides'])==18
assert len([s for s in content['slides'] if not s['backup']])==14
assert content['slides'][13]['rows']==[[r['name'],r['slide_summary']] for r in roles['members']]
assert content['slides'][13]['seconds']==30 and not content['slides'][13]['backup']
assert sum(s['seconds'] for s in content['slides'] if not s['backup'])==content['timing']['speech_seconds']
assert content['timing']['planned_total_seconds']<900
assert content['timing']['planned_total_seconds']==sum(content['timing'][k] for k in ('speech_seconds','pause_transition_seconds','contribution_reserve_seconds'))
for index in (9,10,11):assert not content['slides'][index]['backup']
canonical={}
for file in ('output/extension-analysis/summary.json','output/sparse-analysis/summary.json'):
 for row in read(file)['aggregate']:canonical[row['dataset'],row['source'],row['head']]=row
for row in e['multivariate']['cells']:
 key=(row['dataset'],row['source'],'mlp_whitened' if row['head']=='whitened_mlp' else row['head'])
 for metric in ('fair_crps','joint_energy_scaled'):
  assert row['metrics'][metric]==canonical[key]['metrics'][metric],key
cells={(r['dataset'],r['source'],r['head']):r for r in e['multivariate']['cells']}
for series,head in zip(content['slides'][10]['chart']['series'],('mlp','s4','whitened_mlp')):
 assert series['values']==[cells[d,'gp',head]['metrics']['fair_crps']['mean'] for d in ('gp_multi','etth1_uni','etth1_multi','exchange_multi')]
assert content['slides'][4]['chart']['series'][0]['values']==[100*e['main']['geometry'][k]['test']['top2_energy'] for k in ('target','white','gp')]
for row,cell in zip(content['slides'][5]['rows'],('white|mlp','white|s4','gp|mlp','gp|s4','gp|whitened_mlp')):
 m=e['main']['groups'][cell]['metrics']
 assert row[1:]==[f"{m['pc1_mse']['mean']:.3f}",f"{m['residual_mse']['mean']:.4f}",f"{m['roughness_ratio']['mean']:.3f}"],cell
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(ROOT/'output/final-presentation/defense.pptx') as z:
 media_hashes={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
 for i,s in enumerate(content['slides'],1):
  xml=ET.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
  text=' '.join(x.text or '' for x in xml.findall('.//a:t',ns))
  normalized=' '.join(text.split())
  values=s.get('headers',[])+[v for row in s.get('rows',[]) for v in row]
  for value in values:assert ' '.join(str(value).split()) in normalized,(i,value)
  if s['kind']=='figure':assert sha(s['figure']) in media_hashes, f'Figure missing or altered: {s["figure"]}'
print(f"Verified {len(p['files'])} delivered files, {len(e['source_paths'])} evidence inputs, 18 slide texts, multivariate-source identity and timing below 15 minutes.")
