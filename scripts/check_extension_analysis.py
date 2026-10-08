"""Independent final aggregation/OT-slice audit; does not import the analyzer."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_extension import independent_scores,mean_scores
from fm_stress.extended_data import build_data


def read(path):return json.loads(path.read_text())
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'results/extension';path=ROOT/'output/extension-analysis/summary.json'
    summary=read(path);manifest=read(folder/'manifest.json');cfg=manifest['config']
    assert summary['status']=='complete' and summary['observed_runs']==96
    assert cfg==summary['config'] and summary['fingerprint']==manifest['fingerprint']
    records={}
    for dataset in cfg['datasets']:
        for p in (folder/dataset/'runs').glob('*.json'):
            r=read(p);key=(dataset,r['head'],r['source'],r['seed'])
            assert key not in records;records[key]=r
    assert len(records)==96
    scalar_pairs=0
    for cell in summary['aggregate']:
        rows=[records[(cell['dataset'],cell['head'],cell['source'],seed)] for seed in cfg['seeds']]
        assert cell['n']==3
        for metric,value in cell['metrics'].items():
            data=[r['velocity_metrics'][metric.removeprefix('velocity_')] if metric.startswith('velocity_')
                  else r['forecast_means'][metric] for r in rows]
            close(np.mean(data),value['mean']);close(np.std(data,ddof=1),value['sd']);scalar_pairs+=1
        close(np.mean([r['seconds'] for r in rows]),cell['seconds']['mean'])
        close(np.std([r['seconds'] for r in rows],ddof=1),cell['seconds']['sd'])
        close(np.mean([r['velocity_metrics']['pc_mse'] for r in rows],axis=0),cell['per_pc_velocity_mse'])
    def contrast_terms(row):
        if row['type']=='interaction':
            h=row['head'];return [(h,'gp',1),('mlp','gp',-1),(h,'white',-1),('mlp','white',1)]
        ah,asource=row['a'].split('/');bh,bsource=row['b'].split('/')
        return [(ah,asource,1),(bh,bsource,-1)]
    for row in summary['contrasts']:
        diffs=[sum(sign*records[(row['dataset'],head,source,seed)]['forecast_means'][row['metric']]
                   for head,source,sign in contrast_terms(row)) for seed in cfg['seeds']]
        close(diffs,row['paired_seed_differences']);close(np.mean(diffs),row['mean_difference'])
        close([min(diffs),max(diffs)],row['seed_range'])
    assert len(summary['contrasts'])==104
    for row in summary['context_contrasts']:
        delta=sum(sign*np.mean([records[(row['dataset'],head,source,seed)]['per_context'][row['metric']]
                     for seed in cfg['seeds']],axis=0) for head,source,sign in contrast_terms(row))
        close(delta.mean(),row['mean_difference'])
    ot=summary['ot_only'];assert ot['status']=='complete'
    uni=build_data(cfg['datasets']['etth1_uni']);multi=build_data(cfg['datasets']['etth1_multi'])
    channel=multi.metadata['channel_names'].index('OT');assert channel==ot['multivariate_ot_channel_index']
    for key in ('scaler_mean','scaler_scale'):close(uni.metadata[key][0],multi.metadata[key][channel])
    independent_ot={}
    for row in ot['records']:
        key=(row['dataset'],row['head'],row['source'],row['seed']);run=records[key]
        with np.load(folder/row['dataset']/'samples'/(run['name']+'.npz')) as a:
            ix=a['indices'];x=a['generated']
            np.testing.assert_array_equal(uni.splits['test'].starts[ix],multi.splits['test'].starts[ix])
            y=uni.splits['test'].target[ix]
            close(y,multi.splits['test'].target[ix,:,channel:channel+1])
            if row['dataset']=='etth1_multi':x=x[:,:,:,channel:channel+1]
            scores=independent_scores(x,y)
        for metric,values in scores.items():close(values,row['per_context'][metric])
        values=mean_scores(scores)
        for metric,value in values.items():close(value,row['means'][metric])
        independent_ot[key]=values
    assert len(independent_ot)==48
    ot_scalar_pairs=0
    for cell in ot['aggregate']:
        data=[independent_ot[(cell['dataset'],cell['head'],cell['source'],seed)] for seed in cfg['seeds']]
        for metric,v in cell['metrics'].items():
            close(np.mean([r[metric] for r in data]),v['mean']);close(np.std([r[metric] for r in data],ddof=1),v['sd']);ot_scalar_pairs+=1
    for row in ot['contrasts']:
        d=[independent_ot[('etth1_multi',row['head'],row['source'],seed)][row['metric']]
           -independent_ot[('etth1_uni',row['head'],row['source'],seed)][row['metric']] for seed in cfg['seeds']]
        close(d,row['paired_seed_differences']);close(np.mean(d),row['mean_difference'])
    result={'status':'passed','fingerprint':summary['fingerprint'],'summary_sha256':digest(path),
        'verified_runs':len(records),'aggregate_cells':len(summary['aggregate']),
        'aggregate_metric_mean_sd_pairs':scalar_pairs,'paired_seed_contrasts':len(summary['contrasts']),
        'context_contrast_means':len(summary['context_contrasts']),
        'ot_saved_array_rescorings':len(independent_ot),'ot_aggregate_mean_sd_pairs':ot_scalar_pairs,
        'ot_paired_seed_contrasts':len(ot['contrasts']),'ot_channel_index':channel,
        'scope':'Independent aggregation and paired-difference arithmetic; independent off-diagonal scoring of all48OTarray slices; scaler/origin/target alignment. Does not independently validate bootstrap confidence coverage.',
        'script_sha256':digest(Path(__file__))}
    destination=ROOT/'results/diagnostics/extension_analysis_verification.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
