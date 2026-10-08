"""Independently verify summary aggregation and paired contrast arithmetic."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fm_stress.experiment import save_json


def read(path):return json.loads(path.read_text())
def close(a,b,label):
    if not np.allclose(a,b,rtol=1e-10,atol=1e-11):raise AssertionError(label)

def value(run,key):
    return run['velocity_metrics'][key[len('velocity_'):]] if key.startswith('velocity_') else run['forecast_means'][key]

def independent_interval(delta,block):
    # Circular blocks; write indices directly without using production helpers.
    n=len(delta);rng=np.random.default_rng(9141)
    starts=rng.integers(n,size=(10000,(n+block-1)//block))
    ids=np.concatenate([((starts[:,j,None]+np.arange(block))%n) for j in range(starts.shape[1])],axis=1)[:,:n]
    means=np.asarray(delta)[ids].mean(axis=1)
    return np.percentile(means,[2.5,97.5])


def main():
    path=ROOT/'output/sparse-analysis/summary.json';summary=read(path);cfg=summary['config']
    audit=read(ROOT/'results/sparse-extension/independent_verification.json')
    assert summary['status']=='complete' and summary['observed_new_runs']==264
    assert audit['complete'] and audit['verified_runs']==264 and audit['fingerprint']==summary['fingerprint']
    records={}
    for directory in ('results/extension','results/sparse-extension'):
        for file in (ROOT/directory).glob('*/runs/*.json'):
            r=read(file);records[(file.parent.parent.name,r['head'],r['source'],r['seed'])]=r
    assert len(records)==360
    scalar_checks=activity_checks=0
    assert len(summary['aggregate'])==120
    for row in summary['aggregate']:
        runs=[records[(row['dataset'],row['head'],row['source'],seed)] for seed in cfg['seeds']]
        assert row['n']==3 and row['seeds']==cfg['seeds']
        for key,stats in row['metrics'].items():
            a=np.array([value(r,key) for r in runs]);close([a.mean(),a.std(ddof=1)],[stats['mean'],stats['sd']],f'{row["head"]}/{key}');scalar_checks+=1
        for key,stats in row['activity'].items():
            a=np.array([r['validation_activity'][key] for r in runs]);close([a.mean(),a.std(ddof=1)],[stats['mean'],stats['sd']],'activity');activity_checks+=1
        close(np.mean([r['velocity_metrics']['pc_mse'] for r in runs],axis=0),row['per_pc_velocity_mse'],'PC vector aggregation')
        assert row['best_steps']==[r['best_step'] for r in runs]
        assert row['selected_final_step_count']==sum(r['best_step']==r['steps'] for r in runs)
    for row in summary['contrasts']:
        diffs=np.array([sum(t['weight']*value(records[(row['dataset'],t['head'],t['source'],seed)],row['metric']) for t in row['terms']) for seed in cfg['seeds']])
        close(diffs,row['paired_seed_differences'],'seed contrasts');close(diffs.mean(),row['mean_difference'],'contrast mean')
        close(diffs.std(ddof=1),row['sd_difference'],'contrast SD')
        if 'ci_low' in row:close(independent_interval(diffs,1),[row['ci_low'],row['ci_high']],'seed percentile interval')
    for row in summary['context_contrasts']:
        delta=np.zeros(row['contexts'])
        for t in row['terms']:
            for seed in cfg['seeds']:
                delta+=t['weight']*np.asarray(records[(row['dataset'],t['head'],t['source'],seed)]['per_context'][row['metric']])/len(cfg['seeds'])
        close(delta.mean(),row['mean_difference'],'context contrast mean')
        close(independent_interval(delta,row['block_length']),[row['ci_low'],row['ci_high']],'context block bootstrap')
    final=sum(r['best_step']==r['steps'] for r in records.values() if r['head'] in cfg['heads'])
    assert final==summary['training']['new_checkpoints_at_final_step']
    result={'status':'passed','summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'fingerprint':summary['fingerprint'],
            'checked_runs':360,'aggregate_cells':120,'metric_mean_sd_pairs':scalar_checks,'activity_mean_sd_pairs':activity_checks,
            'seed_contrasts':len(summary['contrasts']),'context_contrasts':len(summary['context_contrasts']),
            'scope':'Independent means/SD, saved run identity, PC vector aggregation, paired seed and context effects, circular-block percentile arithmetic. Statistical coverage assumptions are not established by this check.'}
    save_json(ROOT/'results/diagnostics/sparse_analysis_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
