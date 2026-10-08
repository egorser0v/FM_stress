"""Independent arithmetic audit of the 64/192 dictionary supplement."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_sparse_analysis import close,independent_interval,value
from fm_stress.experiment import save_json


def main():
    path=ROOT/'output/dictionary-supplement/summary.json';s=json.loads(path.read_text());records={}
    assert s['status']=='complete' and s['rank_diagnostics']['status']=='verified'
    for atoms,folder in [(64,'sparse-extension'),(192,'sparse-fullrank-dictionary')]:
        for d in ['etth1_multi','exchange_multi']:
            for f in (ROOT/'results'/folder/d/'runs').glob('*_dictionary_*_seed*.json'):
                r=json.loads(f.read_text());records[(d,atoms,r['head'],r['source'],r['seed'])]=r
    assert len(records)==48 and len(s['aggregate'])==16
    count=0
    for row in s['aggregate']:
        runs=[records[(row['dataset'],row['atoms'],row['head'],row['source'],seed)] for seed in [31,32,33]]
        for m,v in row['metrics'].items():
            a=np.array([value(r,m) for r in runs]);close([a.mean(),a.std(ddof=1)],[v['mean'],v['sd']],'supplement aggregation');count+=1
    for row in s['seed_contrasts']:
        diff=np.array([sum(t['weight']*value(records[(row['dataset'],t['atoms'],t['head'],row['source'],seed)],row['metric']) for t in row['terms']) for seed in [31,32,33]])
        close(diff,row['paired_seed_differences'],'supplement seed effects');close([diff.mean(),diff.std(ddof=1)],[row['mean_difference'],row['sd_difference']],'supplement seed summary')
    for row in s['context_contrasts']:
        diff=np.zeros(row['contexts'])
        for t in row['terms']:
            for seed in [31,32,33]:diff+=t['weight']*np.asarray(records[(row['dataset'],t['atoms'],t['head'],row['source'],seed)]['per_context'][row['metric']])/3
        close(diff.mean(),row['mean_difference'],'supplement context effects');close(independent_interval(diff,row['block_length']),[row['ci_low'],row['ci_high']],'supplement confidence intervals')
    result={'status':'passed','summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runs':48,'aggregate_cells':16,'metric_mean_sd_pairs':count,'seed_contrasts':len(s['seed_contrasts']),'context_contrasts':len(s['context_contrasts'])}
    save_json(ROOT/'results/diagnostics/dictionary_analysis_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
