#!/usr/bin/env python3
"""No-flow negative control for all frozen capacity geometry/source/draw groups."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import check_kernel_scan as a
OUT=ROOT/'results/followup-capacity-baselines'

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def sorted_crps(x,y):
    z=np.sort(np.asarray(x,float),axis=1);n=z.shape[1]
    weights=2*np.arange(1,n+1)-n-1
    half_pair=(z*weights[None,:,None]).sum(axis=1)/(n*(n-1))
    return (np.abs(z-y[:,None,:]).mean(axis=1)-half_pair).mean(axis=1)

def main():
    started=time.time();root=ROOT/'results/followup-capacity';cfg=a.read(root/'config.json');manifest=a.read(root/'manifest.json')
    assert a.stable_hash(cfg)==manifest['config_sha256']
    assert a.stable_hash({k:v for k,v in manifest.items() if k!='fingerprint'})==manifest['fingerprint']
    plan={'name':'zero-vector-field negative control','scope':'All 12 frozen capacity kernel/source/draw groups; no training, checkpoints, selection or tuning. Frozen test data are scored, including all three planned draws.','capacity_config_sha256':a.sha(root/'config.json'),'capacity_manifest_sha256':a.sha(root/'manifest.json'),'script_sha256':a.sha(__file__),'independent_definitions_sha256':a.sha(ROOT/'scripts/check_kernel_scan.py'),'expected_groups':12,'precision':'Velocity truth and PCA use float64; unchanged source paths are cast to float32, matching model-generation initial states.'}
    if (OUT/'config.json').exists():assert a.read(OUT/'config.json')==plan,'Baseline configuration changed'
    save(OUT/'config.json',plan)
    rows=[]
    for kernel in cfg['kernels']:
        geom=cfg['geometries'][0]
        spec={'kernel':kernel,'lookback':geom['lookback'],'horizon':geom['horizon'],'lengthscale':geom['lengthscale'],'variance':cfg['variance'],'nugget':cfg['nugget'],'threshold':cfg['normalization_threshold']}
        for rep in cfg['replicates']:
            ds=rep['data_seed'];train=a.windows(spec,cfg['n_train'],ds);test=a.windows(spec,cfg['n_test'],ds+2)
            for source in cfg['sources']:
                key=f'{kernel}_draw{ds}_{source}';training_source=a.source_draw(spec,len(train.target),source,ds+100)
                training_velocity=train.target-training_source;values,q=a.principal_axes(training_velocity)
                xt,u,t,eps=a.pairs(spec,test.target,source,ds,'test')
                nc,ns=cfg['n_ensemble_contexts'],cfg['ensemble_size'];es=a.source_draw(spec,nc*ns,source,ds+1000).reshape(nc,ns,-1)
                forecasts=es.astype(np.float32);paths=eps[:cfg['n_paths']].astype(np.float32)
                raw_pc=np.mean((u@q.T)**2,axis=0)
                gr,tr=a.roughness(paths),a.roughness(test.target[:len(paths)])
                scores=a.forecast_scores(forecasts,test.target[:nc]);alternative=sorted_crps(forecasts,test.target[:nc])
                diff=float(np.max(np.abs(scores['fair_crps']-alternative)))
                assert np.allclose(scores['fair_crps'],alternative,rtol=1e-12,atol=1e-12),'Direct-pair and sorted CRPS disagree'
                pc1,res=float(raw_pc[0]),float(raw_pc[2:].mean());flags=a.literal_rule(pc1,res,gr,tr)
                hashes={k:a.array_sha(v) for k,v in [('raw_history',test.raw_context),('raw_target',test.raw_target),('source',eps),('times',t),('ensemble_source',es)]}
                # Every existing run must use exactly these sources/history, regardless of head.
                runfolder=root/kernel/geom['id']/f'draw{ds}'/'runs';matched=[]
                for rp in sorted(runfolder.glob(source+'_*budget12000.json')):
                    r=a.read(rp);assert hashes==r['test_input_hashes'];matched.append({'path':str(rp.relative_to(ROOT)),'sha256':a.sha(rp)})
                samplepath=OUT/'samples'/(key+'.npz');samplepath.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(samplepath,ensemble=forecasts,generated=paths,target=test.target[:nc],history=test.context[:nc],pca_components=q,pca_eigenvalues=values,velocity_pc_mse=raw_pc,**{f'score_{k}':v for k,v in scores.items()})
                record={'identity':[kernel,ds,source],'baseline':'v(x,t,h)=0; forecast=source; ignores history','scope':plan['scope'],'geometry':spec,'train_rows':len(train.target),'test_velocity_rows':len(test.target),'contexts':nc,'members':ns,'input_hashes':hashes,'train_velocity_sha256':a.array_sha(training_velocity),'pca_sha256':a.array_sha(q),'source_covariance_sha256':a.array_sha(a.source_cov(spec,source)),'capacity_manifest_sha256':plan['capacity_manifest_sha256'],'script_sha256':plan['script_sha256'],'samples_sha256':a.sha(samplepath),'matched_existing_runs':matched,'metrics':{'pc1_mse':pc1,'pc2_mse':float(raw_pc[1]),'residual_mse':res,'pc_mse':raw_pc.tolist(),'generated_roughness':gr,'target_roughness':tr,'roughness_ratio':gr/tr,'velocity_rule_pass':flags[0],'roughness_rule_pass':flags[1],'works':flags[2]},'forecast_means':{k:float(v.mean()) for k,v in scores.items()},'per_context':{k:v.tolist() for k,v in scores.items()},'direct_vs_sorted_crps_max_absolute_difference':diff}
                save(OUT/'runs'/(key+'.json'),record);rows.append(record);print('BASELINE',key,flush=True)
    expected={(k,r['data_seed'],s) for k in cfg['kernels'] for r in cfg['replicates'] for s in cfg['sources']}
    assert len(rows)==12 and {tuple(r['identity']) for r in rows}==expected
    cells=[]
    for kernel in cfg['kernels']:
        for source in cfg['sources']:
            rr=[r for r in rows if r['identity'][0]==kernel and r['identity'][2]==source]
            cells.append({'kernel':kernel,'source':source,'draws':[r['identity'][1] for r in rr],'metrics':{key:float(np.mean([r['metrics'][key] for r in rr])) for key in ['pc1_mse','residual_mse','roughness_ratio']},'forecast_means':{key:float(np.mean([r['forecast_means'][key] for r in rr])) for key in scores},'literal_successes':sum(r['metrics']['works'] for r in rr)})
    result={'complete':True,'groups':12,'scope':plan['scope'],'config_sha256':a.sha(OUT/'config.json'),'elapsed_seconds':time.time()-started,'max_crps_formula_disagreement':max(r['direct_vs_sorted_crps_max_absolute_difference'] for r in rows),'cells':cells,'record_hashes':{str(p.relative_to(ROOT)):a.sha(p) for p in sorted((OUT/'runs').glob('*.json'))}}
    save(OUT/'summary.json',result)
    lines=['# No-flow negative control','','This checkpoint-free baseline sets the velocity field to zero. Every forecast equals its initial source and ignores the observed history. It is a diagnostic of source-only apparent success, not a competing learned model. All 12 frozen capacity kernel/source/draw groups are scored; no result selects settings or data. The third test draw is accessed explicitly as part of this fixed scoring plan.','','Metrics below average three independent draws. Raw velocity metrics use training-only, source-specific velocity PCA; forecast metrics use identical 256 contexts and 32 source draws per context. Compare raw velocity errors within a source, since source noise floors differ.','','| Target / source | PC1 MSE | Residual MSE | Roughness ratio | Fair CRPS | 90% coverage | Literal successes |','|---|---:|---:|---:|---:|---:|---:|']
    for r in cells:
        m,f=r['metrics'],r['forecast_means'];lines.append(f"| {r['kernel']} / {r['source']} | {m['pc1_mse']:.6f} | {m['residual_mse']:.6f} | {m['roughness_ratio']:.4f} | {f['fair_crps']:.6f} | {f['coverage_90']:.4f} | {r['literal_successes']}/3 |")
    lines += ['','Direct O(S²) pairwise fair CRPS independently agrees with the sorted-order formula to a maximum absolute difference of '+f"{result['max_crps_formula_disagreement']:.3g}. All individual results, input hashes, source covariance/PCA hashes, generated arrays and per-context proper/calibration scores are retained under `results/followup-capacity-baselines/`. Original capacity results and fit counts are unchanged.",'']
    (ROOT/'docs/followup_no_flow_baseline.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
