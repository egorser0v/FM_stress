#!/usr/bin/env python3
"""Post-hoc full-ensemble Euler refinement, no training or model selection."""
import argparse, itertools, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import check_kernel_scan as audit
from fm_stress.models import create_model
from fm_stress.experiment import CoordinateModel
CFG=ROOT/'configs/followup_solver.json'
OUT=ROOT/'results/followup-solver'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def require(ok,message):
    if not ok:raise ValueError(message)

def synchronize(device):
    if device=='mps':torch.mps.synchronize()

@torch.no_grad()
def integrate(model,source,history,steps,batch_size,device):
    result=[]
    for start in range(0,len(source),batch_size):
        x=torch.tensor(source[start:start+batch_size],dtype=torch.float32,device=device)
        h=torch.tensor(history[start:start+batch_size],dtype=torch.float32,device=device)
        for k in range(steps):
            v=model(x,torch.full((len(x),),k/steps,dtype=torch.float32,device=device),h)
            x=x+v/steps
        require(bool(torch.isfinite(x).all()),'Nonfinite integration state')
        result.append(x.cpu().numpy())
    return np.concatenate(result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--limit',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    c=audit.read(CFG);root=ROOT/c['capacity_root'];cfg=audit.read(root/'config.json');manifest=audit.read(root/'manifest.json')
    require(audit.stable_hash(cfg)==manifest['config_sha256'],'Capacity config hash')
    require(audit.stable_hash({k:v for k,v in manifest.items() if k!='fingerprint'})==manifest['fingerprint'],'Capacity manifest fingerprint')
    for name,digest in manifest['source_hashes'].items():
        original=ROOT/name
        if name=='docs/followup_capacity_protocol.md' and audit.sha(original)!=digest:
            original=root/'frozen_protocol.md'
        require(original.exists() and audit.sha(original)==digest,'Capacity frozen source changed: '+name)
    require(c['selected_budget']==12000 and c['coarse_steps']==64 and c['fine_steps']==128,'Fixed solver design changed')
    sourcefiles=['scripts/audit_followup_solver.py','scripts/check_kernel_scan.py','fm_stress/models.py','fm_stress/experiment.py','docs/followup_solver_protocol.md']
    ownmanifest={'config_sha256':audit.sha(CFG),'capacity_config_sha256':audit.sha(root/'config.json'),'capacity_manifest_sha256':audit.sha(root/'manifest.json'),'source_hashes':{f:audit.sha(ROOT/f) for f in sourcefiles},'scope':c['scope']}
    ownmanifest['fingerprint']=audit.stable_hash(ownmanifest)
    if a.prepare:
        require(not (OUT/'manifest.json').exists() or audit.read(OUT/'manifest.json')==ownmanifest,'Existing frozen solver manifest differs')
        save(OUT/'config.json',c);save(OUT/'manifest.json',ownmanifest);print('FROZEN',ownmanifest['fingerprint']);return
    require((OUT/'manifest.json').exists() and audit.read(OUT/'manifest.json')==ownmanifest,'Prepare/freeze solver manifest first; numerical source changed')
    require(audit.read(OUT/'config.json')==c,'Solver config changed')
    if a.device=='mps':
        require(torch.backends.mps.is_available(),'MPS unavailable')
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1','Silent MPS CPU fallback forbidden')
    require(not a.smoke or a.device=='cpu','Smoke must use CPU')
    expected=[]
    for kernel,heads in cfg['kernel_heads'].items():
        for rep in cfg['replicates']:
            for source in cfg['sources']:
                for head in heads:expected.append((kernel,rep['data_seed'],source,head))
    require(len(expected)==c['expected_models'],'Expected model count')
    # Smoke selection is fixed by configuration, independent of forecast outcomes.
    if a.smoke:expected=[tuple(v) for v in c['smoke_identities']]
    if a.limit:expected=expected[:a.limit]
    cache={};rows=[];begin=time.time()
    for kernel,ds,source,head in expected:
        run=root/kernel/'F64_l8'/f'draw{ds}'/'runs'/f'{source}_{head}_budget12000.json'
        require(run.exists(),'Missing planned capacity run: '+str(run))
        r=audit.read(run);gx=run.parent.parent;ck=gx/'checkpoints'/(run.stem+'.pt');sp=gx/'samples'/(run.stem+'.npz')
        require((r['kernel'],r['data_seed'],r['source'],r['head'],r['budget'])==(kernel,ds,source,head,12000) and r['fingerprint']==manifest['fingerprint'],'Run identity/fingerprint')
        require(audit.sha(ck)==r['checkpoint_sha256'] and audit.sha(sp)==r['samples_sha256'],'Checkpoint or saved ensemble hash')
        identity=f'{kernel}_draw{ds}_{source}_{head}';dest=OUT/('smoke' if a.smoke else 'runs')/(identity+'.json')
        inputs={'run_sha256':audit.sha(run),'checkpoint_sha256':audit.sha(ck),'samples64_sha256':audit.sha(sp)}
        if dest.exists():
            old=audit.read(dest)
            require(old['fingerprint']==ownmanifest['fingerprint'] and old['inputs']==inputs,'Stale solver cache')
            require(audit.sha(ROOT/old['samples_path'])==old['samples128_sha256'],'Cached arrays changed')
            rows.append(old);continue
        g=r['geometry'];spec={**g,'threshold':g['normalization_threshold']};key=(kernel,ds)
        if key not in cache:cache[key]=audit.windows(spec,cfg['n_test'],ds+2)
        test=cache[key];nc,ns=cfg['n_ensemble_contexts'],cfg['ensemble_size']
        eps=audit.source_draw(spec,nc*ns,source,ds+1000).reshape(nc,ns,-1)
        _,_,times,pairsource=audit.pairs(spec,test.target,source,ds,'test')
        rebuilt={k:audit.array_sha(v) for k,v in [('raw_history',test.raw_context),('raw_target',test.raw_target),('source',pairsource),('times',times),('ensemble_source',eps)]}
        require(rebuilt==r['test_input_hashes'],'Independently reconstructed input hashes differ')
        with np.load(sp) as z:
            old=np.array(z['ensemble']);target=np.array(z['ensemble_target'])
        require(old.shape==(nc,ns,g['horizon']),'Saved ensemble shape')
        require(np.allclose(target,test.target[:nc],rtol=1e-10,atol=1e-10),'Saved target differs')
        n=c['smoke_contexts'] if a.smoke else nc
        sourcearr=eps[:n];hist=np.repeat(test.context[:n],ns,axis=0);y=test.target[:n];base=old[:n]
        arch=r['architecture'];model=CoordinateModel(create_model(arch['family'],horizon=g['horizon'],lookback=g['lookback'],**{**cfg['model'],'width':arch['width'],'depth':arch['depth']}))
        checkpoint=torch.load(ck,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict']);model=model.to(a.device).eval()
        require(checkpoint['step']==r['best_step'],'Selected step mismatch')
        require(all(checkpoint[k]==r[k] for k in ['kernel','data_seed','seed','source','head','budget','fingerprint']),'Checkpoint identity mismatch')
        require(checkpoint['config']==cfg,'Checkpoint configuration mismatch')
        nr=min(c['replay_contexts'],n);synchronize(a.device);start=time.perf_counter()
        replay=integrate(model,sourcearr[:nr].reshape(-1,g['horizon']),hist[:nr*ns],c['coarse_steps'],cfg['generation_batch_size'],a.device).reshape(nr,ns,-1)
        replayrms=float(np.sqrt(np.mean((replay.astype(float)-base[:nr])**2)));replayscale=float(np.sqrt(np.mean(base[:nr].astype(float)**2)))
        require(replayrms<=c['relative_replay_tolerance']*max(replayscale,1e-3),'Euler64 replay outside tolerance')
        synchronize(a.device);fine_start=time.perf_counter()
        fine=integrate(model,sourcearr.reshape(-1,g['horizon']),hist,c['fine_steps'],cfg['generation_batch_size'],a.device).reshape(n,ns,-1)
        synchronize(a.device);fine_seconds=time.perf_counter()-fine_start
        coarse_scores=audit.forecast_scores(base,y);fine_scores=audit.forecast_scores(fine,y)
        if not a.smoke:
            for metric,v in coarse_scores.items():require(np.allclose(v,r['per_context'][metric],rtol=1e-6,atol=1e-9),'Saved coarse score mismatch '+metric)
        scores={};arrays={'ensemble64':base,'ensemble128':fine,'source':sourcearr,'history':test.context[:n],'target':y}
        for metric in coarse_scores:
            before,after=coarse_scores[metric],fine_scores[metric];delta=after-before
            scores[metric]={'mean64':float(before.mean()),'mean128':float(after.mean()),'delta128_minus64':float(delta.mean()),'max_context_abs_delta':float(np.abs(delta).max())}
            arrays['score64_'+metric]=before;arrays['score128_'+metric]=after;arrays['delta_'+metric]=delta
        samplepath=OUT/('smoke' if a.smoke else 'samples')/(identity+'.npz');samplepath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(samplepath,**arrays)
        record={'identity':[kernel,ds,source,head],'budget':12000,'selected_step':r['best_step'],'fingerprint':ownmanifest['fingerprint'],'inputs':inputs,'input_hashes':rebuilt,'contexts':n,'members':ns,'steps':[64,128],'device':a.device,'torch':torch.__version__,'platform':platform.platform(),'cpu_threads':torch.get_num_threads(),'replay_contexts':nr,'replay64_rms':replayrms,'replay64_relative_rms':replayrms/max(replayscale,1e-3),'fine_integration_seconds':fine_seconds,'elapsed_seconds':time.perf_counter()-start,'endpoint_rmse':float(np.sqrt(np.square(fine.astype(float)-base).mean())),'roughness64':audit.roughness(base),'roughness128':audit.roughness(fine),'target_roughness':audit.roughness(y),'scores':scores,'samples_path':str(samplepath.relative_to(ROOT)),'samples128_sha256':audit.sha(samplepath)}
        save(dest,record);rows.append(record);print('VERIFIED',identity,n,'contexts',round(fine_seconds,2),'seconds',flush=True)
        del model
        if a.device=='mps':torch.mps.empty_cache()
    # Keep cell and draw scores paired. S4 is a reference, not selected winner.
    contrasts=[];lookup={tuple(r['identity']):r for r in rows}
    for kernel,heads in cfg['kernel_heads'].items():
        for source in cfg['sources']:
            for head in heads:
                if head=='s4':continue
                per=[]
                for rep in cfg['replicates']:
                    ds=rep['data_seed'];left=lookup.get((kernel,ds,source,head));right=lookup.get((kernel,ds,source,'s4'))
                    if left is None or right is None:continue
                    b=left['scores']['fair_crps']['mean64']-right['scores']['fair_crps']['mean64'];f=left['scores']['fair_crps']['mean128']-right['scores']['fair_crps']['mean128']
                    per.append({'draw':ds,'head_minus_s4_64':b,'head_minus_s4_128':f,'contrast_change':f-b,'sign_flipped':bool(np.sign(b)!=np.sign(f))})
                if per:
                    b=float(np.mean([v['head_minus_s4_64'] for v in per]));f=float(np.mean([v['head_minus_s4_128'] for v in per]))
                    contrasts.append({'kernel':kernel,'source':source,'head':head,'reference':'s4','draws':per,'mean_head_minus_s4_64':b,'mean_head_minus_s4_128':f,'mean_contrast_change':f-b,'mean_sign_flipped':bool(np.sign(b)!=np.sign(f))})
    complete=not a.smoke and len(rows)==c['expected_models'] and all(r['contexts']==256 and r['members']==32 for r in rows)
    save(OUT/('smoke_summary.json' if a.smoke else 'summary.json'),{'complete':complete,'scope':c['scope'],'fingerprint':ownmanifest['fingerprint'],'models':len(rows),'expected_models':c['expected_models'],'elapsed_seconds_this_invocation':time.time()-begin,'records':rows,'paired_head_contrasts':contrasts,'interpretation':'Euler64 versus Euler128 is a sensitivity check, not proof of exact integration. Negative head-minus-S4 CRPS favors the named head. Same contexts, ensemble sources and frozen checkpoints; no refitting or selection.'})
    print('COMPLETE',complete,'MODELS',len(rows),flush=True)
if __name__=='__main__':main()
