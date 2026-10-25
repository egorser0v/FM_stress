#!/usr/bin/env python3
"""Evaluate all primary checkpoints against an information-matched Bayes field."""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fm_stress.normalized_history_oracle import NormalizedHistoryOracle
from fm_stress.data import GPConfig, kernel_covariance, source_covariance, sample_dataset, conditional_target, conditional_oracle_velocity, conditional_oracle_noise
from fm_stress.experiment import CoordinateModel, save_json
from fm_stress.models import create_model

OUT=ROOT/'results/followup-oracle'
FIG=ROOT/'output/followup-oracle'
CONFIG={'n_jacobian':256,'times':[.1,.3,.5,.7,.9], 'jacobian_source_seed':424242,
        'n_fresh':2048,'fresh_data_seed':9251041,'fresh_source_seed':9251042,'fresh_time_seed':9251043,
        'quadrature_orders':[32,64,128],'max_order':256,'relative_tolerance':1e-4,
        'secondary_added_after_initial_probes':'covariance-weighted local Jacobian discrepancy; exploratory',
        'device':'cpu','torch_threads':2,'status':'supplementary post-hoc diagnostic; no training'}

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,x): save_json(path,x)
def rms(x): return float(np.sqrt(np.mean(np.square(x))))

def checked_flow(o,x,t,h,noise=False):
    outputs={q:o.flow(x,t,h,order=q,noise_covariance=noise) for q in CONFIG['quadrature_orders']}
    checks=[]
    for lo,hi in [(32,64),(64,128)]:
        checks.append({'low':lo,'high':hi,**{k+'_relative_rms':rms(outputs[lo][k]-outputs[hi][k])/max(rms(outputs[hi][k]),1e-14) for k in ['velocity','jacobian']},
                       'velocity_max_abs':float(np.max(np.abs(outputs[lo]['velocity']-outputs[hi]['velocity']))),
                       'jacobian_max_abs':float(np.max(np.abs(outputs[lo]['jacobian']-outputs[hi]['jacobian'])))})
    hi=128
    if max(checks[-1]['velocity_relative_rms'],checks[-1]['jacobian_relative_rms'])>CONFIG['relative_tolerance']:
        hi=256; outputs[hi]=o.flow(x,t,h,order=hi,noise_covariance=noise)
        checks.append({'low':128,'high':256,**{k+'_relative_rms':rms(outputs[128][k]-outputs[256][k])/max(rms(outputs[256][k]),1e-14) for k in ['velocity','jacobian']},
                       'velocity_max_abs':float(np.max(np.abs(outputs[128]['velocity']-outputs[256]['velocity']))),
                       'jacobian_max_abs':float(np.max(np.abs(outputs[128]['jacobian']-outputs[256]['jacobian'])))})
    diag={'steps':checks,'chosen_order':hi,'converged':max(checks[-1]['velocity_relative_rms'],checks[-1]['jacobian_relative_rms'])<=CONFIG['relative_tolerance'],
          'max_threshold_tail_probability':float(outputs[hi]['threshold_tail_probability'].max()),
          'small_scale_count':int(outputs[hi]['small_scale_branch'].sum()),'truncated_rule_count':int(outputs[hi]['truncated_quantile_rule'].sum()),
          'min_quadrature_weight_ess':float(outputs[hi]['weight_ess'].min())}
    return outputs[hi],diag

def jac(model,x,t,h):
    x=torch.tensor(x,dtype=torch.float32,requires_grad=True)
    v=model(x,torch.full((len(x),),float(t)),torch.tensor(h,dtype=torch.float32))
    J=torch.stack([torch.autograd.grad(v[:,i].sum(),x,retain_graph=True)[0] for i in range(v.shape[1])],1)
    return v.detach().numpy(),J.detach().numpy()

def jac_metrics(J,B,Q,input_covariance=None):
    out={}
    Arot=Q@J@Q.T; Brot=Q@B@Q.T
    for key,sl in [('full',slice(None)),('main',slice(0,2)),('residual',slice(2,None))]:
        A=Arot[:,sl,sl]; ref=Brot[:,sl,sl]; D=A-ref
        dn=np.linalg.norm(D,axis=(1,2)); bn=np.linalg.norm(ref,axis=(1,2)); an=np.linalg.norm(A,axis=(1,2))
        out[key]={'mean_relative_error':float(np.mean(dn/np.maximum(bn,1e-12))),
                  'pooled_relative_error':rms(D)/max(rms(ref),1e-12),'absolute_rms':rms(D),'reference_rms':rms(ref),
                  'mean_norm_ratio':float(np.mean(an/np.maximum(bn,1e-12))),
                  'mean_cosine':float(np.mean((A*ref).sum((1,2))/np.maximum(an*bn,1e-12))),
                  'reference_min_norm':float(bn.min())}
    if input_covariance is not None:
        Crot=Q@input_covariance@Q.T
        for key,sl in [('full',slice(None)),('main',slice(0,2)),('residual',slice(2,None))]:
            A=Arot[:,sl,sl]; ref=Brot[:,sl,sl]; D=A-ref; C=Crot[:,sl,sl]
            err=float(np.einsum('nij,njk,nik->n',D,C,D).mean())
            scale=float(np.einsum('nij,njk,nik->n',ref,C,ref).mean())
            out[key]['covariance_weighted_relative_error']=float(np.sqrt(max(err,0)/max(scale,1e-24)))
            out[key]['covariance_weighted_error_energy']=err
            out[key]['covariance_weighted_reference_energy']=scale
    out['gain_per_pc']=np.mean((Arot*Brot).sum(1)/np.maximum((Brot*Brot).sum(1),1e-24),0).tolist()
    return out

def mse_groups(error,Q):
    p=np.square(error@Q.T).mean(0)
    return {'overall':float(np.square(error).mean()),'pc1':float(p[0]),'pc2':float(p[1]),'residual':float(p[2:].mean()),'tail_pc7_16':float(p[6:].mean()),'per_pc':p.tolist()}

def velocity_metrics(pred,u,oracle,noise,Q):
    d=pred-oracle; r=oracle-u
    p=d@Q.T; q=r@Q.T
    cross=2*(p*q).mean(0)
    noise_pc=np.einsum('ij,njk,ik->ni',Q,noise,Q).mean(0)
    return {'label_mse':mse_groups(pred-u,Q),'oracle_discrepancy':mse_groups(d,Q),'oracle_label_mse':mse_groups(r,Q),
            'signed_cross_term_per_pc':cross.tolist(),'expected_noise_per_pc':noise_pc.tolist(),
            'decomposition_max_abs':float(np.max(np.abs(np.square((pred-u)@Q.T).mean(0)-np.square(p).mean(0)-np.square(q).mean(0)-cross)))}

def main():
    started=time.time(); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    mainroot=ROOT/'results/main'; cfg=json.loads((mainroot/'config.json').read_text()); geo=json.loads((mainroot/'geometry.json').read_text())
    cells=[(s,k,seed) for s,k in [('white','mlp'),('white','s4'),('gp','mlp'),('gp','s4'),('gp','whitened_mlp')] for seed in cfg['seeds']]
    files=[ROOT/'fm_stress/normalized_history_oracle.py',Path(__file__),ROOT/'docs/followup_oracle_protocol.md',ROOT/'fm_stress/data.py',ROOT/'fm_stress/models.py',ROOT/'fm_stress/experiment.py',mainroot/'config.json',mainroot/'geometry.json']
    files += [mainroot/'checkpoints'/f'{s}_{k}_seed{seed}.pt' for s,k,seed in cells]
    manifest={'config':CONFIG,'sha256':{str(p.relative_to(ROOT)):digest(p) for p in files},'created_before_probes':True}
    dump(OUT/'config.json',CONFIG); dump(OUT/'manifest.json',manifest)
    torch.set_num_threads(CONFIG['torch_threads'])
    gp=GPConfig(**cfg['data']); old=sample_dataset(gp,cfg['n_test'],cfg['data_seed']+2)
    fresh=sample_dataset(gp,CONFIG['n_fresh'],CONFIG['fresh_data_seed'])
    n=CONFIG['n_jacobian']; h=old.context[:n]; y=old.target[:n]
    pm,pc=conditional_target(gp,old); pm=pm[:n]; pc=pc[:n]
    fm,fc=conditional_target(gp,fresh)
    refs={}; diagnostics={}; rows=[]; velrows=[]
    for source in ['white','gp']:
        cs=source_covariance(gp,source)
        o=NormalizedHistoryOracle(kernel_covariance(gp),gp.lookback,cs,gp.normalization_threshold)
        Q=np.asarray(geo['pca'][source]['components'])
        eps=np.random.default_rng(CONFIG['jacobian_source_seed']).normal(size=y.shape)@np.linalg.cholesky(cs).T
        source_refs=[]; source_diag=[]
        _,matched_target_cov=o.target_moments(h)
        for t in CONFIG['times']:
            x=(1-t)*eps+t*y; got,diag=checked_flow(o,x,t,h,noise=True)
            S=t*t*pc+(1-t)**2*cs; U=t*pc-(1-t)*cs
            Jp=np.linalg.solve(S.swapaxes(-1,-2),U.swapaxes(-1,-2)).swapaxes(-1,-2)
            vp=conditional_oracle_velocity(x,t,pm,pc,cs)
            source_refs.append({'x':x,'matched':got,'privileged_jacobian':Jp,'privileged_velocity':vp, 'input_covariance':t*t*matched_target_cov+(1-t)**2*cs})
            diag['t']=t;diag['matched_vs_privileged']=jac_metrics(got['jacobian'],Jp,Q)
            diag['posterior_weight_derivative_relative_rms']=rms(got['jacobian']-got['component_average_jacobian'])/rms(got['jacobian'])
            # Independent central differences of the implemented conditional velocity.
            sample=8; step=1e-5; num=[]
            for j in range(gp.horizon):
                dx=np.zeros_like(x[:sample]); dx[:,j]=step
                a=o.flow(x[:sample]+dx,t,h[:sample],order=diag['chosen_order'],jacobian=False)['velocity']
                b=o.flow(x[:sample]-dx,t,h[:sample],order=diag['chosen_order'],jacobian=False)['velocity']
                num.append((a-b)/(2*step))
            num=np.stack(num,2)
            diag['finite_difference_relative_rms']=rms(num-got['jacobian'][:sample])/rms(got['jacobian'][:sample])
            source_diag.append(diag)
        f_eps=np.random.default_rng(CONFIG['fresh_source_seed']).normal(size=fresh.target.shape)@np.linalg.cholesky(cs).T
        ft=np.random.default_rng(CONFIG['fresh_time_seed']).uniform(0,1,len(fresh))
        fx=(1-ft[:,None])*f_eps+ft[:,None]*fresh.target; fu=fresh.target-f_eps
        fref,fdiag=checked_flow(o,fx,ft,fresh.context,noise=True)
        fp=conditional_oracle_velocity(fx,ft,fm,fc,cs)
        # Privileged conditional label noise, calculated explicitly at every probe.
        S=ft[:,None,None]**2*fc+(1-ft[:,None,None])**2*cs
        U=ft[:,None,None]*fc-(1-ft[:,None,None])*cs
        fnoise=fc+cs-U@np.linalg.solve(S,U.swapaxes(-1,-2))
        matched_mean,matched_cov=o.target_moments(fresh.context)
        fdiag['conditional_target_mean_privileged_discrepancy_mse']=float(np.square(matched_mean-fm).mean())
        fdiag['mean_conditional_target_covariance_trace_matched']=float(np.trace(matched_cov,axis1=1,axis2=2).mean())
        fdiag['mean_conditional_target_covariance_trace_privileged']=float(np.trace(fc,axis1=1,axis2=2).mean())
        fdiag['matched_vs_privileged_velocity_mse']=float(np.square(fref['velocity']-fp).mean())
        # Conditional target simulation: sampling from the derived law, independent MC.
        sim=o.sample_target(h[:4],40000,99117); sm,sc=o.target_moments(h[:4])
        fdiag['conditional_sampler_mean_max_abs_error']=float(np.max(np.abs(sim.mean(1)-sm)))
        fdiag['conditional_sampler_covariance_relative_rms']=rms(np.array([np.cov(v,rowvar=False) for v in sim])-sc)/rms(sc)
        refs[source]={'old':source_refs,'fresh':{'x':fx,'t':ft,'u':fu,'matched':fref,'privileged':fp,'privileged_noise':fnoise},'Q':Q}
        diagnostics[source]={'jacobian_probes':source_diag,'fresh':fdiag}
        np.savez_compressed(OUT/f'oracle_{source}.npz',history=h,target=y,source=eps,times=CONFIG['times'],
             matched_velocity=np.stack([r['matched']['velocity'] for r in source_refs]),
             matched_jacobian=np.stack([r['matched']['jacobian'] for r in source_refs]),
             privileged_velocity=np.stack([r['privileged_velocity'] for r in source_refs]),
             privileged_jacobian=np.stack([r['privileged_jacobian'] for r in source_refs]),
             fresh_x=fx,fresh_time=ft,fresh_history=fresh.context,fresh_label=fu,
             fresh_matched_velocity=fref['velocity'],fresh_privileged_velocity=fp,
             fresh_matched_noise=fref['noise_covariance'],fresh_privileged_noise=fnoise)
        print('oracle',source,'complete',flush=True)
    for source,kind,seed in cells:
        name=f'{source}_{kind}_seed{seed}'
        state=torch.load(mainroot/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=False)
        m=CoordinateModel(create_model('s4' if kind=='s4' else 'mlp',horizon=gp.horizon,lookback=gp.lookback,**cfg['model']))
        m.load_state_dict(state['state_dict']); m.eval()
        Q=refs[source]['Q']; jacobians=[]; predictions=[]
        for t,r in zip(CONFIG['times'],refs[source]['old']):
            pred,J=jac(m,r['x'],t,h); jacobians.append(J); predictions.append(pred)
            rows.append({'source':source,'kind':kind,'seed':seed,'time':t,
                         'matched':jac_metrics(J,r['matched']['jacobian'],Q,r['input_covariance']),
                         'privileged':jac_metrics(J,r['privileged_jacobian'],Q,r['input_covariance'])})
        f=refs[source]['fresh']
        with torch.no_grad():
            pred=m(torch.tensor(f['x'],dtype=torch.float32),torch.tensor(f['t'],dtype=torch.float32),torch.tensor(fresh.context,dtype=torch.float32)).numpy()
        velrows.append({'source':source,'kind':kind,'seed':seed,
                        'matched':velocity_metrics(pred,f['u'],f['matched']['velocity'],f['matched']['noise_covariance'],Q),
                        'privileged':velocity_metrics(pred,f['u'],f['privileged'],f['privileged_noise'],Q)})
        np.savez_compressed(OUT/f'model_{name}.npz',jacobian=np.stack(jacobians),prediction=np.stack(predictions),fresh_prediction=pred)
        print(name,'complete',flush=True)
    summary={}
    for source,kind in [('white','mlp'),('white','s4'),('gp','mlp'),('gp','s4'),('gp','whitened_mlp')]:
        rr=[r for r in rows if r['source']==source and r['kind']==kind]
        vv=[r for r in velrows if r['source']==source and r['kind']==kind]
        entry={}
        for ref in ['matched','privileged']:
            entry[ref]={}
            for block in ['full','main','residual']:
                entry[ref][block]={key:float(np.mean([r[ref][block][key] for r in rr])) for key in rr[0][ref][block]}
            entry[ref]['residual_relative_error_per_seed']=[float(np.mean([r[ref]['residual']['mean_relative_error'] for r in rr if r['seed']==s])) for s in cfg['seeds']]
            entry[ref]['gain_per_pc']=np.mean([r[ref]['gain_per_pc'] for r in rr],0).tolist()
            entry[ref]['fresh_velocity']={key:{g:float(np.mean([r[ref][key][g] for r in vv])) for g in ['overall','pc1','pc2','residual','tail_pc7_16']} for key in ['label_mse','oracle_discrepancy','oracle_label_mse']}
        summary[f'{source}|{kind}']=entry
    converged=all(d['converged'] for s in diagnostics.values() for d in s['jacobian_probes']+[s['fresh']])
    result={'config':CONFIG,'numerical_diagnostics':diagnostics,'rows':rows,'velocity_rows':velrows,'summary':summary,
            'all_quadratures_converged':converged,'checkpoint_count':len(cells),'elapsed_seconds':time.time()-started,
            'source_hashes_unchanged':all(digest(ROOT/p)==v for p,v in manifest['sha256'].items())}
    dump(OUT/'summary.json',result)
    plot(result)
    print(json.dumps({'all_quadratures_converged':converged,'elapsed_seconds':result['elapsed_seconds'],'summary':summary},indent=2),flush=True)

def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'mlp':'#2867a5','s4':'#d5653e','whitened_mlp':'#338875'}
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    for ax,source in zip(axs,['white','gp']):
        for kind in ['mlp','s4','whitened_mlp']:
            k=f'{source}|{kind}'
            if k not in result['summary']:continue
            vals=[result['summary'][k][r]['residual']['mean_relative_error'] for r in ['privileged','matched']]
            ax.plot([0,1],vals,'o-',color=colors[kind],label=kind.replace('_',' '),linewidth=2)
        ax.set_xticks([0,1],['Raw-history reference','Normalized-history reference'])
        ax.set_title('White source' if source=='white' else 'GP source',loc='left')
        ax.set_ylabel('Residual-block relative Jacobian error')
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle('Does matching the oracle’s information change the head comparison?',x=.02,ha='left')
    fig.tight_layout();fig.savefig(FIG/'oracle_information_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,source in zip(axs,['white','gp']):
        for kind in ['mlp','s4','whitened_mlp']:
            key=f'{source}|{kind}'
            if key not in result['summary']:continue
            ax.plot(np.arange(1,17),result['summary'][key]['matched']['gain_per_pc'],'o-',color=colors[kind],label=kind.replace('_',' '))
        ax.axhline(1,color='#777777',linestyle='--',linewidth=1)
        ax.axvspan(2.5,16.5,color='#eeeeee',zorder=0)
        ax.set_title('White source' if source=='white' else 'GP source',loc='left')
        ax.set_xlabel('Training-velocity principal component');ax.set_ylabel('Response gain versus matched oracle')
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
    fig.suptitle('Learned velocity response, with the same observed history',x=.02,ha='left')
    fig.tight_layout();fig.savefig(FIG/'matched_oracle_pc_response.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
