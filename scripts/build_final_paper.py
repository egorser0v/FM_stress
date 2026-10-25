#!/usr/bin/env python3
"""Build the final paper's canonical evidence and LaTeX tables, then compile.

All displayed values are rounded only at export. evidence.json preserves the
full precision of the recorded artifacts and names every consumed source.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, statistics, subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/final-paper'
CONSUMED = {}

def read(path):
    p=ROOT/path
    CONSUMED[str(path)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())

def dump(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')

def table(filename, headers, rows, align=None):
    align=align or ('l'+'r'*(len(headers)-1))
    lines=[r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/filename).write_text('\n'.join(lines)+'\n')

def author_contributions(payload):
    if payload.get('status') != 'user_approved':
        raise ValueError('Author contributions must have explicit user approval')
    def tex(text):
        replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
        return ''.join(replacements.get(c,c) for c in text)
    paragraphs=[r'\paragraph{Author contributions.}']
    paragraphs.extend(r'\textbf{'+tex(member['name'])+'.} '+tex(member['statement']) for member in payload['members'])
    (OUT/'contributions.tex').write_text('\n\n'.join(paragraphs)+'\n')

def figures(geom, trajectory):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/final-paper-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.55,2.15),layout='constrained')
    for key,name,color in [('target','Target','#222222'),('white','White velocity','#b45c22'),('gp','GP velocity','#20689b')]:
        ax.semilogy(np.arange(1,17),geom['spectra'][key]['train']['pc_variance'],marker='.',label=name,color=color)
    ax.set(xlabel='Training PCA direction',ylabel='Variance',xticks=[1,4,8,12,16]);ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'geometry.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    if trajectory:
        gs=[x for x in trajectory['groups'] if x['suite']=='main']
        fig,ax=plt.subplots(figsize=(3.55,2.1),layout='constrained');ix=np.arange(3);w=.34
        for shift,key,name,color in [(-w/2,'R1_ratio','First differences','#20689b'),(w/2,'R2_ratio','Second differences','#b45c22')]:
            ax.bar(ix+shift,[x['means'][key] for x in gs],w,label=name,color=color)
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set(xticks=ix,xticklabels=['MLP','S4','Whitened MLP'],ylabel='Generated / target',ylim=(0,1.85));ax.legend(frameon=False,fontsize=7,loc='upper left')
        fig.savefig(OUT/'roughness.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)

def multivariate():
    extension=read('output/extension-analysis/summary.json')
    sparse=read('output/sparse-analysis/summary.json')
    selected=[]
    for package,origin,heads in [(extension,'output/extension-analysis/summary.json',{'mlp','s4'}),(sparse,'output/sparse-analysis/summary.json',{'mlp_whitened'})]:
        for row in package['aggregate']:
            if row['head'] not in heads:continue
            selected.append({'dataset':row['dataset'],'head':'whitened_mlp' if row['head']=='mlp_whitened' else row['head'],'source':row['source'],'n_seeds':row['n'],'seeds':[31,32,33],'parameters':row['parameters'],'metrics':row['metrics'],'best_steps':row.get('best_steps',[]),'source_path':origin})
    datasets={}
    for key,row in extension['datasets'].items():
        baselines=sparse['datasets'][key]['baselines']
        datasets[key]={k:row[k] for k in ['channels','lookback','horizon','kind','split_sizes','geometry']}
        datasets[key]['baselines']={k:{'means':v['means'],'type':v.get('type')} for k,v in baselines.items()}
        datasets[key]['source_channel_correlation']=sparse['datasets'][key]['metadata']['source_channel_correlation']
    supplement=read('output/dictionary-supplement/summary.json')
    audits={}
    for name,path in [('extension_runs','results/extension/independent_verification.json'),('extension_analysis','results/diagnostics/extension_analysis_verification.json'),('sparse_analysis','results/diagnostics/sparse_analysis_verification.json'),('dictionary_analysis','results/diagnostics/dictionary_analysis_verification.json')]:
        d=read(path);audits[name]={k:v for k,v in d.items() if not isinstance(v,(dict,list))}
    result={'cells':selected,'datasets':datasets,'audits':audits,
            'protocol':{'steps':2000,'batch_size':128,'seeds':[31,32,33],'evaluation_contexts':64,'ensemble_members':32,'euler_steps':64,'normalization':'Fixed training-only channel mean/std; no per-window history normalization.','source_covariances':'White I_F x C_train; GP normalized RBF_ell8 x same C_train; 5% channel-correlation shrinkage.','joint_energy_definition':'Fair whole-path Euclidean energy divided by sqrt(F*C).','scope':'Selected 24 source/head/dataset cells (72 archived fixed checkpoints); no fresh training. Three optimizer seeds share fixed datasets/periods.'},
            'extension':{k:extension[k] for k in ['status','expected_runs','observed_runs','config','aggregate','uncertainty']},
            'sparse':{**{k:sparse[k] for k in ['status','expected_new_runs','observed_new_runs','frozen_control_runs','training','interpretation']},'dictionary_supplement':{k:supplement[k] for k in ['status','expected_counts','observed_counts']},'total_archived_new_fits':288},
            'ot_only':{k:extension['ot_only'][k] for k in ['status','aligned_contexts','aggregate','contrasts','interpretation']}}
    names={'gp_multi':'GP (3)','etth1_uni':'ETTh1 OT (1)','etth1_multi':'ETTh1 (7)','exchange_multi':'Exchange (8)'}
    cells={(c['dataset'],c['source'],c['head']):c for c in selected}
    for metric,filename in [('fair_crps','multivariate_crps_table.tex'),('joint_energy_scaled','multivariate_energy_table.tex')]:
        rows=[]
        for dataset in names:
            for source in ['white','gp']:
                rows.append([names[dataset]+(' / W' if source=='white' else ' / G'),*[f"{cells[(dataset,source,head)]['metrics'][metric]['mean']:.4f}" for head in ['mlp','s4','whitened_mlp']],f"{datasets[dataset]['baselines']['ridge_residual_bootstrap']['means'][metric]:.4f}"])
        table(filename,['Dataset (C) / source','MLP','S4','Whiten.','Ridge'],rows)
    return result

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    main=defaultdict(list)
    for p in sorted((ROOT/'results/main/runs').glob('*.json')):
        d=read(str(p.relative_to(ROOT)));main[f"{d['source']}|{d['head']}"].append(d)
    groups={}
    for key,rows in main.items():
        metric_keys=['pc1_mse','pc2_mse','residual_mse','residual_normalized_mse']
        metrics={k:{'mean':statistics.mean(r['metrics'][k] for r in rows),'sd':statistics.stdev(r['metrics'][k] for r in rows)} for k in metric_keys}
        metrics['roughness_ratio']={'mean':statistics.mean(r['metrics']['success']['roughness_ratio'] for r in rows),'sd':statistics.stdev(r['metrics']['success']['roughness_ratio'] for r in rows)}
        groups[key]={'n_seeds':len(rows),'seeds':[r['seed'] for r in rows],'parameters':rows[0]['parameters'],'metrics':metrics,'literal_passes':sum(r['metrics']['success']['works'] for r in rows),'roughness_passes':sum(r['metrics']['success']['roughness_rule_pass'] for r in rows)}
    geom=read('results/main/geometry.json')
    ens=read('results/ensemble-evaluation/summary.json')
    cap=read('output/followup-capacity/summary.json')
    norm=read('output/followup-normalization/summary.json')
    label=read('output/followup-label-noise/summary.json')
    solver=read('output/followup-label-solver/summary.json')
    cb=read('results/followup-capacity-baselines/summary.json')
    oracle=read('results/followup-oracle/summary.json')
    compute={'main':{'fits':25,'logged_seconds':sum(r['seconds'] for rows in main.values() for r in rows),'timing_field':'results/main/runs/*.json:seconds','scope':'Recorded run duration; not dedicated device hours or architecture benchmark.'}}
    for name in ['followup-capacity','followup-label-noise']:
        records={}
        training_paths=[]
        for p in sorted((ROOT/'results'/name).glob('**/runs/*.json')):
            if 'pre-audit-interrupted' in str(p):continue
            d=read(str(p.relative_to(ROOT)))
            if 'training' in d:
                # Capacity repeats the training record for each budget checkpoint.
                records[str(p.relative_to(ROOT)).replace('_budget6000.json','').replace('_budget12000.json','')]=d['training']
                training_paths.append(str(p.relative_to(ROOT)))
        compute[name]={'unique_training_records':len(records),'logged_training_seconds':sum(r['seconds'] for r in records.values()),'source_paths':training_paths,'timing_field':'training.seconds','device':'mps','scope':'Sum of logged training durations only; concurrent runs must not be interpreted as serial elapsed or dedicated GPU hours.'}
    nfiles=sorted((ROOT/'results/followup-normalization/runs').glob('*.json'))
    compute['followup-normalization']={'fits':len(nfiles),'logged_seconds':sum(read(str(p.relative_to(ROOT)))['seconds'] for p in nfiles),'timing_field':'results/followup-normalization/runs/*.json:seconds','scope':'Recorded full-run durations; concurrent MPS load prevents speed comparisons.'}
    evidence={'schema_version':1,'main':{'groups':groups,'literal_passes':sum(g['literal_passes'] for g in groups.values()),'roughness_passes':sum(g['roughness_passes'] for g in groups.values()),'runs':25,'geometry':{k:{s:{m:v[s][m] for m in ['top1_energy','top2_energy','total_variance','roughness']} for s in ['train','val','test']} for k,v in geom['spectra'].items()},'config':read('configs/main.json'),'paired_contrasts':read('output/analysis/paired_seed_contrasts.json')},'ensemble':ens,'capacity':cap,'normalization':{k:v for k,v in norm.items() if k not in ['baselines']},'normalization_baselines':{k:{'alpha':v['alpha'],'forecast_means':v['forecast_means']} for k,v in norm['baselines'].items()},'label_noise':{k:v for k,v in label.items() if k in ['complete','observed_runs','rows','contrasts','interactions','best_at_final_step','literal_successes','full_solver_sensitivity']},'label_solver':solver,'capacity_no_flow':cb,'matched_oracle':oracle,'compute':compute}
    trajectory=ROOT/'results/trajectory-metrics/summary.json'
    if trajectory.exists():evidence['trajectory_metrics']=read('results/trajectory-metrics/summary.json')
    evidence['pca_audit']=read('results/final-pca-audit/summary.json')
    mixture=read('output/prior-mixture/summary.json')
    evidence['mixture']={k:v for k,v in mixture.items() if k in ['complete','verified_runs','config','cells','contrasts','uncertainty']}
    evidence['balanced_objective']=read('output/controlled-followup/summary.json')
    evidence['multivariate']=multivariate()
    evidence['contributions']=read('docs/contributions.json')
    author_contributions(evidence['contributions'])
    evidence['source_paths']=dict(sorted(CONSUMED.items()))
    evidence['scope']='Primary experiment and selected, explicitly exploratory follow-ups. Repeated checkpoints are not independent fits. Author contributions reproduce the user-approved role statement.'
    dump(OUT/'evidence.json',evidence)
    names={'white|mlp':'White / MLP','white|s4':'White / S4','gp|mlp':'GP / MLP','gp|s4':'GP / S4','gp|whitened_mlp':'GP / whitened MLP'}
    rows=[]
    for key in names:
        g=groups[key];m=g['metrics'];rows.append([names[key],f"{m['pc1_mse']['mean']:.3f}",f"{m['residual_mse']['mean']:.5f}",f"{m['roughness_ratio']['mean']:.3f}",f"{g['literal_passes']}/5"])
    table('main_table.tex',['Source / head',r'$M_1$',r'$M_R$',r'$R/R_y$','Pass'],rows)
    table('geometry_table.tex',['Quantity',r'$E_2$ val',r'$E_2$ test'],[[k.title(),f"{v['val']['top2_energy']:.4f}",f"{v['test']['top2_energy']:.4f}"] for k,v in geom['spectra'].items()])
    table('ensemble_table.tex',['GP-source head','Fair CRPS','Energy','Cov90'],[[{'mlp':'MLP','s4':'S4','whitened_mlp':'Whitened MLP'}[k],f"{v['metrics']['fair_crps']['mean']:.4f}",f"{v['metrics']['fair_energy_score']['mean']:.4f}",f"{v['metrics']['coverage_90']['mean']:.3f}"] for k,v in ens['heads'].items()])
    table('capacity_table.tex',['Source / head',r'$M_R$','CRPS',r'$R/R_y$'],[[c['source']+' / '+{'mlp_original':'MLP56','mlp_rank_clear':'MLP65','mlp_wide':'MLP72','s4':'S4'}[c['head']],f"{c['metrics']['residual_mse']['mean']:.4f}",f"{c['metrics']['fair_crps']['mean']:.4f}",f"{c['metrics']['roughness_ratio']['mean']:.3f}"] for c in cap['cells'] if c['kernel']=='rbf' and c['budget']==12000])
    table('normalization_table.tex',['Pipeline','GP MLP','GP S4','Ridge'],[[mode.replace('_',r'\_'),*[f"{norm['table'][f'{mode}|{s}|{h}|6000']['mean']['fair_crps']:.4f}" for s,h in [('gp','mlp'),('gp','s4')]],f"{norm['baselines'][mode]['forecast_means']['fair_crps']:.4f}"] for mode in ['global','center','scale','history','history_rawloss']])
    lrows=[]
    for source in ['white','gp']:
        for head in ['mlp','s4']:
            a=[r for r in label['rows'] if r['source']==source and r['head']==head]
            lrows.append([source+' / '+head.upper(),*[f"{statistics.mean(r['fair_crps'] for r in a if r['label']==lab):.4f}" for lab in ['stochastic','oracle']],*[f"{statistics.mean(r['oracle_residual_mse'] for r in a if r['label']==lab):.5f}" for lab in ['stochastic','oracle']]])
    table('label_table.tex',['Source / head','CRPS S','CRPS C','Gap S','Gap C'],lrows)
    if trajectory.exists():
        t=evidence['trajectory_metrics']
        rows=[]
        for g in t.get('groups',[]):
            if g['suite']=='main':
                m=g['means'];rows.append([g['group'].replace('_',r'\_'),f"{m['R1_ratio']:.3f}",f"{m['R2_ratio']:.3f}",f"{m['variogram_plugin']:.5f}",f"{m['variogram_fair']:.5f}"])
        table('trajectory_table.tex',['Head',r'$R_1/R_{1y}$',r'$R_2/R_{2y}$','VS plug-in','VS fair'],rows)
    pmain=groups['gp|mlp']['metrics']['residual_mse']['mean'];pwhite=groups['gp|whitened_mlp']['metrics']['residual_mse']['mean']
    macros={'WhiteningReduction':f'{100*(1-pwhite/pmain):.1f}','MainLoggedSeconds':f"{compute['main']['logged_seconds']:.1f}"}
    (OUT/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    figures(geom,evidence.get('trajectory_metrics'))
    authored=['main.tex','contributions.tex','geometry.pdf','roughness.pdf']
    dump(OUT/'build_manifest.json',{'consumed_sources':evidence['source_paths'],'evidence_sha256':hashlib.sha256((OUT/'evidence.json').read_bytes()).hexdigest(),'authored_sources':{p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in authored}})
    return evidence

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-compile',action='store_true');args=parser.parse_args()
    build()
    if not args.no_compile:
        compiler=shutil.which('pdflatex') or '/Library/TeX/texbin/pdflatex'
        for _ in range(2):subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=OUT,check=True,stdout=subprocess.DEVNULL)
    print(OUT/'evidence.json')
