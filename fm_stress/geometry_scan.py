"""CPU-only, outcome-independent lengthscale sensitivity for the data geometry.

This script never selects a winning trained model or changes a training config.
Its output is a labeled supplementary geometry audit, not a primary result.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .data import GPConfig, sample_dataset, sample_source
from .metrics import fit_pca, spectrum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/main.json',type=Path)
    parser.add_argument('--output',default='results/geometry-sensitivity',type=Path)
    parser.add_argument('--lengthscales',type=float,nargs='+',default=[4,6,8,10,12,16,24])
    parser.add_argument('--n-train',type=int,default=8192)
    parser.add_argument('--n-heldout',type=int,default=8192)
    parser.add_argument('--seed',type=int,default=37041)
    args=parser.parse_args()
    config=json.loads(args.config.read_text());base=GPConfig(**config['data'])
    results=[]
    for lengthscale in args.lengthscales:
        gp=replace(base,lengthscale=lengthscale)
        data={split:sample_dataset(gp,args.n_train if split=='train' else args.n_heldout,args.seed+i)
              for i,split in enumerate(['train','val','test'])}
        for kind in ['target','white','gp']:
            arrays={}
            for i,split in enumerate(['train','val','test']):
                arrays[split]=data[split].target
                if kind!='target':arrays[split]=arrays[split]-sample_source(gp,len(data[split]),kind,args.seed+100+10*i)
            pca=fit_pca(arrays['train'])
            for split,array in arrays.items():
                report=spectrum(array,pca)
                results.append({'lengthscale':lengthscale,'quantity':kind,'split':split,
                                'n':len(array),'seed':args.seed,**report})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'geometry_scan.json').write_text(json.dumps({'status':'sensitivity_only_no_model_outcomes',
        'base_config':base.to_dict(),'lengthscales':args.lengthscales,'seed':args.seed,
        'pca_fit':'training draws only, separately per lengthscale/quantity','results':results},indent=2)+'\n')
    # Table and static scientific plot; rendering imports are delayed for CLI reuse.
    from .analyze import write_csv, setup_style, save_figure
    import matplotlib.pyplot as plt
    write_csv(args.output/'geometry_scan.csv',[{k:v for k,v in row.items() if not isinstance(v,list)} for row in results])
    setup_style();fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    colors={'target':'#20333F','white':'#526DAB','gp':'#13837E'}
    labels={'target':'Target patches','white':'White-source velocity','gp':'GP-source velocity'}
    for kind in ['target','white','gp']:
        rows=[r for r in results if r['quantity']==kind and r['split']=='test']
        x=[r['lengthscale'] for r in rows]
        axes[0].plot(x,[r['top2_energy'] for r in rows],'o-',color=colors[kind],label=labels[kind],lw=1.8)
        axes[1].plot(x,[r['mean_within_patch_std'] for r in rows],'o-',color=colors[kind],label=labels[kind],lw=1.8)
    axes[0].axhline(.9,ls='--',color='#B85348',lw=1,label='90% geometry threshold')
    for ax in axes:
        ax.axvline(base.lengthscale,ls=':',color='#9DAEB7',label='Primary lengthscale' if ax is axes[1] else None)
        ax.set_xlabel('RBF lengthscale (time steps)');ax.grid(alpha=.5);ax.legend(fontsize=8)
    axes[0].set(title='Held-out top-two energy',ylabel='Variance share',ylim=(0,1.03))
    axes[1].set(title='Check against constant-patch collapse',ylabel='Mean within-patch standard deviation')
    fig.suptitle('Geometry sensitivity · fixed F=16, L=32 · no model selection',fontweight='bold')
    save_figure(fig,args.output,'geometry_sensitivity')
    print(f'Saved {len(results)} spectrum records to {args.output}')


if __name__=='__main__':main()
