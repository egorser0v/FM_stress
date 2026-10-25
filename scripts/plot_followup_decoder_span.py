#!/usr/bin/env python3
"""Plot verified forecast variance in the MLP decoder's immutable directions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def main():
    data=json.loads((ROOT/'results/followup-reproduction/decoder_span.json').read_text())
    if not data['complete']:raise SystemExit('Full nine-MLP span audit is required')
    fields=[('observed_source_variance_trace','Initial source','#9cacae'),
            ('observed_ensemble_variance_trace','Generated forecasts','#c45d37'),
            ('test_target_marginal_variance_trace','Test targets','#15858b')]
    groups=[('rbf','white'),('rbf','gp'),('ou','gp')]
    fig,ax=plt.subplots(figsize=(9,4.6));fig.patch.set_facecolor('white')
    for j,(field,label,color) in enumerate(fields):
        vals=[[r[field] for r in data['runs'] if (r['identity']['kernel'],r['identity']['source'])==g] for g in groups]
        xx=np.arange(3)+(j-1)*.23
        ax.bar(xx,[np.mean(v) for v in vals],width=.21,label=label,color=color,zorder=2)
        for x,v in zip(xx,vals):
            ax.scatter(x+np.linspace(-.045,.045,len(v)),v,s=15,color='#263d45',zorder=3)
            label_value=f'{np.mean(v):.2f}' if np.mean(v)>=.005 else f'{np.mean(v):.1e}'
            ax.annotate(label_value,(x,max(v)),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(range(3),['RBF target\nwhite source','RBF target\nGP source','OU target\nGP source'])
    ax.set_ylabel('Variance in the eight decoder-complement directions')
    ax.set_ylim(bottom=0);ax.margins(y=.17);ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2,zorder=0);ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.15))
    fig.suptitle('The narrow MLP cannot change source variation in eight directions',x=.07,ha='left',fontsize=14,fontweight='bold')
    fig.text(.07,.02,'F = 64, MLP width = 56; three independent data/training draws. Dots are draws.\nTargets use marginal variance over 8,192 windows; source/forecast variance is within 256 ensembles of 32 draws.',fontsize=9,color='#526871')
    fig.tight_layout(rect=(.02,.1,.99,.9))
    out=ROOT/'output/followup-reproduction/decoder_null_variance.png';out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180,bbox_inches='tight')
    print(out)
if __name__=='__main__':main()
