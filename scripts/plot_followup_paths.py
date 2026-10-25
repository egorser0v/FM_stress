"""Show fixed examples from the capacity study, without choosing by model scores."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'results/followup-capacity/rbf/F64_l8/draw607041'
    heads=[('mlp_original','MLP 56 × 3'),('mlp_rank_clear','MLP 65 × 2'),
           ('mlp_wide','MLP 72 × 3'),('s4','S4 59 × 3')]
    sources=['white','gp']
    data={(source,head):np.load(folder/'samples'/f'{source}_{head}_budget12000.npz')
          for source in sources for head,_ in heads}
    context=0
    reference=data[('white','mlp_original')]
    for arrays in data.values():
        assert np.array_equal(arrays['path_history'][context],reference['path_history'][context])
        assert np.array_equal(arrays['ensemble_target'][context],reference['ensemble_target'][context])
    fig,axes=plt.subplots(2,4,figsize=(12,6.2),sharex=True,sharey=True)
    for row,source in enumerate(sources):
        for col,(head,title) in enumerate(heads):
            ax=axes[row,col];a=data[source,head]
            assert np.array_equal(a['path_target'][context],a['ensemble_target'][context])
            ax.plot(np.arange(-24,0),a['path_history'][context,-24:],color='#425b65',lw=1.4)
            for path in a['ensemble'][context,:8]:
                ax.plot(np.arange(64),path,color='#2c7f9a',alpha=.32,lw=.75)
            ax.plot(np.arange(64),a['ensemble_target'][context],color='#ce643d',lw=1.8)
            ax.axvline(-.5,color='#9aadb4',ls=':',lw=.8)
            ax.set_title(title,loc='left',fontsize=11)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelsize=8)
            if col==0:ax.set_ylabel(('White source' if source=='white' else 'GP source')+'\nNormalized value')
            if row==1:ax.set_xlabel('Relative forecast time')
    fig.suptitle('A fixed held-out history: eight generated paths per model',x=.02,ha='left',fontsize=15)
    fig.text(.02,.015,'Gray: observed history   ·   Orange: one realized future   ·   Blue: independent generated futures',fontsize=10,color='#425b65')
    fig.tight_layout(rect=(0,.045,1,.94))
    out=ROOT/'output/followup-capacity';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'fixed_context_paths.png',dpi=190);plt.close(fig)
    (out/'fixed_context_paths.json').write_text(json.dumps({
        'data_seed':607041,'context_index':context,'member_indices':list(range(8)),
        'budget':12000,'selection_rule':'First draw, first held-out context and first eight members, chosen by index, not score.',
        'input_sha256':{str((folder/'samples'/f'{source}_{head}_budget12000.npz').relative_to(ROOT)):hashlib.sha256((folder/'samples'/f'{source}_{head}_budget12000.npz').read_bytes()).hexdigest() for source in sources for head,_ in heads},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'The realized target is one possible future, not a unique correct generated path. All panels share axes; aggregate metrics use all contexts.'
    },indent=2)+'\n')


if __name__=='__main__':main()
