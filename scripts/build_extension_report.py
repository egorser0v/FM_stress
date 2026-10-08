"""Build a report from the complete, independently checked extension only."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_report import Report, register_fonts, styles, WIDTH, NAVY, MUTED
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

LABELS = {'gp_multi':'Correlated GP: three channels','etth1_uni':'ETTh1: oil temperature',
          'etth1_multi':'ETTh1: seven channels','exchange_multi':'Exchange: eight channels'}
HEADS = {'mlp':'Dense MLP','sparse_mlp':'Sparse MLP','s4':'S4','unet':'U-Net'}


def read(path): return json.loads((ROOT/path).read_text())


class Extension(Report):
    def __init__(self):
        self.s = styles(); self.story = []
        self.args = SimpleNamespace(analysis=ROOT/'output/extension-analysis')
        self.summary = read('output/extension-analysis/summary.json')
        self.audit = read('results/extension/independent_verification.json')
        assert self.summary['status']=='complete' and self.summary['observed_runs']==96
        assert self.audit['complete'] and self.audit['verified_runs']==96, 'Complete independent audit required'
        assert self.audit['fingerprint']==self.summary['fingerprint']==read('results/extension/manifest.json')['fingerprint']
        analysis_audit=read('results/diagnostics/extension_analysis_verification.json')
        assert analysis_audit['status']=='passed' and analysis_audit['summary_sha256']==hashlib.sha256((ROOT/'output/extension-analysis/summary.json').read_bytes()).hexdigest()
        self.cfg = read('configs/extension.json')
        assert self.cfg==self.summary['config'], 'Report configuration differs from analyzed experiment'
        self.rows = self.summary['aggregate']

    def cells(self,dataset):
        return sorted([r for r in self.rows if r['dataset']==dataset],
                      key=lambda r:(self.cfg['sources'].index(r['source']),self.cfg['heads'].index(r['head'])))

    def fmt(self,row,metric):
        x=row['metrics'][metric]
        return f"{x['mean']:.4g} ± {x['sd']:.2g}"

    def opening(self):
        self.title('Smooth source, unstructured head | October 2026','Beyond univariate GP',True)
        self.p('Sparse MLP, U-Net and real multivariate targets','h2')
        self.p('Egor Serov · Kirill Frolov · Daniil Koblov · Vasilii Lyamin','small')
        self.p('This extension adds a fixed-mask sparse MLP, a temporal U-Net, genuinely joint multivariate forecasting, and two real datasets. Dense MLP and S4 are retrained as controls. The original univariate synthetic experiments remain unchanged.')
        self.table(['Completed comparison','Protocol'],[
            ['96 MPS fits','4 datasets × 4 heads × 2 sources × 3 seeds'],
            ['Capacity and training','Approximately 64k active parameters; 2,000 updates; best validation checkpoint'],
            ['Conditional forecasts','64 test contexts × 32 generated futures; 64 Euler steps'],
            ['External validity','ETTh1 from Sundial evaluation; Exchange from TSFlow evaluation'],
            ['Verification','Fresh-data reconstruction, independent metric recomputation, CPU/MPS model checks']], [140,WIDTH-140])
        rows=[]; baseline_wins=0
        for dataset in self.cfg['datasets']:
            best=min(self.cells(dataset),key=lambda r:r['metrics']['fair_crps']['mean'])
            bases=read(f'results/extension/{dataset}/baselines.json')['records']
            bn=min(bases,key=lambda k:bases[k]['means']['fair_crps'])
            bv=bases[bn]['means']['fair_crps']
            baseline_wins += bv < best['metrics']['fair_crps']['mean']
            rows.append([LABELS[dataset], HEADS[best['head']]+' / '+best['source'],
                         self.fmt(best,'fair_crps'),bn.replace('_',' '),f'{bv:.4g}'])
        self.p('Lowest observed mean CRPS in each dataset','h2')
        self.table(['Dataset','Neural cell','CRPS ± seed SD','Best baseline','CRPS'],rows,[108,108,98,118,WIDTH-432])
        self.p('These are descriptive minima among tested cells, not statistically established winners. CRPS is in train-standardized units and should only be compared within a dataset. Baselines are fitted once and have no optimization-seed SD. White means temporally white with the same cross-channel correlation as the GP source.','small')
        self.p(f'**Keep the baseline comparison in view.** On {baseline_wins} of the four datasets, a simple baseline has lower CRPS than the best mean neural cell. Architecture ranking alone therefore does not establish forecasting usefulness. All scores and the uncertainty/calibration checks are reported below.','box')
        self.p('**A source-dependent crossover appears on ETTh1 OT.** S4 has lower CRPS than dense MLP with the GP source (0.144 vs 0.195), but higher CRPS with the white source (0.170 vs 0.142). Its GP-source residual velocity MSE is nevertheless worse. This is evidence about forecasting at this budget, not confirmation of the original small-PC failure mechanism.','small')
        self.p('This is a fixed-budget generalization study, not a full reproduction of pretrained Sundial or TSFlow. Real-data and multivariate results do not replace the original assignment\'s univariate rank-two geometry or its declared success rule.','small')

    def protocol(self):
        self.title('01 | Design','What changes, and what stays comparable')
        self.table(['Head','Architecture','Sparse/temporal detail'],[
            ['Dense MLP','Joint flattened horizon and channels; adaptive LayerNorm','Unstructured learned affine maps'],
            ['Sparse MLP','Same family, fixed approximately 50% masks on all linear maps','Wider to match active capacity; dense kernels and about twice allocated weights'],
            ['S4','Three rank-one HiPPO/NPLR temporal blocks with channel mixing','Reuses the independently validated project S4 operator'],
            ['U-Net','Three temporal resolutions, skip connections and conditional convolutions','Two down/up stages; no convolution over flattened channel indices']], [75,215,WIDTH-290])
        rows=[]
        for name,d in self.cfg['datasets'].items():
            meta=read(f'results/extension/{name}/data_metadata.json')
            cap=read(f'results/extension/{name}/capacities.json')
            rows.append([LABELS[name],f"{d['lookback']} → {d['horizon']}",
                         '/'.join(str(meta['split_sizes'][s]) for s in ('train','val','test')),
                         ' / '.join(f"{cap[h]['active']:,}" for h in self.cfg['heads'])])
        self.table(['Dataset','L → F','Train / val / test windows','Active counts: MLP / sparse / S4 / U-Net'],rows,[110,50,145,WIDTH-305])
        self.p('**Information and scaling.** Every head receives the same full history through an identity encoder. All channels use fixed training-only channel mean and standard deviation; no validation or test value enters fitting. Training targets may enter training statistics and PCA. Unlike the original experiment, this extension does not apply per-window history normalization.')
        self.p('**Sources.** Independent source/target pairs use the same train-estimated channel correlation. The white source has identity temporal covariance; the GP source has an RBF temporal kernel of lengthscale 8. Both have unit marginal source variance. Cross-channel correlation is held fixed while temporal covariance changes.')
        self.p('**Optimization.** Raw velocity MSE, AdamW learning rate 0.001, batch 128, gradient clipping 1, 2,000 updates. Validation is checked every 250 updates. There is no head-specific rate search here. Equal active capacity and updates do not imply equal computational cost or equal convergence.')
        self.p('ETTh1 uses the 12/4/4-month row split and 48→24 forecasting; Exchange uses a 70/10/20% chronological split and 32→16 forecasting. Whole history+future windows lie within their split. Test targets do not overlap, but temporal dependence remains. Data revisions, SHA-256 checksums, exact starts and scalers are saved.','small')

    def scores(self):
        self.title('02 | Forecast distributions','Compare predictions on the same targets')
        self.figure('score_crps','Marginal fair CRPS. Lower is better. Markers/error bars summarize three optimization seeds on the same 64 held-out contexts. Compare sources within a dataset, not across datasets.',370)
        rows=[]
        for head in self.cfg['heads']:
            values=[]
            for dataset in self.cfg['datasets']:
                cells={r['source']:r for r in self.cells(dataset) if r['head']==head}
                values.append(f"{cells['gp']['metrics']['fair_crps']['mean']-cells['white']['metrics']['fair_crps']['mean']:+.4f}")
            rows.append([HEADS[head],*values])
        self.p('Mean CRPS difference: GP source minus white source','h2')
        self.table(['Head','GP data (3C)','ETTh1 (1C)','ETTh1 (7C)','Exchange (8C)'],rows,[95,102,102,102,WIDTH-401])
        self.p('Negative values favor the GP source within that head and dataset; these are descriptive mean differences. Joint energy evaluates the whole horizon × channel vector and is divided by sqrt(F×C). It is reported alongside CRPS in every dataset table and as a separate vector figure.','small')
        self.p('Ensemble scores use off-diagonal corrections for 32 samples. The full CSV includes channel-sum CRPS, point MSE, 90% interval coverage/width and roughness. A visually smooth path can still be an inaccurate or poorly calibrated forecast.','small')

    def dataset(self,name):
        self.title('03 | Measured results',LABELS[name])
        cells=self.cells(name)
        rows=[[HEADS[r['head']],r['source'],self.fmt(r,'fair_crps'),self.fmt(r,'joint_energy_scaled'),
               self.fmt(r,'mean_forecast_mse'),f"{100*r['metrics']['coverage_90']['mean']:.1f}%"] for r in cells]
        self.table(['Head','Source','CRPS ↓','Joint energy ↓','Mean MSE ↓','90% coverage'],rows,[77,51,96,96,96,WIDTH-416])
        self.p('Mean ± sample SD over three training seeds. Coverage is the marginal proportion of target coordinates inside central 90% sample intervals. It is not simultaneous whole-path coverage.','small')
        rows=[[HEADS[r['head']],r['source'],self.fmt(r,'velocity_pc1_mse'),self.fmt(r,'velocity_residual_mse'),
               self.fmt(r,'roughness_ratio')] for r in cells]
        self.p('Connection to the original stress-test metrics','h2')
        self.table(['Head','Source','PC1 velocity MSE','Residual velocity MSE','Roughness ratio'],rows,[77,51,125,125,WIDTH-378])
        self.p('PCA is fitted to training velocities separately for each source. Residual error is the per-component average over PCs 3 onward in joint F×C coordinates. Roughness is computed along time only, with target-normalized ratio 1 meaning equal average absolute adjacent increments. Velocity labels differ across sources; lower cross-source velocity MSE alone does not establish a better forecast.','small')
        bases=read(f'results/extension/{name}/baselines.json')
        br=[[key.replace('_',' '),f"{r['means']['fair_crps']:.4g}",f"{r['means']['mean_forecast_mse']:.4g}",
             f"{100*r['means']['coverage_90']:.1f}%"] for key,r in bases['records'].items()]
        self.table(['Baseline','CRPS','MSE','90% coverage'],br,[235,90,90,WIDTH-415])
        self.p(f"Ridge alpha {bases['selected_ridge_alpha']:g} was chosen on validation MSE. Residual bootstrap resamples entire training residual vectors and can understate uncertainty. Deterministic baselines have degenerate intervals.",'small')

    def ot(self):
        self.title('04 | Joint versus univariate','Does observing other channels help OT?')
        ot=self.summary['ot_only']
        self.p('Both ETTh1 experiments use the same held-out timestamps and OT training scaler. The joint model is rescored on its OT output alone. Comparing its seven-channel average against the OT-only model would answer a different question.')
        rows=[]
        for r in ot['aggregate']:
            # Analyzer records the original dataset name for the OT slice.
            scope={'etth1_uni':'ETTh1 OT-only','etth1_multi':'ETTh1 joint'}[r['dataset']]
            rows.append([scope,HEADS[r['head']],r['source'],
                         self.fmt(r,'fair_crps'),self.fmt(r,'mean_forecast_mse')])
        self.table(['Training target','Head','Source','OT CRPS','OT mean MSE'],rows,[110,72,51,135,WIDTH-368])
        self.p('A joint model predicts all seven channels with approximately the same total active-parameter budget as the univariate model. This is an operational comparison, not a clean causal estimate of adding information alone: output dimension, channel mixing and parameter allocation also change.','small')
        self.p('OT-only comparisons have paired seed differences in `output/extension-analysis/summary.json`. The separate full-dataset head/source comparisons also have exploratory context intervals: circular moving blocks of four selected contexts, with block length eight as sensitivity for real data. These condition on the selected test segment and fixed trained seeds; neither three seeds nor two real datasets support a universal architecture claim.','small')

    def verification(self):
        self.title('05 | Scope and reproducibility','What the checks establish')
        rows=[]
        for d in self.cfg['datasets']:
            g=read(f'results/extension/{d}/geometry.json')
            rows.append([LABELS[d],f"{g['target']['validation_top2_energy']:.3f}",
                         f"{g['white']['validation_top2_energy']:.3f}",f"{g['gp']['validation_top2_energy']:.3f}",
                         f"{g['gp']['validation_top2C_energy']:.3f}"])
        self.table(['Validation variance share','Target E2','White velocity E2','GP velocity E2','GP velocity E2C'],rows,[163,75,90,90,WIDTH-418])
        self.p('Training-fitted PCA. Multivariate top-2C and per-channel spectra supplement top-2 energy; channel modes can increase joint rank even when each channel is smooth. The original rank-two gate is not imposed on generalization experiments. Full spectra and plots are supplied separately.','small')
        allruns=[json.loads(p.read_text()) for p in (ROOT/'results/extension').glob('*/runs/*.json')]
        boundary=sum(r['best_step']==self.cfg['steps'] for r in allruns)
        self.p(f'**Optimization limit.** {boundary}/96 runs select the last checked step. This alone neither proves nor disproves convergence; the full validation curves are supplied. Results describe a 2,000-update budget and one learning rate, not best achievable performance.')
        rows=[]
        for d in self.cfg['datasets']:
            rr=[json.loads(p.read_text()) for p in (ROOT/'results/extension'/d/'runs').glob('*.json')]
            sol=[r['solver_sensitivity'] for r in rr if r['solver_sensitivity']]
            rows.append([LABELS[d],f"{max(s['endpoint_rmse'] for s in sol):.3g}",
                         f"{max(abs(s['refined']['fair_crps']-s['base']['fair_crps']) for s in sol):.3g}"])
        self.table(['Euler 64 → 128 check','Max endpoint RMSE','Max |CRPS change|'],rows,[230,135,WIDTH-365])
        self.p('The solver check covers every head/source at seed 31, first eight selected contexts. It is a numerical sensitivity check, not a guarantee for all contexts or larger step counts.','small')
        self.p('**Independent checks.** The audit reconstructs data and train-only transforms, verifies chronological splits, source covariance/PCA, best-checkpoint selection and file hashes, and independently recomputes all saved forecast and velocity metrics. Saved MPS checkpoint outputs are also reproduced on CPU. Full-model forward/backward/update parity passed for all four heads at 1, 3, 7 and 8 channels.')
        self.p('Reproduce with `scripts/fetch_extension_data.py`, `python -m fm_stress.extended_experiment`, `scripts/check_extension.py`, and `python -m fm_stress.analyze_extension`. Use the project virtual environment and MPS access. Exact commands and interpretation are in `docs/extension_protocol.md`; findings are in `docs/extension_findings.md`.','small')
        self.p('Primary sources: [Sundial / TimeFlow](https://arxiv.org/html/2502.00816v2), [TSFlow, including multivariate Appendix B.8](https://arxiv.org/html/2410.03024v2), [ETDataset](https://github.com/zhouhaoyi/ETDataset), [Exchange data](https://github.com/laiguokun/multivariate-time-series-data). This study uses their motivating ideas and datasets, with explicitly different small-model protocols.','small')


def footer(canvas,doc):
    canvas.saveState();canvas.setFillColor(MUTED);canvas.setFont('RS',8)
    canvas.drawString(40,25,'FM stress test | Multivariate and real-series extension')
    canvas.drawRightString(A4[0]-40,25,str(doc.page));canvas.restoreState()


def main():
    register_fonts();r=Extension();r.opening();r.protocol();r.scores()
    for name in r.cfg['datasets']:r.dataset(name)
    r.ot();r.verification()
    dest=ROOT/'output/pdf/multivariate-real-extension.pdf'
    SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=42,
                      title='FM stress test: sparse MLP, U-Net, multivariate and real targets',
                      author='Egor Serov; Kirill Frolov; Daniil Koblov; Vasilii Lyamin').build(r.story,onFirstPage=footer,onLaterPages=footer)
    print(dest)


if __name__=='__main__':main()
