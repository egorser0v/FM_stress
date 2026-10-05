"""Build the final eleven-page research report from complete saved experiments.

Run only after 25 main runs, 15 shorter-lengthscale runs, nine optimization checks and docs/conclusions.md exist:
  .venv/bin/python scripts/build_report.py --results results/main \
      --analysis output/analysis --output output/pdf/research-report.pdf

This script never trains, modifies results or substitutes normalized successes for
assignment-rule successes. Missing optional diagnostics are explicitly disclosed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fm_stress.analyze import collect, values, LABELS, CELLS
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image)
from PIL import Image as PILImage

NAVY=colors.HexColor('#183C4A');TEAL=colors.HexColor('#147D81')
INK=colors.HexColor('#263B46');MUTED=colors.HexColor('#5E737D')
PALE=colors.HexColor('#EDF4F4');LINE=colors.HexColor('#CEDDE1')
EXPECTED_PAGES=11
WIDTH=A4[0]-80-12  # SimpleDocTemplate frame has 6 pt inner padding per side


def read(path):return json.loads(Path(path).read_text())


def register_fonts():
    candidates=list((ROOT/'.venv/lib').glob('python*/site-packages/matplotlib/mpl-data/fonts/ttf'))
    if not candidates:
        raise RuntimeError('Embedded DejaVu fonts unavailable; install matplotlib in .venv before report export')
    d=candidates[0]
    for name,file in [('RS','DejaVuSans.ttf'),('RS-Bold','DejaVuSans-Bold.ttf'),('RS-Italic','DejaVuSans-Oblique.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(d/file)))
    pdfmetrics.registerFontFamily('RS',normal='RS',bold='RS-Bold',italic='RS-Italic',boldItalic='RS-Bold')


def styles():
    return {
        'title':ParagraphStyle('title',fontName='RS-Bold',fontSize=25,leading=29,textColor=NAVY,spaceAfter=14),
        'h1':ParagraphStyle('h1',fontName='RS-Bold',fontSize=20,leading=24,textColor=NAVY,spaceAfter=12),
        'h2':ParagraphStyle('h2',fontName='RS-Bold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=8,spaceAfter=5),
        'kicker':ParagraphStyle('kicker',fontName='RS-Bold',fontSize=8.5,leading=11,textColor=TEAL,spaceAfter=9),
        'body':ParagraphStyle('body',fontName='RS',fontSize=9.5,leading=13.2,textColor=INK,spaceAfter=8),
        'small':ParagraphStyle('small',fontName='RS',fontSize=8.1,leading=11.3,textColor=MUTED,spaceAfter=7),
        'cell':ParagraphStyle('cell',fontName='RS',fontSize=8.1,leading=10.5,textColor=INK),
        'headcell':ParagraphStyle('headcell',fontName='RS-Bold',fontSize=8,leading=10.5,textColor=colors.white),
        'box':ParagraphStyle('box',fontName='RS',fontSize=9.4,leading=13.4,textColor=NAVY,backColor=PALE,
                             borderPadding=10,spaceBefore=14,spaceAfter=18),
        'code':ParagraphStyle('code',fontName='RS',fontSize=8,leading=11.5,textColor=INK,backColor=PALE,
                              borderPadding=7,spaceBefore=6,spaceAfter=10)}


def markup(s):
    # Basic safe inline Markdown: links, bold, inline code. No raw markup input.
    tokens=[];pos=0
    for m in re.finditer(r'\[([^]]+)\]\(([^)]+)\)',s):
        tokens.append(escape(s[pos:m.start()]));tokens.append(f'<link color="#147D81" href="{escape(m.group(2))}">{escape(m.group(1))}</link>');pos=m.end()
    tokens.append(escape(s[pos:]));text=''.join(tokens)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font color="#147D81">\1</font>',text)
    return text


class Report:
    def __init__(self,args,config,records,groups):
        self.args=args;self.cfg=config;self.records=records;self.groups=groups;self.s=styles();self.story=[]
        self.geo=read(args.results/'geometry.json');self.oracle=read(args.results/'oracle.json')

    def p(self,text,style='body'):
        self.story.append(Paragraph(markup(text),self.s[style]))

    def title(self,kicker,title,first=False):
        if self.story:self.story.append(PageBreak())
        self.p(kicker.upper(),'kicker');self.p(title,'title' if first else 'h1')

    def table(self,headers,rows,widths=None):
        cells=[[Paragraph(escape(str(x)),self.s['headcell']) for x in headers]]
        cells += [[Paragraph(escape(str(x)),self.s['cell']) for x in row] for row in rows]
        table=Table(cells,colWidths=widths or [WIDTH/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LINEBELOW',(0,-1),(-1,-1),.5,LINE)]))
        self.story.extend([table,Spacer(1,9)])

    def figure(self,stem,caption,max_height=290):
        path=self.args.analysis/f'{stem}.png'
        if not path.exists():raise FileNotFoundError(f'Required final figure missing: {path}')
        with PILImage.open(path) as im:w,h=im.size
        scale=min(WIDTH/w,max_height/h)
        self.story.append(Image(str(path),width=w*scale,height=h*scale))
        self.p(caption,'small')

    def mean_sd(self,cell,key):
        vals=np.asarray([values(r)[key] for r in self.groups[cell]],float)
        return f'{vals.mean():.4g} ± {vals.std(ddof=1):.2g}'

    def summary_page(self):
        c=self.cfg
        self.title('Final research report · Tasks 1–3','Smooth source,\nunstructured head',True)
        self.p('A controlled flow-matching stress test','h2')
        self.p('Egor Serov · Kirill Frolov · Daniil Koblov · Vasilii Lyamin','small')
        self.p('**Question.** Does the choice of a white-noise or smooth Gaussian-process source interact with the architecture of a conditional velocity network? We compare a Sundial-style adaptive-LayerNorm MLP with a small TSFlow-style S4 head, then add a PCA-whitened MLP control.')
        self.p('This report presents completed experiments on synthetic, history-normalized GP patches. It evaluates a narrow design hypothesis about source and head compatibility; it is not a pretrained Sundial or full TSFlow reproduction.')
        n=len(self.records);passes=sum(r['metrics']['success']['works'] for r in self.records)
        self.table(['Evidence','Completed primary experiment'],[
            ['Design','2 sources × 2 heads, plus mandatory GP-source whitened MLP'],
            ['Replications',f'{len(c["seeds"])} matched optimization seeds per cell; {n} completed runs'],
            ['Data',f'L={c["data"]["lookback"]}, F={c["data"]["horizon"]}; RBF lengthscale {c["data"]["lengthscale"]:g}'],
            ['Training',f'{c["steps"]:,} optimizer steps/run; batch {c["batch_size"]}; {c["euler_steps"]} Euler steps'],
            ['Evaluation',f'{c["n_test"]:,} held-out velocity pairs; {c["n_paths"]:,} generated paths/run'],
            ['Literal assignment rule',f'{passes}/{n} individual runs satisfy both two-sided 20% conditions']],widths=[140,WIDTH-140])
        gp_energy=self.geo['spectra']['gp']['test']['top2_energy']
        self.p(f'**Geometry established.** The matched-GP velocity has held-out top-two PCA energy {gp_energy:.4f}, above the assignment’s 0.90 gate. The complete spectrum, source contrast and nonconstant-patch checks appear on page 3.')
        def avg(cell,key):
            return float(np.mean([values(r)[key] for r in self.groups[cell]]))
        mlp_res=avg(('gp','mlp'),'residual_mse');s4_res=avg(('gp','s4'),'residual_mse')
        res_change=100*(avg(('gp','whitened_mlp'),'residual_mse')/mlp_res-1)
        pc1_change=100*(avg(('gp','whitened_mlp'),'pc1_mse')/avg(('gp','mlp'),'pc1_mse')-1)
        rough_passes=sum(r['metrics']['success']['roughness_rule_pass'] for r in self.records)
        self.p(f'**Measured GP-source comparison.** Mean residual MSE is {mlp_res:.5f} for the raw MLP and {s4_res:.5f} for S4. Relative to the raw MLP, whitening changes residual MSE by {res_change:+.1f}% and PC1 MSE by {pc1_change:+.1f}%. These comparisons concern the implemented models and matched training budget.')
        self.p(f'**Smoothness is comparable.** {rough_passes}/{n} runs satisfy the ±20% roughness condition. The separate velocity-error equality condition determines the literal pass outcome; failing it does not by itself mean generated paths are rough or the models are unusable.')
        if passes==0:
            self.p('**Primary criterion outcome.** No run passes the literal joint rule. This is reported as a failure of the specified pass criterion, not converted into a success using another normalization. Raw errors, normalized diagnostics, oracle noise and solver checks are shown separately. Any recommendation is consequently qualified.','box')
        else:
            self.p(f'**Primary criterion outcome.** {passes} of {n} runs pass the joint rule. A pass is interpreted together with actual error magnitudes, oracle diagnostics and cross-seed variability; it does not by itself establish the claimed mechanism.','box')
        self.p('Reading guide: scientific comparison (p. 2), data and models (pp. 3–4), required results (pp. 5–6), scale/noise/solver diagnostics (pp. 7–8), robustness and reproducibility (p. 9), optimization sensitivity (p. 10), recommendation (p. 11).','small')

    def literature_page(self):
        self.title('01 · Scientific context','The missing comparison')
        self.p('**TimeFlow in Sundial** starts from white Gaussian noise and uses a small AdaLN MLP conditioned on a Transformer representation. Published forecasting results concern the pretrained system, including multi-patch prediction. GIFT-Eval evaluation uses 23 datasets, 97 configurations and 100 generated paths, with MASE and CRPS. These results support that complete system’s forecasting performance; they do not establish whether its head is equally suitable for a smooth GP source. [Liu et al., 2025](https://arxiv.org/html/2502.00816v2).')
        self.p('**TSFlow** combines GP priors with a DiffWave-style residual backbone containing S4 temporal layers; it also investigates coupling and conditional-prior choices. Its eight-dataset evaluation uses nine-quantile CRPS estimates, 100 samples and five seeds. Prior ablations within that architecture demonstrate possible source benefits, but do not answer the head question when source and architecture both differ from Sundial. [Kollovieh et al., 2025](https://arxiv.org/html/2410.03024v2).')
        self.p('**S4** supplies a structured state-space transformation along time. Its long-sequence modeling results motivate a temporal alternative to patchwise dense regression, but do not imply an advantage at F=16. A controlled short-horizon experiment is needed. [Gu, Goel and Ré, 2022](https://arxiv.org/abs/2111.00396).')
        self.table(['Aspect','TimeFlow / Sundial','TSFlow','This experiment'],[
            ['Source','White Gaussian noise','GP prior','White and matched-kernel GP'],
            ['Velocity/transport head','AdaLN MLP','DiffWave-style S4','Small AdaLN MLP / 3-block S4'],
            ['Conditioning','Pretrained Transformer','Forecasting backbone','Same tiny encoder design'],
            ['Evidence','Full-system forecast scores','Forecast scores; prior ablations','Velocity-PC errors; path roughness'],
            ['Question left open','GP source under MLP','Same source under matched heads','Source × head interaction']],widths=[76,137,137,WIDTH-350])
        self.p('The initially disputed cell is **smooth source → smooth target with an unstructured MLP**, under independent coupling. White noise → smooth target is the published reference geometry, not the disputed cell. TSFlow is not a direct counterexample until heads and experimental conditions are matched.')
        self.p('Low-rank velocity covariance alone does not prove poor optimization. Flow-matching labels are random conditional targets; the learned field estimates their conditional mean. Whitening changes coordinates and the training loss, while S4 changes the function class and temporal bias. These interventions allow a local comparison, not automatic causal attribution. [Lipman et al., 2023](https://arxiv.org/abs/2210.02747).')
        self.p('The standalone one-page critical synthesis is supplied as `output/pdf/related-work.pdf`; implementation revisions and adaptations are recorded in `docs/model_provenance.md`.','small')

    def data_page(self):
        c=self.cfg;g=c['data']
        self.title('02 · Data and acceptance gate','Verify the geometry before training')
        self.p(f'Each example is a joint GP draw containing {g["lookback"]} observed and {g["horizon"]} future points. The squared-exponential covariance has lengthscale {g["lengthscale"]:g}, variance {g.get("variance",1):g} and diagonal nugget {g.get("nugget",1e-6):g}. Windows are independent. The primary split contains {c["n_train"]:,} training, {c["n_val"]:,} validation and {c["n_test"]:,} test windows.')
        self.p('Sundial’s released normalization is applied exactly: subtract the history mean and divide by its population standard deviation; replace a standard deviation ≤0.01 with 1. The future uses those same history statistics. No future value contributes to normalization.')
        self.p('Sources are independently drawn in normalized observation coordinates. “Matched GP” means the same kernel as the raw generator. After random context normalization, target and source marginal covariances are not identical; the report does not claim otherwise.')
        rows=[]
        for key,label in [('target','Target patches'),('white','White-source velocity'),('gp','GP-source velocity')]:
            stat=self.geo['spectra'][key]['test']
            rows.append([label,f'{stat["top1_energy"]:.4f}',f'{stat["top2_energy"]:.4f}',f'{stat["mean_within_patch_std"]:.4f}',f'{stat["roughness"]:.4f}'])
        self.table(['Held-out quantity','PC1 share','PC1+2 share','Within-patch SD','Roughness'],rows,widths=[165,75,85,95,WIDTH-420])
        self.figure('geometry_spectra','Figure 1. Test variance is projected into axes fitted only on training draws. The matched-GP velocity passes the geometry gate; white-source energy provides the numerical contrast. Nonzero within-patch variation rules out constant-series collapse.',max_height=255)
        self.p('Lengthscale 8 was chosen after a preliminary geometry pilot, before main model training. Primary data use a fresh seed. The supplementary lengthscale scan is disclosed as sensitivity analysis, not retroactive preregistration. The final test set did not determine the chosen lengthscale or checkpoints.','small')

    def models_page(self):
        c=self.cfg
        self.title('03 · Implementation and controlled training','Matched capacity, explicit adaptations')
        self.p('The MLP adapts the released TimeFlow architecture: sinusoidal time embedding, AdaLN residual blocks and a conditional final readout. The S4 head uses three residual temporal blocks with genuine HiPPO-initialized normal-plus-low-rank S4, bidirectional temporal mixing, gated residual/skip branches and a small final readout.')
        rows=[]
        for cell in [('white','mlp'),('white','s4'),('gp','whitened_mlp')]:
            rows.append([LABELS[cell],f'{self.groups[cell][0]["parameters"]:,}',str(c['model'].get('depth',3)),str(c['model'].get('width','—'))])
        self.table(['Head','Parameters','Blocks','Width'],rows,widths=[245,105,80,WIDTH-430])
        self.p('Both heads receive the same encoder architecture and identical initial encoder weights for each seed; each copy then trains jointly with its own head. Thus final encoder representations can differ. Capacity and training opportunity are controlled, but this is not a frozen, literally shared encoder comparison.')
        self.p(f'All runs use AdamW, learning rate {c["learning_rate"]:g}, batch {c["batch_size"]}, gradient clipping {c["clip_grad"]:g} and {c["steps"]:,} steps. SSM parameters retain no-weight-decay treatment. Validation selects checkpoints in each model’s training coordinates; test labels are used only after selection. The whitened MLP therefore has a rescaled training/selection objective.')
        self.figure('learning_curves','Figure 2. Thin curves show individual optimization seeds; thick curves show their means. Raw validation MSE is comparable in observation coordinates. The whitening objective has different coordinate weighting.',max_height=235)
        self.p('**MPS adaptation.** Complex HiPPO initialization runs once on CPU. Trainable S4 kernels use real state matrices, exact bilinear discretization and finite-horizon Krylov kernels on MPS. Numerical tests compare kernels and gradients against an independent complex reference. This implementation targets short patches, not long-sequence speed claims.')
        self.p('**Objective disclosure.** Released Sundial code predicts targets with its own weighted objective. This project reuses architecture but follows the assignment’s explicit velocity target u=y−ε and independent interpolation. No pretrained weights or TimeBench pretraining are used.','small')

    def primary_page(self):
        self.title('04 · Required Task 2 / Task 3 results','Three metrics, five completed cells')
        self.p('Velocity error is projected into a source-specific PCA basis fitted to training velocities. PC1 measures the dominant direction; residual MSE averages per-component errors over PCs 3–F. PC2 is separately disclosed on page 7. Generated roughness is the mean absolute adjacent jump after the fixed Euler budget.')
        rows=[]
        for cell in CELLS:
            passes=sum(r['metrics']['success']['works'] for r in self.groups[cell])
            rows.append([LABELS[cell],self.mean_sd(cell,'pc1_mse'),self.mean_sd(cell,'residual_mse'),self.mean_sd(cell,'generated_roughness'),self.mean_sd(cell,'roughness_ratio'),f'{passes}/{len(self.groups[cell])}'])
        self.table(['Cell','PC1 MSE','PCs 3–F MSE','Roughness','R / target','Passes'],rows,widths=[114,79,96,87,82,WIDTH-458])
        target=self.records[0]['metrics']['target_roughness']
        self.p(f'Mean ± sample SD across {len(self.cfg["seeds"])} optimization seeds. Target roughness = {target:.6f}. All generated paths and velocity errors are evaluated in the original normalized observation coordinates, including the inverse-mapped whitening control.','small')
        self.figure('required_metrics','Figure 3. Points show individual seeds; summary uncertainty is seed SD. The green band marks the ±20% roughness interval. The first two panels show raw error magnitudes, not normalized substitutes.',max_height=245)
        self.p('**Literal success rule:** 0.8 ≤ residual MSE / PC1 MSE ≤ 1.2 AND 0.8 ≤ generated roughness / target roughness ≤ 1.2.','box')
        self.p('The rule is two-sided: residual error far below PC1 error also fails it. Because the GP spectrum is strongly anisotropic, this equality condition does not generally coincide with accurate regression. Pass counts are retained exactly; the following diagnostics examine this limitation without rewriting the criterion.')
        self.p('Paired 95% bootstrap intervals for all source/head and whitening contrasts are released in `output/analysis/paired_seed_contrasts.csv`. They resample matched seeds (10,000 resamples) and are conditional on this fixed dataset and hyperparameters. Five seeds provide limited interval resolution.','small')

    def paths_page(self):
        self.title('05 · Generated trajectories','The same contexts in every cell')
        self.p('These are the first three held-out contexts (indices 0, 1 and 2) for seed 0, fixed by index rather than selected for appearance. The final eight observed history points are gray; the observed future draw is dashed; the generated future is solid. Each column uses the same vertical scale across all five models.')
        self.figure('generated_paths_report','Figure 4. All four required cells and the whitening control are shown. The generated and observed futures are separate stochastic draws under the same context. Their pointwise difference is not treated as a unique deterministic forecast error.',max_height=500)
        self.p('The full six-context comparison is supplied as `generated_paths_all_cells.pdf`; the four-cell overview is `generated_paths_2x2.pdf`. The report uses fewer columns solely to keep axis labels legible. Complete arrays, histories and source draws are saved in `results/main/samples/`; plotted indices and provenance are recorded in `path_figure_manifest.json`.','small')

    def diagnostics_page(self):
        self.title('06 · Scale-sensitive diagnostic','Where the velocity error lies')
        self.figure('velocity_errors_per_pc','Figure 5. Mean raw and variance-normalized per-PC errors; shaded bands are seed SD. Gray dashed curves, when present, are the privileged GP oracle. The residual metric averages PCs 3–F; PC2 is retained visibly.',max_height=365)
        rows=[]
        for cell in CELLS:
            rows.append([LABELS[cell],self.mean_sd(cell,'pc2_mse'),self.mean_sd(cell,'pc1_normalized_mse'),self.mean_sd(cell,'residual_normalized_mse')])
        self.table(['Cell','PC2 raw MSE','PC1 normalized','Residual normalized'],rows,widths=[160,105,120,WIDTH-385])
        self.p('Normalized error divides each PC MSE by its training velocity variance, floored at max(10⁻⁶ λ₁, 10⁻⁸), before averaging residual PCs. The same floor defines whitening. This secondary measure reveals errors in small-variance directions but is not the assignment’s primary raw MSE or a replacement pass criterion.')
        gp_pca=self.geo['pca']['gp'];eigenvalues=np.asarray(gp_pca['eigenvalues'],float)
        floor=float(gp_pca['floor']);floored=int(np.sum(eigenvalues<floor))
        self.p(f'**Actual GP floor:** {floor:.6g}; {floored}/{len(eigenvalues)} training-velocity directions fall below it. Their normalized errors use the floor rather than their smaller empirical variance. This regularization materially affects interpretation of the near-null tail and is shared with the whitening control.','small')
        self.p('Whitening uses an invertible linear map fitted on training velocities. Input patches, source, velocity and integrated outputs are transformed consistently; velocity targets are not incorrectly centered by a fixed mean. Whitening retains all coordinates and changes their scale—it does not by itself prove that F−2 dimensions were unused.','small')

    def robustness_records(self):
        candidates=list((self.args.results/'robustness').glob('*.json'))
        candidates+=list((ROOT/'results/diagnostics').glob('*.json'))
        learned={};oracle=None
        for p in sorted(candidates):
            d=read(p)
            if d.get('config_sha256') not in (None,self.records[0].get('config_sha256')):
                continue
            if d.get('status') not in ('complete',None):continue
            for name,record in d.get('runs',{}).items():
                if 'rows' in record:learned[name]=record
            if 'oracle' in d and 'sources' in d['oracle']:oracle=d['oracle']
            elif 'sources' in d and 'reference_note' in d:oracle=d
        return learned,oracle

    def oracle_page(self):
        self.title('07 · Oracle and solver audit','Separate regression noise from integration error')
        self.p('Conditional-flow labels u=y−ε remain random given an interpolated state and context. Even the optimal conditional mean field can have nonzero label MSE. A Gaussian analytic diagnostic is computed using the complete raw history and its normalization statistics.')
        rows=[]
        for source,label in [('white','White source'),('gp','GP source')]:
            r=self.oracle['sources'][source]
            rows.append([label,f'{r["pc1_mse"]:.5g}',f'{r["residual_mse"]:.5g}',f'{r["success"]["residual_to_pc1"]:.5g}',str(r['success']['works'])])
        self.table(['Privileged oracle','PC1 MSE','Residual MSE','Residual / PC1','Literal pass'],rows,widths=[145,95,95,100,WIDTH-435])
        self.p('**Information advantage.** Learned heads see only normalized history, whereas the oracle sees the raw history and its location/scale. Its error is a privileged lower-bound diagnostic, not an attainable guarantee. The normalized target is not globally Gaussian; conditioning on raw history makes its affine Gaussian law exact.','box')
        learned,oracle=self.robustness_records()
        if learned:
            rows=[]
            for name,r in sorted(learned.items()):
                primary=next((x for x in r['rows'] if x['euler_steps']==self.cfg['euler_steps']),r['rows'][0])
                last=max(r['rows'],key=lambda x:x['euler_steps'])
                rows.append([LABELS.get((r['source'],r['kind']),name),f"{primary['roughness_to_target']:.3f}",f"{last['roughness_to_target']:.3f}",f"{primary['paired_endpoint_rmse_over_target_rms']:.4f}"])
            self.table(['Saved-checkpoint solver check',f'R ratio K={self.cfg["euler_steps"]}','R ratio largest K','Endpoint RMSE / target RMS'],rows,widths=[177,105,105,WIDTH-387])
            self.p('CPU check on the first 256 test histories, seed 0, using saved MPS-trained checkpoints. K=64 values differ from the primary 1,024-path estimates. Endpoint RMSE compares K=64 with K=256 under identical sources; primary scores are unchanged.','small')
        else:self.p('Learned-checkpoint step-sensitivity records were not available at export; no learned-solver convergence claim is made.','small')
        if oracle:
            rows=[]
            for source,records in oracle['sources'].items():
                for row in records:
                    rows.append([source,str(row['euler_steps']),f"{row['analytic_expected_roughness_to_exact']:.4f}",f"{row['conditional_trace_ratio_mean']:.4f}",f"{row['fraction_conditional_directions_below_80pct_variance']:.3f}"])
            self.table(['Oracle source','K','Expected R / exact','Conditional variance trace ratio','Directions <80% variance'],rows,widths=[100,45,115,135,WIDTH-395])
            self.p('Analytic covariance propagation can reveal suppressed conditional variance even when roughness looks accurate. This is a solver limitation of the near-singular Gaussian problem; accurate mean-path roughness alone does not establish distributional calibration.','small')
        else:self.p('Analytic oracle Euler covariance diagnostics were not available at export. The exact conditional sampler and regression-noise diagnostics remain separate from any finite-step ODE assertion.','small')

    def supplementary_page(self):
        self.title('08 · Robustness and reproducibility','What changes outside the primary setting')
        sensitivity=ROOT/'results/geometry-sensitivity/geometry_scan.json'
        if sensitivity.exists():
            d=read(sensitivity);rows=[]
            for ell in d['lengthscales']:
                r={x['quantity']:x for x in d['results'] if x['lengthscale']==ell and x['split']=='test'}
                rows.append([str(ell),f"{r['target']['top2_energy']:.4f}",f"{r['white']['top2_energy']:.4f}",f"{r['gp']['top2_energy']:.4f}"])
            self.table(['Lengthscale','Target E₂','White velocity E₂','GP velocity E₂'],rows,widths=[100,135,140,WIDTH-375])
            self.p('Supplementary geometry scan, with independently seeded draws and training-only PCA. It does not select a trained winner. Long lengthscales can make white-source velocities nearly rank two as normalization inflates future variance, so smoothness alone does not guarantee the desired source contrast.','small')
        alt=self.args.lengthscale_results
        if (alt/'config.json').exists():
            ac,ar,ag,status=collect(alt)
            if status['complete']:
                rows=[]
                for cell in CELLS:
                    group=ag[cell]
                    if not group:continue
                    row=[LABELS[cell],str(len(group))]
                    for key in ('pc1_mse','residual_mse','roughness_ratio'):
                        a=np.asarray([values(r)[key] for r in group]);row.append(f'{a.mean():.4g}')
                    rows.append(row)
                self.table([f'Lengthscale {ac["data"]["lengthscale"]:g}','Seeds','PC1 MSE','Residual MSE','R / target'],rows,widths=[160,45,105,105,WIDTH-415])
                alt_geo=read(alt/'geometry.json')
                self.p(f'GP-velocity test E₂={alt_geo["spectra"]["gp"]["test"]["top2_energy"]:.4f}. Literal passes: {sum(r["metrics"]["success"]["works"] for r in ar)}/{len(ar)}; roughness passes: {sum(r["metrics"]["success"]["roughness_rule_pass"] for r in ar)}/{len(ar)}. Shortening the lengthscale does not resolve the raw equality criterion.','small')
                self.p('This second geometry is a labeled follow-up after the primary criterion audit. Its own configuration, split seed and runs are preserved. It does not overwrite the primary result or restore an architecture claim simply by changing the test.','small')
            else:self.p('The shorter-lengthscale follow-up was incomplete at export and is excluded from quantitative conclusions.','small')
        else:self.p('A shorter-lengthscale trained follow-up is not included in this export. The data-only sensitivity scan does not replace such an experiment.','small')
        self.p('Reproduction and leakage audit','h2')
        self.p('Every trained run records seed, parameter count, validation-selected step, device, elapsed time and configuration hash. Test windows, sources and times are shared across matched comparisons; no test metric selects checkpoints. Checkpoints, training curves, PCA arrays and per-seed metrics remain available. The optional real-series extension is omitted; conclusions stay within the declared synthetic geometry.')
        self.p('Main commands: `python -m fm_stress.experiment train --config configs/main.json --output results/main --device mps`; then `python -m fm_stress.analyze --results results/main --output output/analysis --require-complete`. Geometry and oracle subcommands reproduce their separate records.','small')

    def optimization_page(self):
        self.title('09 · Additional optimization check','How sensitive is the ranking to training?')
        records=[read(p) for p in sorted((self.args.optimization_results/'runs').glob('*.json'))]
        groups={head:[r for r in records if r['head']==head] for head in ('mlp','s4','whitened_mlp')}
        first=records[0];options=first['options']
        nseeds=len({r['seed'] for r in records})
        self.p(f'This is a separate post-primary sensitivity analysis: the three GP-source heads receive {options["additional_steps"]:,} extra optimizer steps at learning rate {options["learning_rate"]:g}, starting from their validation-selected checkpoints with Adam state reset. The original checkpoint remains eligible for selection. Main results and checkpoints are not overwritten.')
        self.p(f'The check first ran on seed 0. A validation-only improvement/ranking trigger expanded it to seeds 1 and 2, giving {nseeds} matched seeds and {len(records)} warm restarts. It is an adaptively extended diagnostic, not an independent preregistered confirmation or a replacement for the five-seed primary table.')
        def fmt(v):
            a=np.asarray(v,float)
            return f'{a.mean():.4g} ± {a.std(ddof=1):.2g}'
        rows=[]
        for head,rs in groups.items():
            rows.append([LABELS[('gp',head)],fmt([r['initial_validation_raw_mse'] for r in rs]),
                         fmt([r['best_validation_raw_mse'] for r in rs]),
                         fmt([100*r['validation_improvement_fraction'] for r in rs])])
        self.table(['GP-source head','Initial validation raw MSE','Selected validation raw MSE','Own objective improvement (%)'],rows,widths=[145,120,120,WIDTH-385])
        self.p('Mean ± seed SD. Raw validation MSE has common observation-space units. The final column measures within-model improvement in its selection objective; whitening uses a differently weighted objective, so those percentages are not a common cross-model score.','small')
        initial_order=sorted(groups,key=lambda h:np.mean([r['initial_validation_raw_mse'] for r in groups[h]]))
        final_order=sorted(groups,key=lambda h:np.mean([r['best_validation_raw_mse'] for r in groups[h]]))
        names={'mlp':'MLP','s4':'S4','whitened_mlp':'whitened MLP'}
        before=' < '.join(names[h] for h in initial_order);after=' < '.join(names[h] for h in final_order)
        self.p(f'**Validation raw-MSE ordering, lower is better:** before restart, {before}; after restart, {after}. This ordering describes these three seeds and the bounded additional budget. It does not establish convergence or a universal architecture ranking.','box')
        rows=[]
        for head,rs in groups.items():
            rows.append([names[head],fmt([r['metrics']['pc1_mse'] for r in rs]),
                         fmt([r['metrics']['residual_mse'] for r in rs]),
                         fmt([r['metrics']['success']['roughness_ratio'] for r in rs]),
                         f"{sum(r['metrics']['success']['works'] for r in rs)}/{len(rs)}"])
        self.table(['Restarted head','Test PC1 MSE','Test residual MSE','R / target','Literal passes'],rows,widths=[130,105,110,100,WIDTH-445])
        self.p('Test labels are evaluated only after the additional validation selection. These are supplemental outcomes of the changed training schedule. The same original-coordinate metrics and literal two-sided 20% rule are retained; an improved validation score is not relabeled as passing the assignment rule.')
        self.p('For exact per-seed before/after metrics, saved states, selected extra steps and learning curves, see `results/main/optimization_sensitivity/`. The validation-based expansion decision is recorded alongside the runs. Finite-budget sensitivity limits a mechanistic explanation based solely on the initial head comparison.','small')

    def conclusion_page(self):
        self.title('10 · Interpretation and recommendation','A design recommendation with its limits')
        text=self.args.conclusions.read_text().strip()
        # Read the authored, final evidence-based interpretation. Do not fabricate it.
        if re.search(r'\b(TODO|TBD|PLACEHOLDER)\b',text,re.I):raise ValueError('Conclusions contain unfinished placeholders')
        for para in re.split(r'\n\s*\n',text):
            para=para.strip()
            if para.startswith('#'):
                title=para.lstrip('# ').strip()
                if title.lower() not in ('conclusions','conclusions and recommendation','recommendation'):
                    self.p(title,'h2')
            else:
                self.p(para.replace('\n',' '))
        self.p('Source and implementation references','h2')
        self.p('[Sundial / TimeFlow](https://arxiv.org/abs/2502.00816), ICML 2025; [TSFlow](https://arxiv.org/abs/2410.03024), ICLR 2025; [S4](https://arxiv.org/abs/2111.00396), ICLR 2022; [Flow Matching](https://arxiv.org/abs/2210.02747), ICLR 2023. The two supplied assignment PDFs define the project’s required geometry, metrics, 20% rule and control.','small')
        self.p('Official code revisions: Sundial `3212e42564493f520593e5414af4367fc4b49226`; state-spaces/S4 `e757cef57d89e448c413de7325ed5601aceaac13`; inspected TSFlow `ff893176a41a1037f63efae674aab71480eee7d0`. Apache-2.0 reference sources and licenses are preserved for Sundial and S4. TSFlow’s unlicensed repository source is inspected, not vendored. See `vendor/manifest.json` and `docs/model_provenance.md` for checksums and adaptations.','small')
        self.p('AI assistance was used for implementation, analysis tooling and writing. Numerical equivalence tests, independent known-value metric tests, split/geometry checks and saved experiment records provide verification; the report does not treat generated prose as evidence.','small')

    def build(self):
        for method in (self.summary_page,self.literature_page,self.data_page,self.models_page,self.primary_page,
                       self.paths_page,self.diagnostics_page,self.oracle_page,self.supplementary_page,self.optimization_page,self.conclusion_page):method()
        self.args.output.parent.mkdir(parents=True,exist_ok=True)
        page_count=[]
        def footer(canvas,doc):
            page_count.append(doc.page)
            canvas.setStrokeColor(LINE);canvas.line(40,35,A4[0]-40,35)
            canvas.setFont('RS',7.5);canvas.setFillColor(MUTED)
            canvas.drawString(40,22,'Smooth source, unstructured head  ·  Synthetic flow-matching stress test')
            canvas.drawRightString(A4[0]-40,22,f'{doc.page} / {EXPECTED_PAGES}')
        doc=SimpleDocTemplate(str(self.args.output),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=48,
            title='Smooth source, unstructured head: experimental research report',
            author='Egor Serov; Kirill Frolov; Daniil Koblov; Vasilii Lyamin')
        doc.build(self.story,onFirstPage=footer,onLaterPages=footer)
        if max(page_count)!=EXPECTED_PAGES:
            raise RuntimeError(f'Report layout overflow: expected {EXPECTED_PAGES} pages, got {max(page_count)}. Inspect and repair before delivery.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results/main')
    p.add_argument('--analysis',type=Path,default=ROOT/'output/analysis')
    p.add_argument('--lengthscale-results',type=Path,default=ROOT/'results/lengthscale6')
    p.add_argument('--optimization-results',type=Path,default=ROOT/'results/main/optimization_sensitivity')
    p.add_argument('--conclusions',type=Path,default=ROOT/'docs/conclusions.md')
    p.add_argument('--output',type=Path,default=ROOT/'output/pdf/research-report.pdf')
    args=p.parse_args()
    config,records,groups,status=collect(args.results)
    if not status['complete'] or len(records)!=25 or len(config.get('seeds',[]))!=5:
        raise SystemExit('Refusing final export: main experiment must have all 25 runs (five cells × five seeds) and sample archives.')
    if not (args.lengthscale_results/'config.json').exists():
        raise SystemExit('Refusing final export: the declared shorter-lengthscale follow-up is missing.')
    alt_config,alt_records,_,alt_status=collect(args.lengthscale_results)
    if not alt_status['complete'] or len(alt_records)!=15 or len(alt_config.get('seeds',[]))!=3:
        raise SystemExit('Refusing final export: lengthscale follow-up must have all 15 runs (five cells × three seeds).')
    if not args.conclusions.exists() or not args.conclusions.read_text().strip():
        raise SystemExit('Refusing final export: authored docs/conclusions.md is required.')
    required=['geometry.json','oracle.json','config.json']
    for name in required:
        if not (args.results/name).exists():raise FileNotFoundError(args.results/name)
    actual_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if actual_hash!=status.get('config_sha256'):
        raise SystemExit('Refusing final export: saved config and run hashes disagree.')
    geometry=read(args.results/'geometry.json')
    if geometry.get('config_sha256')!=actual_hash or not geometry.get('gate_pass'):
        raise SystemExit('Refusing final export: primary geometry gate/hash is invalid.')
    if any(record.get('device')!='mps' for record in records):
        raise SystemExit('Refusing MPS experiment report: at least one primary run did not use MPS.')
    analysis_status=read(args.analysis/'analysis_status.json')
    if not analysis_status['complete'] or analysis_status.get('config_sha256')!=status.get('config_sha256'):
        raise SystemExit('Refusing final export: analysis must be complete and match main run configuration.')
    optimization_records=[read(p) for p in sorted((args.optimization_results/'runs').glob('*.json'))]
    expected_opt={(head,seed) for head in ('mlp','s4','whitened_mlp') for seed in (0,1,2)}
    actual_opt={(r['head'],r['seed']) for r in optimization_records}
    if len(optimization_records)!=9 or actual_opt!=expected_opt or any(r.get('status')!='complete' or r.get('config_sha256')!=actual_hash for r in optimization_records):
        raise SystemExit('Refusing final export: all nine matching optimization-sensitivity runs are required.')
    register_fonts();report=Report(args,config,records,groups);report.build()
    manifest={'report':str(args.output),'pages':EXPECTED_PAGES,'runs':len(records),'config_sha256':status['config_sha256'],
              'results':str(args.results),'analysis':str(args.analysis),
              'conclusions_sha256':hashlib.sha256(args.conclusions.read_bytes()).hexdigest(),
              'lengthscale_results':str(args.lengthscale_results),'lengthscale_runs':len(alt_records),
              'optimization_runs':len(optimization_records),'note':'Eleven-page count verified by report layout; visually render all pages before delivery.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
