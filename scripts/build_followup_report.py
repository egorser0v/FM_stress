"""Build the audited follow-up report; never train or change measurements."""
from pathlib import Path
import csv
import json
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_report import Report, register_fonts, styles, WIDTH, NAVY, MUTED
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate


def read(path):
    return json.loads((ROOT / path).read_text())


class Followup(Report):
    def __init__(self):
        self.s = styles()
        self.story = []
        self.args = SimpleNamespace(analysis=ROOT / 'output')
        self.control = read('output/controlled-followup/summary.json')
        self.ensemble = read('results/ensemble-evaluation/summary.json')
        self.solver = read('results/ensemble-evaluation/solver_sensitivity/summary.json')
        assert self.control['verified_final_runs'] == 12 and self.control['verified_pilots'] == 8
        assert self.ensemble['complete'] and self.ensemble['observed_runs'] == 15
        audit = read('results/diagnostics/followup_verification.json')
        assert audit['status'] == 'passed' and audit['scope'] == 'both', 'Independent full audit required'

    def cell(self, head, loss, metric):
        return next(r[metric] for r in self.control['aggregate'] if r['head'] == head and r['loss'] == loss)

    def score(self, head, metric):
        return self.ensemble['heads'][head]['metrics'][metric]

    def summary(self):
        self.title('Completed follow-up · 4 October 2026', 'What changes the\nS4 / MLP comparison?', True)
        self.p('Smooth source, unstructured head', 'h2')
        self.p('Egor Serov · Kirill Frolov · Daniil Koblov · Vasilii Lyamin', 'small')
        self.p('The original experiment did not show the proposed GP-source MLP disadvantage. This follow-up tests two possible reasons: the weighting of small-variance directions during training, and the limited information supplied by a roughness score. It adds a controlled head-by-loss experiment and a fresh conditional-forecast evaluation.')
        self.table(['New evidence', 'Completed design'], [
            ['Training control', 'MLP / S4 × raw / balanced loss; identical full normalized history; 3 final seeds per cell'],
            ['Validation search', '8 pilots, then locked learning rates before final test evaluation'],
            ['Forecast evaluation', '15 original GP checkpoints; 256 fresh histories × 64 paths; Euler-128'],
            ['Numerical checks', 'Independent S4 kernel, CPU/MPS parity, saved-array recomputation, Euler-256 sensitivity']], [135, WIDTH - 135])
        m = self.cell('mlp', 'raw', 'residual_normalized_mse')['mean']
        s = self.cell('s4', 'raw', 'residual_normalized_mse')['mean']
        mn = self.cell('mlp', 'balanced', 'residual_normalized_mse')['mean']
        sn = self.cell('s4', 'balanced', 'residual_normalized_mse')['mean']
        self.p(f'**The comparison depends on the objective.** Normalized residual MSE favors MLP under raw loss ({m:.3f} versus {s:.3f}), but S4 under balanced loss ({sn:.3f} versus {mn:.3f}). S4 also preserves PC1 better than the balanced MLP. Raw residual MSE still favors MLP under both losses.')
        rawtail = self.cell('s4','raw','target_pca_pc7_16_moment_ratio')['mean']
        baltail = self.cell('s4','balanced','target_pca_pc7_16_moment_ratio')['mean']
        self.p(f'**A tradeoff, not a universal winner.** Balancing changes S4’s generated/target small-direction energy ratio from {rawtail:.1f} to {baltail:.3f}, removing the excess but now undershooting the target. Its mean raw PC1 and residual errors increase. Good roughness alone did not reveal this behavior.')
        rows = []
        for h, label in [('mlp','MLP'), ('s4','S4'), ('whitened_mlp','Whitened MLP')]:
            rows.append([label, f"{self.score(h,'fair_crps')['mean']:.4f}", f"{self.score(h,'fair_energy_score')['mean']:.4f}", f"{100*self.score(h,'coverage_90')['mean']:.1f}%"])
        self.table(['Original checkpoint head', 'CRPS ↓', 'Energy score ↓', '90% coverage'], rows, [190, 95, 110, WIDTH - 395])
        self.p('These forecast scores evaluate the original trained models on new data. They are a separate experiment from the identity-history training control; the rows must not be treated as forecast results of the newly trained models.', 'small')
        self.p('The original results and declared success flags are preserved. Follow-up metrics address relative behavior, not a retrospective replacement of the assignment’s pass criterion.', 'box')

    def audit(self):
        self.title('01 · Why further checks were needed', 'Smoothness can hide directional error')
        self.p('A curve may have nearly correct average roughness while allocating the wrong amount of energy to weak principal components. We therefore projected archived generated paths and targets into a common PCA fitted on training targets. The displayed quantity is the second moment about the training mean, including any bias; it is not conditional variance.')
        self.figure('field-diagnostics/sample_target_pc_energy', 'Archived base runs at both GP lengthscales. All 16 target PCs are retained. The PC index orders variance and should not automatically be interpreted as Fourier frequency.', 250)
        with (ROOT/'output/field-diagnostics/sample_target_pc_second_moments.csv').open() as f:
            records = list(csv.DictReader(f))
        rows = []
        for geom in ['lengthscale8','lengthscale6']:
            for head in ['mlp','s4','whitened_mlp']:
                group = [r for r in records if r['geometry']==geom and r['source']=='gp' and r['head']==head and int(r['target_pc_1based'])>=7]
                generated = sum(float(r['generated_second_moment']) for r in group)/len(group)
                observed = sum(float(r['observed_second_moment']) for r in group)/len(group)
                rows.append([geom.replace('lengthscale',''), head.replace('whitened_mlp','Whitened MLP').upper(), f'{generated:.3g}', f'{observed:.3g}', f'{generated/observed:.3g}'])
        self.table(['Lengthscale', 'Head', 'Generated', 'Target', 'Ratio'], rows, [75,145,95,95,WIDTH-410])
        self.p('Mean second moments per component in PCs 7–16. Large ratios have a very small denominator: they are not overall forecast-quality ratios. MLP suppresses part of this tail, whereas the tested S4 overproduces it; whitening is closer to the observed tail.', 'small')
        parity = read('results/diagnostics/trained_model_cpu_mps_parity.json')['runs']
        kernel = read('results/diagnostics/model_audit_trained_kernels.json')
        self.p(f"**Implementation recheck.** All 49 earlier runs passed saved-array verification. Across 15 trained S4 layers, the independent complex reference differs from the real float64 kernel by at most {max(r['float64_complex_max_abs_error'] for r in kernel):.2g}. For 15 complete GP heads, CPU/MPS maximum output disagreement is {max(r['max_abs_difference'] for r in parity.values()):.2g}. Device roundoff does not account for the observed tail discrepancy.")

    def controlled(self):
        self.title('02 · New training experiment', 'Separate the head from the loss weighting')
        self.p('Both heads receive the same complete 32-point normalized history through an identity encoder. The forecast horizon is 16; GP lengthscale is 8. MLP has 63,352 parameters and S4 has 63,169. S4 retains three temporal blocks with state size 16. This removes the separately learned history-encoder confound within the new suite.')
        self.p('**Raw loss** averages squared velocity errors. **Balanced loss** projects the error onto training velocity PCs and divides each squared component by its regularized training variance. Network inputs and outputs remain in temporal coordinates: S4 never convolves across PCA indices.')
        self.p(f"The floor is max(10⁻⁶ × leading variance, 10⁻⁸); {len(self.control['floored_pc_indices_1based'])} of 16 velocity components are floored. The weighting is therefore regularized, not exact inverse variance in every direction.", 'small')
        self.table(['Stage', 'Protocol'], [
            ['Data', 'Fresh GP draw: 32,768 train / 4,096 validation / 8,192 test; data seed 67041'],
            ['Pilots', 'Each of four cells: 4,000 updates at LR 0.001 and 0.0003, seed 20; validation only'],
            ['Selection', 'All four cells selected LR 0.001; selection locked before final test evaluation'],
            ['Final fits', 'Seeds 10, 11, 12; 8,000 updates; batch 256; best validation checkpoint per run'],
            ['Generation', '1,024 paths per run; 64 Euler steps; matched histories and source draws']], [100, WIDTH-100])
        rows=[]
        for r in self.control['aggregate']:
            vals=[f"{r[k]['mean']:.4g} ± {r[k]['std']:.2g}" for k in ['pc1_mse','residual_mse','roughness_ratio']]
            rows.append([r['head'].upper(), r['loss'], *vals])
        self.table(['Head', 'Loss', 'PC1 MSE', 'Residual MSE', 'Roughness ratio'], rows, [55,75,120,130,WIDTH-380])
        rows=[]
        for r in self.control['contrasts']:
            if r['contrast'].startswith('S4 minus MLP') and r['metric'] in ('residual_mse','residual_normalized_mse'):
                rows.append([r['contrast'], 'Raw residual' if r['metric']=='residual_mse' else 'Normalized residual',
                             f"{r['mean_difference']:+.4g}", f"[{r['ci_low']:+.4g}, {r['ci_high']:+.4g}]"])
        self.table(['Paired contrast', 'Metric', 'Difference', '95% interval'],rows,[140,130,85,WIDTH-355])
        self.p('Paired bootstrap over only three optimization seeds, conditional on this single GP dataset. Intervals are exploratory, with limited seed-level resolution; negative differences favor S4 for the stated error metric.', 'small')
        self.p('Different data, seeds, conditioning and budgets separate this suite from the original experiment. Their difference cannot identify the effect of removing the encoder alone. Two learning rates and a single pilot seed do not establish optimal tuning or convergence.', 'small')

    def controlled_directions(self):
        self.title('03 · Directional behavior', 'Does balancing also change generated paths?')
        self.figure('controlled-followup/per_pc_errors', 'Velocity prediction: mean and range over three final seeds. Right: errors divided by regularized training velocity variances. A smaller normalized error need not imply a smaller raw error.', 205)
        self.figure('controlled-followup/generated_target_pca_moments', 'Generated paths projected into a separate PCA fitted on training TARGETS. This diagnostic has no variance floor. Target and velocity PCA are distinct bases and are not interchangeable.', 205)
        rows=[]
        for r in self.control['aggregate']:
            rows.append([r['head'].upper()+' / '+r['loss'],
                         f"{r['residual_normalized_mse']['mean']:.4g}",
                         f"{r['target_pca_pc7_16_moment_ratio']['mean']:.4g}"])
        self.table(['Head / objective', 'Normalized residual MSE', 'Generated / target tail moment'],rows,[160,160,WIDTH-320])
        targettail=self.control['target_pca_path_moments'][0]['target_pc7_16_second_moment']
        self.p(f'Tail ratios sum second moments in target PCs 7–16 and average the ratio over three seeds. The target sum is only {targettail:.3g}. Values near one match this small-direction statistic, not full conditional calibration. All raw and normalized results are retained.', 'small')

    def forecasts(self):
        self.title('04 · Fresh conditional forecasts', 'Evaluate the distribution, not only its texture')
        self.p('Each of the 15 original GP checkpoints predicts 64 futures for the same 256 fresh histories. Forecasting uses 128 Euler steps, with common source samples across heads and seeds. Fair marginal CRPS measures coordinatewise predictive quality; fair energy score measures the joint forecast path. Both use off-diagonal ensemble corrections and favor smaller values.')
        self.figure('ensemble-analysis/proper_scores', 'Dots represent the five fixed training seeds. The exact GP reference conditions on raw history, information unavailable to these normalized-history networks; it is not an attainable bound for them.', 230)
        rows=[]
        for h,label in [('mlp','MLP'),('s4','S4'),('whitened_mlp','Whitened MLP')]:
            vals=[]
            for k in ['coverage_80','width_80','coverage_90','width_90']:
                v=self.score(h,k)
                vals.append(f"{100*v['mean']:.1f}%" if k.startswith('coverage') else f"{v['mean']:.3f}")
            rows.append([label,*vals])
        self.table(['Head', '80% coverage', '80% width', '90% coverage', '90% width'], rows, [135,95,85,100,WIDTH-415])
        self.p('Coverage is marginal, averaged across histories and forecast coordinates. Width is in normalized units. Narrower intervals are only useful when considered alongside coverage and proper scores; they are not automatically better.', 'small')
        rows=[]
        for r in self.ensemble['contrasts']:
            if r['metric'] in ('fair_crps','fair_energy_score'):
                rows.append([r['a'].replace('whitened_mlp','Whitened MLP').upper()+' − MLP', r['metric'].replace('fair_',''), f"{r['mean_difference']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"])
        self.table(['Contrast', 'Score', 'Difference', '95% interval'], rows, [155,115,90,WIDTH-360])
        self.p('Paired bootstrap resamples the 256 histories after averaging scores over the five fixed training seeds. These exploratory intervals condition on these models and source ensembles; they do not quantify new-training-data uncertainty and are not multiplicity-adjusted.', 'small')

    def examples(self):
        self.title('05 · Inspect forecasts and solver sensitivity', 'The same three histories for every head')
        self.figure('ensemble-analysis/fixed_context_forecasts', 'The first three fresh contexts were fixed before evaluation. Original seed 0; shared vertical scale across heads within a row. Bands are marginal 80% and 90% intervals, not simultaneous path bands. These examples illustrate behavior; aggregate scores use all 256 histories.', 365)
        self.p('Does the Euler step count change the conclusion?', 'h2')
        rows=[]
        for r in self.solver['rows']:
            old,new=r['baseline_means'],r['new_means']
            rows.append([r['head'].replace('whitened_mlp','Whitened MLP').upper(), f"{r['paired_endpoint_rmse']:.4g}", f"{new['fair_crps']-old['fair_crps']:+.4g}", f"{new['fair_energy_score']-old['fair_energy_score']:+.4g}"])
        self.table(['Head', 'Endpoint RMSE', 'Δ CRPS', 'Δ energy'], rows, [155,115,110,WIDTH-380])
        self.p('Euler-256 minus Euler-128, paired on the same 64 source samples for the first 32 fresh histories, seed 0. This checks a numerical sensitivity on a fixed subset; it is not a new head-selection experiment. Endpoint RMSE compares the two integrations, not predictions against the observed future.', 'small')

    def conclusions(self):
        self.title('06 · Interpretation and reproducibility', 'What the evidence supports')
        path=ROOT/'docs/followup_conclusions.md'
        assert path.is_file(), 'Write conclusions after auditing actual outcomes'
        for para in path.read_text().split('\n\n'):
            para=para.strip()
            if not para or para.startswith('#'): continue
            self.p(' '.join(para.splitlines()))
        self.p('Reproduce and inspect', 'h2')
        self.table(['Artifact', 'Purpose'], [
            ['configs/controlled_followup.json', 'Fixed head-by-loss training protocol'],
            ['configs/ensemble_evaluation.json', 'Fresh forecast evaluation and Euler sensitivity'],
            ['scripts/check_followup.py', 'Independent saved-array and provenance checks'],
            ['results/controlled_followup/', 'Pilots, selected rates, final checkpoints and samples'],
            ['results/ensemble-evaluation/', 'Shared inputs, ensembles and per-history scores'],
            ['output/controlled-followup/; output/ensemble-analysis/', 'Machine-readable results and figures']], [230, WIDTH-230])
        self.p('Assignment and provenance: references/project-assignment.pdf; references/project-plan.pdf; docs/model_provenance.md; docs/followup_independent_audit.md. The original task report and its success flags remain in output/pdf/research-report.pdf. Implementation uses the pinned upstream references documented there.', 'small')

    def build(self):
        for method in [self.summary,self.audit,self.controlled,self.controlled_directions,self.forecasts,self.examples,self.conclusions]: method()
        def footer(canvas, doc):
            canvas.saveState();canvas.setFont('RS',7.5);canvas.setFillColor(MUTED)
            canvas.drawString(46,25,'Smooth source, unstructured head · Audited follow-up')
            canvas.drawRightString(A4[0]-46,25,str(doc.page));canvas.restoreState()
        out=ROOT/'output/pdf/s4-mlp-followup.pdf'
        SimpleDocTemplate(str(out),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=40,
                          title='S4 / MLP: controlled follow-up',author='TS Project team').build(self.story,onFirstPage=footer,onLaterPages=footer)
        print(out)


if __name__=='__main__':
    register_fonts()
    Followup().build()
