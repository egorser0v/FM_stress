#!/usr/bin/env python3
"""Build one coherent PDF/HTML report from completed follow-up artifacts."""
import argparse, hashlib, html, json, re, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_report import register_fonts, markup
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from PIL import Image as PILImage


def read(p):return json.loads((ROOT/p).read_text())
def fmt(x,n=4):return f'{x:.{n}f}'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_gate():
    """A final report requires complete independent checks, not only run counts."""
    specifications = [
        ('results/followup-reproduction/all_summary.json', lambda s: s.get('complete') and s.get('passed')),
        ('results/followup-oracle/independent_verification.json', lambda s: s.get('status') == 'passed'),
        ('results/followup-oracle/finite_perturbation/independent_arithmetic_audit.json', lambda s: s.get('status') == 'passed' and s.get('rows') == 250 and s.get('all_frozen_inputs_unchanged')),
        ('results/followup-capacity/independent_verification.json', lambda s: s.get('complete') and s.get('verified_records') == 72 and not s.get('failures')),
        ('results/followup-normalization-audit/summary.json', lambda s: s.get('complete') and s.get('passed') and s.get('verified_fits') == 84),
        ('results/followup-normalization-audit/analysis.json', lambda s: s.get('complete') and s.get('passed') and s.get('newer_unanalyzed_fits') == 0 and s.get('analysis_sha256') == sha(ROOT/'output/followup-normalization/summary.json')),
        ('results/followup-label-noise/stream_verification.json', lambda s: s.get('complete') and s.get('status') == 'passed'),
        ('results/followup-label-noise/independent_verification.json', lambda s: s.get('complete') and len(s.get('records', [])) == 24 and s.get('status') == 'passed available runs'),
        ('results/followup-label-noise/final_hash_verification.json', lambda s: s.get('complete') and s.get('status') == 'passed'),
        ('results/followup-label-solver/summary.json', lambda s: s.get('complete') and s.get('observed') == 24 and s.get('frozen_inputs_unchanged')),
        ('results/followup-label-solver/independent_verification.json', lambda s: s.get('complete') and s.get('status') == 'passed' and len(s.get('records',[])) == 24),
        ('results/followup-label-solver/cross_review_verification.json', lambda s: s.get('complete') and s.get('status') == 'passed' and not s.get('failures') and s.get('frozen_inputs_unchanged') and s.get('summary_sha256') == sha(ROOT/'results/followup-label-solver/summary.json') and s.get('analysis_sha256') == sha(ROOT/'output/followup-label-solver/summary.json')),
        ('results/followup-solver/summary.json', lambda s: s.get('complete') and s.get('models') == 36),
        ('results/followup-solver/cross_review_verification.json', lambda s: s.get('complete') and s.get('status') == 'passed' and not s.get('failures') and s.get('summary_sha256') == sha(ROOT/'results/followup-solver/summary.json')),
        ('results/followup-capacity-baselines/summary.json', lambda s: s.get('complete') and s.get('groups') == 12 and s.get('max_crps_formula_disagreement', 1) < 1e-10),
        ('results/followup-calibration-strata/summary.json', lambda s: s.get('complete') and s.get('validated_fits') == 84 and s.get('budget_records') == 168 and s.get('aggregate_sha256') == sha(ROOT/'output/followup-normalization/summary.json') and s.get('frozen_manifest_sha256') == sha(ROOT/'results/followup-calibration-strata/manifest.json')),
    ]
    issues=[]
    for path,check in specifications:
        if not (ROOT/path).exists() or not check(read(path)):
            issues.append(path)
    if issues:
        raise ValueError('Final independent audits are incomplete or stale: '+', '.join(issues))
    test_log=(ROOT/'results/diagnostics/followup_full_pytest.txt').read_text()
    if not re.search(r'211 passed in [0-9.]+s\s*$',test_log):
        raise ValueError('Final project test suite has not passed')


def consumed_artifacts():
    paths=[
        'output/followup-normalization/summary.json',
        'output/followup-capacity/summary.json',
        'output/followup-label-noise/summary.json',
        'results/followup-oracle/summary.json',
        'results/followup-oracle/finite_perturbation/results.json',
        'results/followup-oracle/finite_perturbation/independent_arithmetic_audit.json',
        'results/followup-reproduction/all_summary.json',
        'results/followup-reproduction/scope_geometry.json',
        'results/followup-oracle/independent_verification.json',
        'results/followup-capacity/independent_verification.json',
        'results/followup-normalization-audit/summary.json',
        'results/followup-normalization-audit/analysis.json',
        'results/followup-label-noise/stream_verification.json',
        'results/followup-label-noise/independent_verification.json',
        'results/followup-label-noise/predictive_baselines.json',
        'results/followup-label-noise/final_hash_verification.json',
        'results/followup-label-solver/summary.json',
        'results/followup-label-solver/manifest.json',
        'results/followup-label-solver/independent_verification.json',
        'results/followup-label-solver/cross_review_verification.json',
        'output/followup-label-solver/summary.json',
        'results/diagnostics/followup_full_pytest.txt',
        'output/followup-capacity/fixed_context_paths.json',
        'docs/followup_capacity_execution_note.txt',
        'results/followup-normalization/execution_notes.txt',
        'results/followup-solver/summary.json',
        'results/followup-solver/cross_review_verification.json',
        'results/followup-solver/config.json',
        'results/followup-solver/manifest.json',
        'results/followup-capacity-baselines/summary.json',
        'output/followup-capacity/vasya_geometry_bridge.json',
        'results/followup-calibration-strata/summary.json',
        'results/followup-calibration-strata/manifest.json',
        'results/followup-calibration-strata/config.json',
        'results/followup-calibration-strata/per_seed.json',
        'README.md',
        'docs/kernel_scan_findings.md',
        'docs/final_report_addendum.md',
        'docs/followup_normalization_final_review.md',
        'docs/followup_research_plan.md',
        'output/followup-research/README_RU.txt',
        'results/diagnostics/followup_delivery_state.json',
    ]
    paths.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'results/followup-label-noise').glob('dataset_*/*/runs/*.json')))
    for folder in ['followup-reproduction','followup-oracle','followup-capacity','followup-normalization','followup-label-noise','followup-calibration-strata']:
        paths.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'output'/folder).glob('*.png')))
    return {path:sha(ROOT/path) for path in paths if (ROOT/path).exists()}


def readable(text):
    # Typeset prose consistently without changing numerical data or file paths.
    text=re.sub(r'\b([a-z]{2,})(?=[0-9])',r'\1 ',text)
    text=re.sub(r'\b(All|Within|Across|Uses|Approximately|Euler|MPS|CPU|MLP)(?=[0-9])',r'\1 ',text)
    text=re.sub(r'([0-9])(?!(?:st|nd|rd|th)\b)([a-z]{2,})\b',r'\1 \2',text)
    text=re.sub(r'(?<=[A-Za-z\)]),(?=\S)',', ',text)
    text=re.sub(r'\s*×\s*',' × ',text)
    text=re.sub(r';(?=\S)','; ',text)
    return text


class Report:
    def __init__(self):self.pages=[]
    def page(self,title,kicker='Follow-up study'):
        self.pages.append({'title':title,'kicker':kicker,'blocks':[]})
    def p(self,text):self.pages[-1]['blocks'].append(('p',readable(text)))
    def table(self,headers,rows):self.pages[-1]['blocks'].append(('table',(headers,[[readable(str(v)) for v in row] for row in rows])))
    def image(self,path,caption):self.pages[-1]['blocks'].append(('image',(path,readable(caption))))


def content(partial=False):
    norm=read('output/followup-normalization/summary.json');cap=read('output/followup-capacity/summary.json')
    oracle=read('results/followup-oracle/summary.json');replay=read('results/followup-reproduction/all_summary.json')
    label=[]
    for p in sorted((ROOT/'results/followup-label-noise').glob('dataset_*/*/runs/*.json')):label.append(json.loads(p.read_text()))
    complete=norm['complete'] and cap['complete'] and len(label)==24 and replay['passed'] and oracle['all_quadratures_converged']
    if not complete and not partial:raise ValueError('All studies must complete before final report export')
    if complete:audit_gate()
    cg={(r['kernel'],r['source'],r['head'],r['budget']):r for r in cap['cells']}
    ng=norm['table'];lg={}
    for r in label:lg.setdefault((r['source'],r['head'],r['label']),[]).append(r)
    def cm(k,s,h,b,m):return cg.get((k,s,h,b),{}).get('metrics',{}).get(m,{}).get('mean',float('nan'))
    def nm(mode,p,h,m='fair_crps',b=6000):return ng.get(f'{mode}|{p}|{h}|{b}',{}).get('mean',{}).get(m,float('nan'))
    def lm(s,h,l,m='fair_crps'):
        rows=lg.get((s,h,l),[])
        return float(np.mean([r['forecast_means'][m] for r in rows])) if rows else float('nan')
    r=Report()
    r.page('Which differences survive a controlled follow-up?','Flow-matching stress test | 25 October 2026')
    r.p('Can an MLP learn a smooth conditional flow as reliably as S4 when its starting noise is a Gaussian process? The original hypothesis concerns a head that learns the broad shape but misses fine variation. This follow-up separates that question from output-dimension constraints, normalization, noisy supervision and numerical integration.')
    r.p('This extends Lyamin’s revision0333803 with independent replay and fresh controlled experiments. Original Tasks 1–3 results and pass flags remain unchanged.')
    r.p(f'**Study ledger:** {cap["independent_training_fits"]}/36 capacity fits, {norm["completed_fits"]}/84 normalization fits and {len(label)}/24 label-noise fits. Repeated budget checkpoints are not independent experiments. '+('All planned training fits are complete.' if complete else '**INTERIM: experiments are still running. Do not treat this export as final.**'))
    r.p('**The old numbers reproduce.** All360 kernel-scan checkpoints reproduce their full8192-pair velocity metrics. Eighteen focused cells also reproduce complete forecast ensembles. The main revisions concern explanation, comparison design and scope.')
    r.p('**The old F64 MLP has a structural constraint.** Its width56 final decoder cannot change variance in at least eight directions. With white noise, wider controls close79.4% and88.4% of the residual-error gap to S4 and largely remove excess roughness. This concerns the long-horizon extension, not the original F16 model or a universal need for temporal heads.')
    r.p('**A Jacobian ranking depends on its metric.** For the GP source, S4 remains closer under an unweighted residual-block norm using a reference conditioned on exactly the normalized history. Weighting perturbations by their conditional covariance reverses the MLP/S4 ranking. Neither local derivative metric is a proper forecast score.')
    r.p(f'**Centering dominates the ETTh1 head contrast.** GP-source MLP/S4 CRPS changes from {fmt(nm("global","gp","mlp"))}/{fmt(nm("global","gp","s4"))} under global scaling to {fmt(nm("center","gp","mlp"))}/{fmt(nm("center","gp","s4"))} with center-only normalization. Similar CRPS can hide very different interval coverage. This is one previously examined test period.')
    r.p('**Conditional-mean supervision helps, but does not solve the task.** Residual discrepancy improves in all12 paired comparisons. Forecast CRPS improves in6/6 MLP pairs and4/6 S4 pairs; neither head reaches the exact predictive reference. The intervention uses synthetic population knowledge and does not isolate optimizer noise alone.')
    r.p('**Practical conclusion:** remove avoidable decoder constraints, specify the complete normalization/source pipeline, and compare forecast scores and uncertainty. No result establishes a universal winner or a general failure of Sundial, TSFlow or MLP heads.')

    r.page('Design and evidence ledger')
    r.table(['Package','Design','New training runs'],[
        ['Reproduction','360 old checkpoints;18 complete forecast cells','0'],
        ['Information-matched oracle','25 original checkpoints; analytic/numerical verification','0'],
        ['Capacity and budget','RBF/OU; F64; white/GP; paired fresh draws;6k/12k','36'],
        ['Normalization and GP level','ETTh1 OT; five pipelines; two heads; three seeds;2k/6k','84'],
        ['Conditional-label noise','F16; white/GP; MLP/S4; ordinary/oracle labels; three fresh draws','24']])
    r.p('Capacity and label studies each use three independent synthetic draws paired with optimization seeds. Normalization uses one chronological real-data split and three optimization seeds, not three datasets. All new training uses MPS; old CUDA results are replayed on CPU and are not pooled with new estimates. Concurrent GPU processes make logged wall times unsuitable as architecture speed benchmarks.')
    r.p('Configurations and numerical source hashes were frozen locally before scored training. This is documented execution order, not external preregistration. Earlier results informed these follow-up questions. The ETTh1 test period was already examined in previous work; no untouched-test claim is made.')
    r.p('Each budget selects the lowest validation objective in its own training prefix. Both budget checkpoints are evaluated only after the full training trajectory completes. There is no test-based checkpoint selection. Software smokes are separate and excluded from the fit count.')
    r.p('The original three metrics remain available: raw-coordinate PC1 velocity MSE, mean residual-PC velocity MSE and generated/target adjacent-jump roughness. Forecast comparisons additionally report fair CRPS, energy score and interval coverage. Lower residual MSE need not imply lower forecast CRPS; narrow intervals need not be calibrated.')
    r.p('**Assignment scope:** the F64 capacity study is outside the original rank-two geometry gate. ETTh1 with history normalization also fails that gate: validation GP-velocity top-two energy is0.821–0.851 across five independent source draws, below0.90. These are supplementary robustness studies, not additional completed columns of the original stress test. The F16 label-noise experiment returns to the original geometry. See the independent scope_geometry.json audit.')

    r.page('Replay and the decoder constraint')
    r.p('Kernels, Gaussian windows, normalization, source draws, PCA scoring and forecast metrics are independently reconstructed. Architecture is reused to load checkpoints. The audit checks hashes, numerical agreement and selection metadata.')
    r.table(['Check','Coverage','Outcome'],[['Full velocity replay','360 ×8192 test pairs','Passed'],['Full forecast replay','18 ×256 contexts ×32 members','Passed'],['Checkpoint integrity','360 kernel checkpoints and24 ETTh1 checkpoint/sample pairs','Passed'],['Frozen original source hashes','Referenced numerical implementation files','Matched']])
    r.p('Forecast replay covers RBF/F64 with white/GP sources and MLP/S4 across three draws, plus OU/F64 GP with MLP/S4 across three draws. The original forecast arrays were omitted from GitHub; they are reconstructed from checkpoints and fixed seeds.')
    bridge={x['geometry']:x for x in read('output/followup-capacity/vasya_geometry_bridge.json')['cells']}
    r.p(f'In the historical RBF/white scan, the ratio of mean residual MSE (S4/MLP) is {fmt(bridge["F16_l2"]["ratio_of_means_s4_over_mlp"],3)} at F16/lengthscale2 and {fmt(bridge["F64_l8"]["ratio_of_means_s4_over_mlp"],3)} at F64/lengthscale8. Both have F/lengthscale=8; only F64 exceeds the MLP width. Horizon also changes the problem, so fresh width controls are needed.')
    r.p('The MLP velocity has the form v(x,t,h)=Wz(x,t,h)+b. For any projection P perpendicular to the column span of W, Pv=Pb. Therefore after unit-time integration, Px(1)=Px(0)+Pb: the conditional variance in that subspace is unchanged. The claim is independent of the optimizer, training duration and numerical Jacobian threshold.')
    r.image('output/followup-reproduction/decoder_null_variance.png','Old F64/width56 MLP checkpoints. Forecast/source variance is conditional within ensembles; target marginal variance is a conservative upper bound on average conditional target variance. Small GP values are not zero.')

    r.page('An oracle for normalized history')
    r.p('The original Gaussian oracle knows the raw history and its location/scale. The network sees history after subtracting its mean and dividing by its standard deviation. Integrating out missing location and scale gives a Student-t predictive law with L−1 degrees of freedom when there is no normalization floor. The implemented reference also handles the floor; its truncation correction is negligible in these draws. The conditional future is generally not Gaussian.')
    r.p('The new reference uses the full normalized history. It removes the privilege of knowing raw location and scale, but the network’s learned encoder can still discard information. A remaining discrepancy does not isolate the head from the encoder.')
    r.p('The Gaussian-source conditional flow is consequently a mixture field. Its derivative includes changes in posterior mixture weights; averaging Gaussian component Jacobians would be wrong. The implementation checks quadrature convergence, finite differences, conditional moments, an independent centered-history derivation and direct numerical integration.')
    os=oracle['summary'];rows=[]
    for h,name in [('mlp','MLP'),('s4','S4'),('whitened_mlp','Whitened MLP')]:
        v=os[f'gp|{h}']['matched'];rows.append([name,fmt(v['residual']['mean_relative_error'],3),fmt(v['residual']['covariance_weighted_relative_error'],3),fmt(v['fresh_velocity']['oracle_discrepancy']['residual'],5)])
    r.table(['GP-source head','Unweighted local J error','Covariance-weighted local J error','Residual velocity gap to oracle'],rows)
    r.p('All three columns are errors, so lower is better. The derivative comparison uses256 histories and five times on the original checkpoints. Velocity discrepancy uses2048 fresh histories. Weighting by conditional input covariance was added after the initial comparison and is explicitly exploratory. It describes local sensitivity weighted by covariance, not forecast accuracy or a proof of optimization causality.')
    r.image('output/followup-oracle/jacobian_metric_sensitivity.png','The MLP/S4 ordering changes with the derivative metric. Equal treatment of tiny-variance directions is consequential.')
    r.p('Finite covariance-scaled probes cover all25 checkpoints and reproduce this ordering within the15-model GP group, in residual input/output PCs: relative errors are approximately0.368 for whitened MLP, 0.597 for raw MLP and0.648 for S4. Shared directions and two fixed scales give actual velocity changes, not just derivatives. The supplementary report retains absolute errors and denominators. This post-hoc check does not establish an optimization mechanism or forecast superiority.')

    for kernel,title in [('rbf','Removing the output bottleneck'),('ou','Does the long-window comparison generalize to OU?')]:
        r.page(title,'Fresh capacity/source/budget intervention')
        heads=['mlp_original','mlp_rank_clear','mlp_wide','s4'] if kernel=='rbf' else ['mlp_rank_clear','s4']
        names={'mlp_original':'MLP56 ×3','mlp_rank_clear':'MLP65 ×2','mlp_wide':'MLP72 ×3','s4':'S4 59 ×3'}
        rows=[]
        for source in ['white','gp']:
            for h in heads:rows.append([source,names[h],fmt(cm(kernel,source,h,12000,'residual_mse')),fmt(cm(kernel,source,h,12000,'fair_crps')),fmt(cm(kernel,source,h,12000,'roughness_ratio'),3)])
        r.table(['Source','Head','Residual MSE','Fair CRPS','Roughness ratio'],rows)
        if kernel=='rbf':
            r.p('F64/L128, lengthscale8, three fresh paired draws. Parameter counts: original MLP73960; width65/depth2 MLP73953; S4 74038. Width72/depth3 is a larger same-depth control (114328 parameters). Width65 removes the fixed decoder nullspace; LayerNorm can still numerically attenuate a derivative direction.')
            original=cm('rbf','white','mlp_original',12000,'residual_mse');s4=cm('rbf','white','s4',12000,'residual_mse')
            closed=100*(original-cm('rbf','white','mlp_rank_clear',12000,'residual_mse'))/(original-s4)
            wide_closed=100*(original-cm('rbf','white','mlp_wide',12000,'residual_mse'))/(original-s4)
            r.p(f'The near-equal-budget MLP closes {closed:.1f}% of the mean white-source residual-error gap to S4; the wider same-depth MLP closes {wide_closed:.1f}%. Roughness approaches the target, while a smaller velocity-error advantage for S4 remains. Forecast CRPS gives a different ordering.')
            r.p('A fixed target-PCA basis gives the same white-source ordering. Width and depth change together in the equal-budget control; interpret it alongside the decoder invariant, wider same-depth model and GP control.')
        else:
            r.p('OU has substantially more small-scale target variation than the original smooth RBF geometry. The same architecture comparison tests whether an apparent head effect is specific to a nearly low-rank target. No cell outside the original top-two-energy gate is retroactively relabeled as the original assignment’s main task.')
            gain_white=100*(1-cm('ou','white','s4',12000,'residual_mse')/cm('ou','white','mlp_rank_clear',12000,'residual_mse'))
            gain_gp=100*(1-cm('ou','gp','s4',12000,'residual_mse')/cm('ou','gp','mlp_rank_clear',12000,'residual_mse'))
            r.p(f'S4 reduces mean residual velocity error by {gain_white:.2f}% with white noise and {gain_gp:.2f}% with a GP source. Mean CRPS differences are much smaller, and a white-source draw reverses the forecast ordering. The remaining velocity advantage is evidence about these models and budgets, not a uniform forecast advantage.')
            r.p('The two budget selections share the same training path. Longer training is a finite-budget sensitivity check, not a convergence proof. Per-draw values and per-budget contrasts are retained in the CSV/JSON tables, even when head orderings differ across metrics.')
        if kernel=='rbf':
            r.page('Capacity across sources and budgets','RBF | three independent data draws')
            r.p('The same four heads are compared at6000 and12000 updates. Each point is a data draw paired across heads; pale and teal points from that draw share a training trajectory. The small horizontal bars show means.')
        r.image(f'output/followup-capacity/{kernel}_capacity.png','Each point is an independent draw; budget checkpoints from the same draw are correlated. Raw velocity error and forecasting quality are shown separately.')
        if kernel=='rbf':
            r.p('With white noise, widening consistently reduces the original MLP’s velocity error. The forecast-score differences are much smaller and draw-dependent. With a GP source, the four heads are close on both metrics: the decoder restriction matters far less when the source already has little variance in the restricted directions.')

    r.page('What the decoder restriction looks like in generated paths')
    r.p('This example uses the first held-out context of the first fresh draw and the first eight ensemble members. Indices were fixed without selecting an unusually good or bad forecast. All panels share axes. The orange curve is one realized future, not a unique correct trajectory for every generated sample.')
    r.image('output/followup-capacity/fixed_context_paths.png','RBF, F64, data seed607041, validation-selected12000-update checkpoints. The white-source original MLP retains visibly irregular variation; both wider decoders and S4 produce smoother paths. Quantitative conclusions use all contexts and all three draws.')
    r.p('The GP source supplies little variation in the original decoder’s fixed nullspace, so the same representational restriction need not create visible jitter there. Smoothness by itself does not establish conditional forecast quality: broad shape, uncertainty and CRPS remain separate checks.')
    noflow=read('results/followup-capacity-baselines/summary.json')
    r.table(['Untrained source / geometry','Fair CRPS','Roughness ratio','Full rule passes'],[[x['kernel'].upper()+' / '+x['source'],fmt(x['forecast_means']['fair_crps']),fmt(x['metrics']['roughness_ratio'],3),str(x['literal_successes'])+'/3'] for x in noflow['cells']])
    r.p('The zero-flow baseline returns the source itself. It already passes the roughness condition in every GP group, but passes the complete assignment rule in none. White and GP have similar pointwise marginal variances and therefore similar marginal CRPS; their temporal dependence and energy scores differ. These are frozen-test diagnostics with no model selection.')

    r.page('Do forecast comparisons survive finer integration?')
    solver=read('results/followup-solver/summary.json') if (ROOT/'results/followup-solver/summary.json').exists() else {'records':[],'paired_head_contrasts':[],'models':0}
    solver_rows=[]
    for contrast in solver['paired_head_contrasts']:
        solver_rows.append([contrast['kernel'].upper()+' / '+contrast['source'],names[contrast['head']],fmt(contrast['mean_head_minus_s4_64'],5),fmt(contrast['mean_head_minus_s4_128'],5)])
    r.table(['Geometry / source','Head compared with S4','CRPS difference: 64 steps','CRPS difference: 128 steps'],solver_rows)
    r.p('Negative differences favor the named MLP; positive differences favor S4. Means use all three paired draws. Every capacity model is checked at the final budget, using the same256 contexts,32 source samples per context and unchanged validation-selected checkpoint. The original64-step score remains primary.')
    if solver['records']:
        worst=max(abs(x['scores']['fair_crps']['delta128_minus64']) for x in solver['records'])
        flips=sum(x['mean_sign_flipped'] for x in solver['paired_head_contrasts'])
        r.p(f'The check covers {solver["models"]}/36 models. The largest absolute change in a model’s mean CRPS is {fmt(worst,5)}. The sign of the mean head contrast changes in {flips}/{len(solver["paired_head_contrasts"])} comparisons. Individual draw contrasts and per-context changes are retained in the solver summary.')
    r.p('This diagnostic was added after seeing small head differences in the first two draws. It tests numerical sensitivity, not a new model or a tuned solver. Independent source/history reconstruction, checkpoint hashes and a64-step replay precede every finer integration. Two step counts do not prove convergence to the exact ODE solution.')
    label_solver_path=ROOT/'output/followup-label-solver/summary.json'
    if label_solver_path.exists():
        ls=read(label_solver_path.relative_to(ROOT));lc={(x['source'],x['head'],x['label']):x for x in ls['cells']}
        r.table(['Label study: CRPS','Ordinary: 64','Mean label: 64','Ordinary: 128','Mean label: 128'],[[src+' / '+head.upper()]+[fmt(lc[(src,head,lab)][f'means{steps}']['fair_crps'],5) for steps,lab in [(64,'stochastic'),(64,'oracle'),(128,'stochastic'),(128,'oracle')]] for src in ['white','gp'] for head in ['mlp','s4']])
        r.p(f'The label study also checks every model and all256 ×32 paths. The largest individual mean-CRPS change is {fmt(ls["max_abs_individual_crps_change"],5)}; {len(ls["crps_label_contrast_sign_changes"])} paired label contrasts reverse sign. Full MPS64 replay matches saved arrays; a separate CPU128 replay checks eight contexts of every model. These numerical checks do not remove sampling or finite-budget uncertainty.')

    r.page('Separating centering, scale and objective','ETTh1 OT | complete normalization pipelines')
    modes=['global','center','scale','history','history_rawloss'];names=['Global','Center only','Scale only','Center + scale','History + raw-unit loss']
    r.table(['Pipeline','White MLP','White S4','GP MLP','GP S4','Ridge baseline'],[[name]+[fmt(nm(mode,p,h)) for p,h in [('white','mlp'),('white','s4'),('gp','mlp'),('gp','s4')]]+[fmt(norm['baselines'][mode]['forecast_means']['fair_crps'])] for mode,name in zip(modes,names)])
    r.p('Fair CRPS at6000 updates, averaged across three paired seeds. Every forecast is mapped back to the same train-standardized units. Primary scoring uses all118 chronological test windows with64 members; the original64-context subset is also retained, so these means must not be directly equated to the earlier64-context CUDA table.')
    r.p('Center-only makes level translation equivariant by construction. Scale-only changes scale handling without removing the mean. History combines them. The raw-unit-loss control retains history coordinates and their source, but multiplies velocity errors by history scale before squaring and uses the matching validation objective.')
    r.p('Center-only nearly eliminates the mean GP-source MLP/S4 score difference. Adding scale normalization produces similar CRPS with substantially narrower intervals. Scale-only is less effective here, and reweighting history-coordinate loss back to original units does not restore center-only coverage.')
    r.p('These are interventions on entire pipelines. In original units, a fixed model-coordinate source becomes location + scale × source; centering/scaling therefore also changes its physical interpretation. Results do not uniquely prove that input extrapolation alone caused an earlier head difference.')
    r.image('output/followup-normalization/normalization_b6000.png','Seed points remain visible. Within-period block-bootstrap intervals and individual contrasts are in the machine-readable summary.')

    r.page('Does the constant component of a GP source matter?')
    rows=[]
    for mode in ['global','history']:
        for p in ['white','gp','gp_zero_level','gp_white_level']:rows.append([mode,{'white':'White','gp':'GP','gp_zero_level':'GP: zero mean','gp_white_level':'GP: white-noise level'}[p],fmt(nm(mode,p,'mlp')),fmt(nm(mode,p,'s4'))])
    r.table(['Pipeline','Source','MLP CRPS','S4 CRPS'],rows)
    r.p('GP-zero-level subtracts each source curve’s mean. GP-white-level preserves its centered shape and replaces the mean with an independent Gaussian of variance1/F, matching the level variance of a white source. Both interventions are applied consistently at training, validation and generation.')
    r.p('These controls do not preserve total source variance. Zero-level GP is intentionally singular; a deterministic history-dependent flow may still map it into a curved subset, so missing stochastic degrees of freedom must not be mistaken for a guaranteed forecasting benefit. Interpret score, coverage, spread and budget jointly.')
    r.p('Removing the GP level worsens mean CRPS for both heads in both pipelines. Replacing it with the white-source level variance does not rescue the globally scaled MLP. Thus these controls do not support the GP constant component alone as an explanation for that model’s earlier weakness.')
    r.image('output/followup-normalization/source_level.png','Source-level controls distinguish the effect of broad GP level variation from its centered temporal shape.')

    r.page('Removing conditional target noise','Oracle-assisted mechanism test | not a deployable method')
    rows=[]
    for source in ['white','gp']:
        for head in ['mlp','s4']:
            vals=[]
            for lab in ['stochastic','oracle']:
                rr=lg.get((source,head,lab),[])
                gap=float(np.mean([x['oracle_discrepancy']['residual_mse'] for x in rr])) if rr else float('nan')
                vals.extend([fmt(gap,5),fmt(lm(source,head,lab))])
            rows.append([source,head.upper()]+vals)
    r.table(['Source','Head','Noisy-label residual gap','Noisy-label CRPS','Mean-label residual gap','Mean-label CRPS'],rows)
    r.p('Residual gap is the directly measured mean squared velocity discrepancy from the normalized-history oracle, in source-specific residual PCs. It is not noisy test loss minus an estimated noise floor. Compare the paired label modes within each source.')
    pairs={}
    for row in label:pairs.setdefault((row['data_seed'],row['source'],row['head']),{})[row['label']]=row
    paired=[(key,rr) for key,rr in pairs.items() if len(rr)==2]
    gap_improved=sum(rr['oracle']['oracle_discrepancy']['residual_mse']<rr['stochastic']['oracle_discrepancy']['residual_mse'] for _,rr in paired)
    crps_improved={head:sum(rr['oracle']['forecast_means']['fair_crps']<rr['stochastic']['forecast_means']['fair_crps'] for key,rr in paired if key[2]==head) for head in ['mlp','s4']}
    counts={head:sum(key[2]==head for key,_ in paired) for head in ['mlp','s4']}
    r.p(f'Mean labels reduce residual oracle discrepancy in {gap_improved}/{len(paired)} pairs. CRPS improves in {crps_improved["mlp"]}/{counts["mlp"]} MLP pairs and {crps_improved["s4"]}/{counts["s4"]} S4 pairs. Both heads still trail the predictive reference. This supports a finite-budget sensitivity to supervision, not a complete explanation of the remaining error.')
    predictive=read('results/followup-label-noise/predictive_baselines.json')['rows']
    matched_crps=float(np.mean([x['scores']['matched']['mean']['fair_crps'] for x in predictive]))
    privileged_crps=float(np.mean([x['scores']['privileged']['mean']['fair_crps'] for x in predictive]))
    r.p(f'Direct samples from the normalized-history predictive reference give mean fair CRPS {fmt(matched_crps,5)}; the raw-history Gaussian reference gives {fmt(privileged_crps,5)}. The latter uses extra information. These baselines use64 members on the same contexts and no ODE solver; network scores use32 members. Fair-score correction makes them comparable in expectation, with different Monte Carlo variance.')
    r.p('Within each source and data draw, paired runs see the same complete6000 ×256 stream of inputs, histories and times. Only the regression target changes: ordinary CFM velocity y−epsilon versus its information-matched conditional expectation. This removes conditional label noise while retaining the population optimum. The source/target distribution is known analytically only because this is a synthetic diagnostic.')
    r.p('Both label modes use the same oracle-discrepancy validation criterion. This is intentional for the mechanism test and differs from ordinary noisy-label checkpoint selection. Test references are precomputed with frozen formulas and never used for selection. Each of three fresh data draws has its own optimization seed.')
    r.p('If denoising improves a head, it supports sensitivity to conditional-mean supervision at this budget; it does not establish an intrinsic conditioning defect. The oracle also replaces each finite training pool’s realized future with population knowledge, so this intervention does not isolate optimizer noise from finite-data target noise. If a gap remains, finite capacity, architecture and optimization remain possible contributors. This is not an available real-data forecasting algorithm.')

    r.page('Calibration, uncertainty and scope')
    r.table(['Pipeline','White MLP','White S4','GP MLP','GP S4','Ridge baseline'],[[name]+[fmt(nm(mode,p,h,'coverage_90'),3) for p,h in [('white','mlp'),('white','s4'),('gp','mlp'),('gp','s4')]]+[fmt(norm['baselines'][mode]['forecast_means']['coverage_90'],3)] for mode,name in zip(modes,names)])
    r.p('These are coverages of nominal90% empirical intervals. Finite ensembles introduce sampling bias: even draws from the correct predictive law need not give90% coverage. Read coverage, interval width and CRPS together. Ridge is refitted under each normalization, with validation-selected penalty and resampled training residuals for uncertainty. It is a direct forecaster, using neither a flow nor a GP source.')
    r.p('Three synthetic draws do not support a broad benchmark claim. ETTh1 supplies one already-examined series and test period. Its descriptive intervals resample paired training seeds and circular blocks of4 or8 forecast contexts; they do not estimate uncertainty across independent datasets. Multiple exploratory contrasts are reported without confirmatory significance claims.')
    boundary_cap=sum(x['best_at_budget'] for x in cap['cells'] if x['budget']==12000)
    boundary_label=sum(x['best_step']==6000 for x in label)
    r.p(f'Checkpoints selected at the final available update: capacity {boundary_cap}/{cap["independent_training_fits"]}; normalization {norm["best_at_budget"]["6000"]}/{norm["completed_fits"]}; label intervention {boundary_label}/{len(label)}. Fixed budgets do not imply convergence. Euler128 covers full capacity and label ensembles, and limited normalization subsets. It tests numerical sensitivity without proving exact integration. CPU/MPS replay tolerance is recorded.')
    r.p('Tiny covariance eigenvalues and whitening floors strongly affect normalized errors; raw, fixed-target-basis and noise-floor diagnostics must remain visible.')
    r.p('A source matching a raw GP kernel does not match the normalized future distribution. The information-matched future is a scale mixture, not generally Gaussian. The new same-information oracle corrects that reference distinction; it does not convert every source into a matched target distribution.')
    r.p('The original literal20% rule and earlier all-fail outcome remain unchanged. All24 new label-study models pass the roughness condition; none passes the full velocity-ratio rule. That equality-style raw-error rule mixes reducible error with different conditional noise levels. This follow-up adds evidence without rewriting the assignment or claiming a general architecture winner.')

    strata_path=ROOT/'results/followup-calibration-strata/summary.json'
    if strata_path.exists():
        strata=read(strata_path.relative_to(ROOT))
        r.page('Calibration by history variability','Exploratory check | training-only quartile boundaries')
        counts=strata['test_bin_counts']
        r.p('Test windows are grouped by the standard deviation of the observed history. Boundaries are the25th and75th percentiles of training histories, fixed before inspecting grouped forecast scores. The groups contain '+', '.join(f'{name}: {n} window'+('s' if n!=1 else '') for name,n in counts.items())+'. All84 fits and both budgets are included in the saved tables.')
        r.p('The high-variability group contains only one test window. Its value is a single case, not evidence about a general high-variability regime. The low group dominates this period. Coverage differences there accompany large changes in interval width without comparably large changes in point-prediction MSE.')
        rows=[]
        for prior in ['white','gp']:
            for head in ['mlp','s4']:
                before=strata['table'][f'center|{prior}|{head}|6000']['bins']['low']['mean']
                after=strata['table'][f'history_rawloss|{prior}|{head}|6000']['bins']['low']['mean']
                rows.append([prior+' / '+head.upper()]+[fmt(before[k],n)+' → '+fmt(after[k],n) for k,n in [('coverage_90',3),('width_90',3),('fair_crps',4),('mean_forecast_mse',4)]])
        r.table(['Low group','Coverage (90%)','Interval width','Fair CRPS','Point MSE'],rows)
        r.p('Table: center-only → history with raw-unit loss, averaged over three seeds. Group-weighted scores recover the overall means. These are descriptive partitions of one period, not independent replications.')
        r.image('output/followup-calibration-strata/coverage_90.png','All five pipelines. Training quartiles define low, middle and high history variability. Companion figures report interval width and fair CRPS; ridge baselines and every seed are also retained.')

    r.page('What changes in the earlier interpretation?')
    r.p('The previous score tables reproduce. The follow-up changes how several results should be explained, rather than invalidating the earlier experiments. The comparison below refers to README.md sections10–11 and docs/kernel_scan_findings.md at revision0333803.')
    r.table(['Earlier interpretation','New evidence','Defensible reading'],[
        ['S4 is closer to the GP residual Jacobian; the small directions are largely unlearned.',
         'A normalized-history oracle retains the unweighted ordering, but covariance-weighted derivatives and finite changes favor MLP. The raw-history Gaussian oracle has extra information.',
         'Report the reference information and perturbation geometry. A local derivative error does not identify an optimization mechanism or forecast ranking.'],
        ['History normalization removes the ETTh1 source/head interaction, attributed to level extrapolation.',
         'The center/scale factorial and raw-unit-loss control compare complete pipelines, including the source in original units. Coverage is evaluated alongside CRPS.',
         'Normalization is consequential. Isolating one cause requires more than the global-versus-history contrast; low CRPS does not guarantee calibrated intervals.'],
        ['The strongest white-source S4 advantage appears at F64.',
         'The original MLP cannot change variance in at least eight output directions. Fresh wider controls largely close the residual-error gap and restore near-target roughness.',
         'A restricted decoder is a major confound. A smaller S4 velocity advantage remains; it does not translate into uniform forecast superiority.'],
        ['The scan gives no material GP-source forecast reason to replace the MLP head.',
         'Fresh capacity/source comparisons, longer budgets, finer integration and an oracle-label intervention test this interpretation from different angles.',
         'Keep conclusions specific to the source, geometry, information set and budget. These small-model studies do not establish a general architecture winner.']])
    r.p('The original assignment’s literal success flags are preserved. The F64 and real-data studies are extensions outside its main geometry gate. The original F16 question is addressed separately by the information-matched oracle and conditional-label experiment.')

    r.page('Reproduction and handoff')
    r.p('No new commit or push is created for this work. The base revision is0333803. New implementations, configurations, protocols and tests are separate from original experiment files. Checkpoints, generated arrays and run records remain in results/followup-*; figures and aggregate tables are in output/followup-*.')
    r.table(['Purpose','Entry point'],[['Old 360 + 18 replay','scripts/audit_followup_reproduction.py'],['Information-matched oracle','scripts/study_normalized_history_oracle.py'],['Capacity experiments','python -m fm_stress.followup_capacity'],['Normalization experiments','python -m fm_stress.followup_normalization'],['Label-noise preparation/training','python -m fm_stress.followup_label_noise'],['Independent checks','scripts/check_followup_capacity.py; check_followup_normalization.py; check_normalized_history_oracle.py'],['Scope and execution order','docs/followup_research_plan.md']])
    r.p('Use the repository virtual environment and MPS access for scored training. Reproduction commands, full configurations and source hashes accompany each study. Missing or incompatible artifacts cause the runners to fail rather than silently mixing configurations. Independent checks distinguish reused architecture from independently rebuilt data and metrics.')
    r.table(['Final verification coverage','Status'],[
        ['Capacity: 72 checkpoints; all 589,824 test velocity predictions; 2,754 checks','Passed'],
        ['Normalization: 84 fits,168 checkpoint artifacts;56 aggregate cells; frozen strata','Passed'],
        ['Label intervention: 24 fits; six complete prepared streams; source/data hashes','Passed'],
        ['Euler64/128: 36 capacity + 24 label models; independent all-array CRPS checks','Passed'],
        ['Project test suite: 211 tests','Passed']])
    r.p('The most defensible recommendation is conditional: first remove avoidable decoder constraints and normalize consistently, then compare proper forecasting scores and calibration across sources. A temporal head may be useful in a particular regime; neither its Jacobian norm nor a comparison against a structurally restricted MLP establishes that it is universally required.')
    return r,complete


def render(report,pdf,html_path):
    register_fonts();width=A4[0]-88
    styles={
      'title':ParagraphStyle('FTitle',fontName='RS-Bold',fontSize=23,leading=28,textColor=colors.HexColor('#173d49'),spaceAfter=15),
      'kicker':ParagraphStyle('FKicker',fontName='RS-Bold',fontSize=8.5,leading=12,textColor=colors.HexColor('#147d81'),spaceAfter=9),
      'body':ParagraphStyle('FBody',fontName='RS',fontSize=10,leading=14.1,textColor=colors.HexColor('#243c46'),spaceAfter=10),
      'small':ParagraphStyle('FSmall',fontName='RS',fontSize=8,leading=11,textColor=colors.HexColor('#57717b'),spaceAfter=10),
      'cell':ParagraphStyle('FCell',fontName='RS',fontSize=8,leading=10.5,textColor=colors.HexColor('#243c46')),
      'head':ParagraphStyle('FHead',fontName='RS-Bold',fontSize=8,leading=10.5,textColor=colors.white)}
    story=[];sections=[]
    for i,page in enumerate(report.pages):
        if i:story.append(PageBreak())
        story.extend([Paragraph(html.escape(page['kicker'].upper()),styles['kicker']),Paragraph(html.escape(page['title']),styles['title'])])
        hs=[f'<div class="kicker">{html.escape(page["kicker"])}</div><h1>{html.escape(page["title"])}</h1>']
        for kind,value in page['blocks']:
            if kind=='p':
                story.append(Paragraph(markup(value),styles['body']));hs.append('<p>'+re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',html.escape(value))+'</p>')
            elif kind=='table':
                headers,rows=value
                data=[[Paragraph(html.escape(str(x)),styles['head']) for x in headers]]+[[Paragraph(html.escape(str(x)),styles['cell']) for x in row] for row in rows]
                t=Table(data,colWidths=[width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173d49')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf4f4')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
                story.extend([t,Spacer(1,12)])
                hs.append('<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>')
            else:
                path,caption=value;src=ROOT/path
                if not src.exists():raise FileNotFoundError(src)
                iw,ih=PILImage.open(src).size;maxheight=210 if src.name=='decoder_null_variance.png' else 320;factor=min(width/iw,maxheight/ih)
                story.append(KeepTogether([Image(str(src),width=iw*factor,height=ih*factor,hAlign='LEFT'),Paragraph(html.escape(caption),styles['small'])]))
                import os
                hs.append(f'<figure><img src="{html.escape(os.path.relpath(src,html_path.parent))}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        sections.append('<section>'+''.join(hs)+'</section>')
    pdf.parent.mkdir(parents=True,exist_ok=True);html_path.parent.mkdir(parents=True,exist_ok=True)
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#cedde1'));c.line(44,36,A4[0]-44,36);c.setFont('RS',8);c.setFillColor(colors.HexColor('#57717b'))
        c.drawString(44,23,'Flow-matching stress test | Follow-up to 0333803 | 25 Oct 2026');c.drawRightString(A4[0]-44,23,str(doc.page))
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=38,leftMargin=38,topMargin=35,bottomMargin=48,title='Flow-matching stress test: controlled follow-up',author='FM stress project team').build(story,onFirstPage=footer,onLaterPages=footer)
    css='body{margin:0;background:#e8eef0;color:#243c46;font:17px/1.6 system-ui}main{max-width:1050px;margin:auto}section{background:white;margin:24px 0;padding:48px 58px;box-shadow:0 2px 12px #173d4912}h1{font-size:31px;line-height:1.2;color:#173d49}.kicker{color:#147d81;font-size:13px;font-weight:700;text-transform:uppercase}table{width:100%;border-collapse:collapse;font-size:14px;margin:24px 0}td,th{padding:12px;text-align:left;vertical-align:top}th{background:#173d49;color:white}tr:nth-child(even){background:#edf4f4}img{max-width:100%}figure{margin:24px 0}figcaption{font-size:13px;color:#57717b}@media print{section{break-before:page;margin:0;box-shadow:none;padding:0}body{background:white}}'
    html_path.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>FM stress: controlled follow-up</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    before=consumed_artifacts()
    report,complete=content(a.allow_partial)
    dest='followup-research' if complete else 'followup-research-interim'
    pdf=ROOT/'output/pdf'/f'{dest}.pdf';html_path=ROOT/'output/followup-research'/f'{dest}.html'
    render(report,pdf,html_path)
    after=consumed_artifacts()
    if before!=after:raise RuntimeError('Input artifacts changed during report construction; rebuild from a stable snapshot')
    (html_path.parent/'report_manifest.json').write_text(json.dumps({'complete':complete,'planned_sections':len(report.pages),'pdf_sha256':sha(pdf),'html_sha256':sha(html_path),'builder_sha256':sha(Path(__file__)),'consumed_artifacts_sha256':after},indent=2)+'\n')
    print(pdf);print(html_path)
