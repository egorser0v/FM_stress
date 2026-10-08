"""Export the complete independently audited sparse/whitening extension.

No partial or synthetic placeholder report is permitted. Export requires all264
main new fits, the24-fit dictionary supplement, all96 frozen controls, independent
run/analysis/rank audits, actual-MPS model parity, and independent aggregate checks.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import subprocess
import sys
from types import SimpleNamespace
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_report import Report, register_fonts, styles, WIDTH, NAVY, MUTED, PALE, LINE

DATASETS = {'gp_multi':'Correlated GP: three channels', 'etth1_uni':'ETTh1: oil temperature',
            'etth1_multi':'ETTh1: seven channels', 'exchange_multi':'Exchange: eight channels'}
SHORT_DATASETS = {'gp_multi':'GP (3 channels)', 'etth1_uni':'ETTh1 (OT only)',
                  'etth1_multi':'ETTh1 (7 channels)', 'exchange_multi':'Exchange (8 channels)'}
HEADS = {'mlp':'MLP', 'sparse_mlp':'Fixed-mask MLP', 's4':'S4', 'unet':'U-Net',
         'mlp_whitened':'Whitened MLP', 'mlp_balanced':'Balanced-loss MLP',
         'history_dense_ae':'Dense history AE', 'history_topk_ae':'TopK history SAE',
         'history_conv_dense':'Dense conv. history', 'history_conv_sparse':'Sparse conv. history',
         'dictionary_dense':'Dense velocity dictionary', 'dictionary_topk':'TopK velocity dictionary',
         'unet_l0':'L0-gated U-Net', 'moe_dense':'Dense mixture', 'moe_topk':'Top-2 mixture'}
ORDER = ['mlp','mlp_whitened','mlp_balanced','sparse_mlp','s4','unet','unet_l0',
         'history_dense_ae','history_topk_ae','history_conv_dense','history_conv_sparse',
         'dictionary_dense','dictionary_topk','moe_dense','moe_topk']
PAIRS = [('history_topk_ae','history_dense_ae'),('history_conv_sparse','history_conv_dense'),
         ('dictionary_topk','dictionary_dense'),('unet_l0','unet'),('moe_topk','moe_dense')]
BASELINES = [('persistence','Persistence'),('seasonal_naive','Seasonal naive'),
             ('ridge_residual_bootstrap','Ridge + residuals'),
             ('multitask_elastic_net_residual_bootstrap','Elastic Net + residuals')]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


class SparseReport(Report):
    def __init__(self):
        self.s = styles()
        self.story = []
        self.args = SimpleNamespace(analysis=ROOT / 'output/sparse-analysis')
        self.summary = read('output/sparse-analysis/summary.json')
        self.cfg = read('configs/sparse_extension.json')
        self.audit = read('results/sparse-extension/independent_verification.json')
        self.manifest = read('results/sparse-extension/manifest.json')
        self.parity = read('results/diagnostics/sparse_model_parity.json')
        self.supplement = read('output/dictionary-supplement/summary.json')
        for analysis,verification in (
            ('output/sparse-analysis/summary.json','results/diagnostics/sparse_analysis_verification.json'),
            ('output/dictionary-supplement/summary.json','results/diagnostics/dictionary_analysis_verification.json')):
            check = read(verification)
            assert check['status']=='passed' and check['summary_sha256']==digest(analysis)
        supplement_audit = read('results/sparse-fullrank-dictionary/independent_verification.json')
        assert supplement_audit['complete'] and supplement_audit['verified_runs']==24
        assert self.supplement['status']=='complete'
        assert self.supplement['observed_counts']=={'main_64':24,'supplement_192':24}
        assert self.supplement['rank_diagnostics']['status']=='verified'
        assert self.supplement['manifest_fingerprints']['192']==supplement_audit['fingerprint']
        assert self.supplement['manifest_fingerprints']['64']==self.manifest['fingerprint']
        for path,value in self.supplement['rank_diagnostics']['files_sha256'].items():
            assert digest(path)==value
        self.rank_runs = self.supplement['rank_diagnostics']['runs']
        assert self.summary['status'] == 'complete'
        assert self.summary['expected_new_runs'] == self.summary['observed_new_runs'] == 264
        assert self.summary['frozen_control_runs'] == 96
        assert self.audit['complete'] and self.audit['expected_runs'] == self.audit['verified_runs'] == 264
        assert self.audit['fingerprint'] == self.summary['fingerprint'] == self.manifest['fingerprint']
        assert self.cfg == self.summary['config'] == self.manifest['config']
        assert self.parity['status'] == 'passed' and len(self.parity['rows']) == 36
        assert all(digest(path) == value for path,value in self.parity['source_sha256'].items())
        self.rows = self.summary['aggregate']
        assert len(self.rows) == 120 and all(row['n'] == 3 for row in self.rows)
        self.cells = {(r['dataset'],r['head'],r['source']):r for r in self.rows}
        self.records = []
        for directory in ('results/extension','results/sparse-extension'):
            for dataset in self.cfg['datasets']:
                for path in sorted((ROOT / directory / dataset / 'runs').glob('*.json')):
                    record = json.loads(path.read_text())
                    self.records.append({'dataset':dataset, **record})
        assert len(self.records) == 360
        # Independent summary arithmetic: do not trust a chart/summary generator
        # simply because the underlying experiment audit passed.
        comparisons = 0
        for row in self.rows:
            records = [r for r in self.records if (r['dataset'],r['head'],r['source']) ==
                       (row['dataset'],row['head'],row['source'])]
            assert sorted(r['seed'] for r in records) == sorted(self.cfg['seeds'])
            for key, statistic in row['metrics'].items():
                values = [r['velocity_metrics'][key.removeprefix('velocity_')]
                          if key.startswith('velocity_') else r['forecast_means'][key] for r in records]
                assert np.isclose(np.mean(values),statistic['mean'],rtol=1e-10,atol=1e-12), (row,key,'mean')
                assert np.isclose(np.std(values,ddof=1),statistic['sd'],rtol=1e-10,atol=1e-12), (row,key,'sd')
                comparisons += 1
        self.aggregate_checks = comparisons

    def compact_table(self, headers, rows, widths):
        def paragraph(value, kind):
            return Paragraph(escape(str(value)).replace('\n','<br/>'),self.s[kind])
        cells = [[paragraph(v,'headcell') for v in headers]]
        cells += [[paragraph(v,'cell') for v in row] for row in rows]
        table = Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
            ('LINEBELOW',(0,-1),(-1,-1),.5,LINE)]))
        self.story.extend([table,Spacer(1,7)])

    def cell(self,dataset,head,source):
        return self.cells[(dataset,head,source)]

    def value(self,dataset,head,source,metric='fair_crps'):
        return self.cell(dataset,head,source)['metrics'][metric]['mean']

    def fmt(self,dataset,head,source,metric,sd=True):
        value = self.cell(dataset,head,source)['metrics'][metric]
        return f"{value['mean']:.4g}\n± {value['sd']:.2g}" if sd else f"{value['mean']:.4g}"

    def baseline_records(self,dataset):
        return self.summary['datasets'][dataset]['baselines']

    def opening(self):
        self.title('Smooth source, unstructured head | October 2026','Whitening and sparse\nflow-matching models',True)
        self.p('Egor Serov · Kirill Frolov · Daniil Koblov · Vasilii Lyamin','small')
        self.p('Completed follow-up on synthetic multivariate GP, ETTh1 and Exchange. Every neural model is evaluated with both a temporally white source and a smooth GP source. Sparse autoencoders condition the flow-matching model on history; they are not scored as standalone predictors.')
        self.compact_table(['Completed evidence','Scope'],[
            ['288 new MPS fits','264 main fits plus a 24-fit dictionary-rank follow-up'],
            ['96 frozen reference fits','MLP, fixed-mask MLP, S4 and U-Net; same data, seeds and scoring'],
            ['New linear control','MultiTask Elastic Net; validation-selected regularization'],
            ['Verification','384 fits reviewed including controls; independent audits and 36 CPU/MPS parity checks']], [133,WIDTH-133])
        self.p('Does joint PCA whitening help MLP?','h2')
        rows = []
        forecast_wins = residual_wins = 0
        for dataset in self.cfg['datasets']:
            for source in self.cfg['sources']:
                raw = self.value(dataset,'mlp',source)
                white = self.value(dataset,'mlp_whitened',source)
                residual_raw = self.value(dataset,'mlp',source,'velocity_residual_mse')
                residual_white = self.value(dataset,'mlp_whitened',source,'velocity_residual_mse')
                forecast_wins += white < raw
                residual_wins += residual_white < residual_raw
                rows.append([SHORT_DATASETS[dataset],source,f'{raw:.4g}',f'{white:.4g}',
                             f'{100*(white/raw-1):+.1f}%',f'{100*(residual_white/residual_raw-1):+.1f}%'])
        self.compact_table(['Dataset','Source','Raw MLP\nCRPS','Whitened\nCRPS','CRPS\nchange','Residual\nMSE change'],rows,[120,42,80,80,80,WIDTH-402])
        self.p(f'Whitening lowers mean CRPS in {forecast_wins}/8 source/dataset cells and raw residual velocity MSE in {residual_wins}/8. Whitening worsens mean CRPS under both sources on both real multivariate targets. These are distinct outcomes: improving velocity regression does not automatically improve the generated forecast distribution. Negative changes favor whitening.','small')
        baseline_better = 0
        for dataset in self.cfg['datasets']:
            neural = min(self.value(dataset,h,s) for h in ORDER for s in self.cfg['sources'])
            best_baseline = min(r['means']['fair_crps'] for r in self.baseline_records(dataset).values())
            baseline_better += best_baseline < neural
        self.p(f'**Baseline check.** Within the main 15-head comparison, the best simple baseline has lower CRPS than the lowest mean neural cell on {baseline_better}/4 datasets. A favorable sparse-versus-dense ranking is therefore not, by itself, evidence of a useful forecasting improvement.','box')
        ew = self.value('exchange_multi','moe_topk','white')
        eg = self.value('exchange_multi','moe_topk','gp')
        mw = self.value('exchange_multi','mlp','white')
        mg = self.value('exchange_multi','mlp','gp')
        self.p(f'**Most promising sparse neural result.** The Top-2 mixture improves Exchange CRPS to {ew:.3f} (white) and {eg:.3f} (GP), versus {mw:.3f} and {mg:.3f} for MLP. It also improves on the dense mixture there. TopK history reconstruction and TopK velocity dictionaries do not show a comparable advantage over their dense controls at this budget.','small')
        self.p('Winner counts are descriptive, with no multiplicity correction. Paired context intervals condition on the selected periods and trained models; three seeds and two real datasets cannot establish a universal architecture claim.','small')

    def design(self):
        self.title('01 | Experimental design','What each comparison tests')
        self.compact_table(['Variant / control','Change inside conditional flow matching'],[
            ['Whitened MLP / raw MLP','Full joint time-channel PCA transform of state and velocity; inverse-mapped outputs; variance-weighted objective. History stays in original standardized coordinates.'],
            ['Balanced-loss MLP / raw MLP','Only the training/selection loss is weighted by inverse PCA variances; state and head outputs remain in original coordinates.'],
            ['TopK history SAE / dense AE','History reconstruction plus a 64-dimensional conditioning code; TopK retains at most 8 nonnegative features. Dense control uses the same encoder and decoder.'],
            ['Sparse / dense conv. history','Three tied convolutional coding steps; sparse variant adds soft thresholds and code L1 loss. Both reconstruct observed history and condition the same FM architecture.'],
            ['TopK / dense velocity dictionary','Predict a signed coefficient vector and combine learned velocity templates; TopK keeps at most 8 of 64 coefficients. No autoencoder reconstruction objective.'],
            ['L0-gated / ordinary U-Net','Learn hard-concrete hidden-channel gates. Stochastic training, deterministic evaluation; measured gate activity distinguishes attenuation from actual pruning.'],
            ['Top-2 / dense mixture','Four small velocity experts; sparse routing uses two. Both receive the same differentiable importance-balancing penalty.'],
            ['Elastic Net / ridge','Linear multivariate forecasting with shared sparse selection of history lag/channel features; whole-vector training-residual bootstrap supplies samples.']], [140,WIDTH-140])
        self.p('Training and held-out evaluation','h2')
        self.p('Approximately 64k allocated trainable parameters per new family, with the same width inside each dense/sparse pair. The 64-feature history AEs compress GP/ETTh1-seven-channel/Exchange histories; only the OT history has fewer than 64 inputs. Their dense controls share this bottleneck. The fixed-mask MLP reference matches active parameters and therefore allocates more weights. MLP whitening controls reuse the reference MLP width. Gates and thresholds add a small parameter overhead.')
        self.p('Each fit uses AdamW, learning rate 0.001, batch 128, 2,000 updates and three seeds (31, 32, 33). Validation is checked every 250 updates. Selection uses held-out CFM objective, excluding auxiliary penalties; whitened and balanced-loss models use their variance-weighted objective. Rates and regularization strengths are not tuned separately by architecture.')
        self.p('All variants receive the same observed history. Train-only global channel statistics normalize data. Real windows lie fully within chronological train/validation/test splits. Both sources have the same train-estimated cross-channel covariance; only temporal covariance differs. PCA is fitted separately by source to training velocities, with all F×C coordinates retained.')
        self.p('Evaluation uses 64 identical held-out contexts, 32 futures per context and 64 Euler steps. Fair CRPS scores marginal distributions; joint energy scores the complete time-channel vector, divided by sqrt(F×C). Roughness is the mean absolute adjacent temporal increment relative to the target. Lower CRPS/energy is better; roughness near 1 alone does not imply calibrated or accurate forecasts.','small')

    def dataset(self,dataset):
        self.title('02 | Forecast distributions',DATASETS[dataset])
        rows = []
        for head in ORDER:
            rows.append([HEADS[head],*[self.fmt(dataset,head,s,m) for m in
                ('fair_crps','joint_energy_scaled','roughness_ratio') for s in ('white','gp')]])
        width = (WIDTH-125)/6
        self.compact_table(['Head','White\nCRPS ↓','GP\nCRPS ↓','White\nenergy ↓','GP\nenergy ↓','White\nroughness','GP\nroughness'],rows,[125,*([width]*6)])
        self.p('Mean on the first line; ± sample SD across three optimization seeds on the second. All scores use train-standardized units. Compare methods within this dataset. White denotes temporally white noise with cross-channel covariance retained.','small')
        self.p('Simple forecasting controls','h2')
        baselines = self.baseline_records(dataset)
        br = []
        for key,label in BASELINES:
            if key not in baselines:
                continue
            metrics = baselines[key]['means']
            br.append([label,f"{metrics['fair_crps']:.4g}",f"{metrics['joint_energy_scaled']:.4g}",
                       f"{metrics['mean_forecast_mse']:.4g}",f"{100*metrics['coverage_90']:.1f}%"])
        self.compact_table(['Baseline','CRPS ↓','Joint energy ↓','Mean MSE ↓','90% coverage'],br,[175,82,82,82,WIDTH-421])
        self.p('Baselines are fitted once, so no optimization-seed SD is attached. Residual samples are drawn from in-sample training errors and can understate uncertainty. Deterministic baselines have degenerate intervals. Complete results also include channel-sum CRPS, interval width, point MSE and coverage for every neural cell.','small')

    def diagnostics(self):
        self.title('03 | Representation diagnostics','Did the models actually become sparse?')
        rows = []
        specs = [('history_dense_ae','code_active_fraction'),('history_topk_ae','code_active_fraction'),
                 ('history_conv_dense','code_active_fraction'),('history_conv_sparse','code_active_fraction'),
                 ('dictionary_dense','code_active_fraction'),('dictionary_topk','code_active_fraction'),
                 ('unet_l0','gate_eval_active_fraction'),('unet_l0','gate_expected_active_fraction'),
                 ('moe_dense','routing_fraction'),('moe_topk','routing_fraction')]
        for head,metric in specs:
            label = HEADS[head]
            if metric == 'gate_expected_active_fraction':label = 'L0 expected active gates'
            if metric == 'gate_eval_active_fraction':label = 'L0 nonzero eval gates'
            values = []
            for dataset in self.cfg['datasets']:
                found = [r['validation_activity'][metric] for r in self.records
                         if r['dataset']==dataset and r['head']==head]
                assert len(found)==6,(dataset,head,metric)
                values.append(f'{100*min(found):.1f}-{100*max(found):.1f}%')
            rows.append([label,*values])
        self.compact_table(['Validation activity','GP (3C)','ETTh1 (1C)','ETTh1 (7C)','Exchange (8C)'],rows,[160,*([(WIDTH-160)/4]*4)])
        gate_fractions = [r['validation_activity']['gate_eval_active_fraction'] for r in self.records if r['head']=='unet_l0']
        gate_conclusion = ('All deterministic L0 gates remain nonzero in these runs: this is learned gating/attenuation, not successful channel pruning.' if all(v==1 for v in gate_fractions) else 'Expected gate probability and deterministic nonzero gate count are distinct; report both.')
        self.p('Ranges cover both sources and all three seeds on the full validation split. Code entries count nonzero coordinates; mixture entries count routed experts. '+gate_conclusion,'small')
        self.p('Sparse variants versus their paired dense controls','h2')
        cr = []
        for sparse,dense in PAIRS:
            differences = [self.value(d,sparse,source)-self.value(d,dense,source)
                           for d in self.cfg['datasets'] for source in self.cfg['sources']]
            cr.append([HEADS[sparse],HEADS[dense],f'{sum(v<0 for v in differences)}/8',
                       f'{min(differences):+.4g} to {max(differences):+.4g}'])
        self.compact_table(['Sparse variant','Control','Lower CRPS','Range of CRPS differences'],cr,[145,135,70,WIDTH-350])
        self.p('These counts summarize observed mean differences, not significant wins. Dense/TopK pairs hold width fixed. Sparse convolution changes both shrinkage and its L1 auxiliary penalty; L0 changes both stochastic training and gating regularization. The comparison does not isolate each ingredient separately.','small')
        self.p('Elastic Net feature selection','h2')
        er = []
        for dataset in self.cfg['datasets']:
            e = self.summary['datasets'][dataset]['sparse_baseline']
            er.append([SHORT_DATASETS[dataset],f"{e['active_input_features']}/{e['total_input_features']}",
                       f"{e['selected_alpha']:g}",f"{e['selected_l1_ratio']:g}"])
        self.compact_table(['Dataset','Active history features','Alpha','L1 ratio'],er,[180,145,90,WIDTH-415])
        self.p('The sparse neural implementations still execute dense PyTorch kernels. Activity reduction is not a measured MPS speedup or memory saving. Training times are recorded, but this suite is not a controlled runtime benchmark.','small')

    def pc_metrics(self,dataset):
        self.title('04 | Velocity error in original coordinates',DATASETS[dataset])
        rows = []
        for head in ORDER:
            rows.append([HEADS[head],*[self.fmt(dataset,head,source,metric) for metric in
                ('velocity_pc1_mse','velocity_residual_mse') for source in ('white','gp')]])
        self.compact_table(['Head','White PC1\nMSE ↓','GP PC1\nMSE ↓','White residual\nMSE ↓','GP residual\nMSE ↓'],rows,[145,*([(WIDTH-145)/4]*4)])
        self.p('Mean ± sample SD across three seeds. Residual MSE is the average per-PC velocity error over PCs 3 onward in the joint F×C space. Outputs of the whitened MLP are mapped back before scoring. The original assignment metrics remain raw, unweighted errors; variance-normalized diagnostics are supplementary.','small')
        geometry = read(f'results/sparse-extension/{dataset}/geometry.json')
        rows = []
        for source in ('white','gp'):
            records = [r for r in self.records if r['dataset']==dataset and r['head']=='mlp_whitened' and r['source']==source]
            floor_counts = {r['pca_floored_components'] for r in records}
            assert len(floor_counts)==1
            rows.append([source,f"{geometry[source]['validation_top2_energy']:.4f}",str(next(iter(floor_counts))),
                         self.fmt(dataset,'mlp',''+source,'velocity_residual_normalized_mse',False),
                         self.fmt(dataset,'mlp_whitened',source,'velocity_residual_normalized_mse',False),
                         self.fmt(dataset,'mlp_balanced',source,'velocity_residual_normalized_mse',False)])
        self.p('Train-fitted geometry and normalized residual diagnostic','h2')
        self.compact_table(['Source','Validation\nvelocity E2','Floored\nPCs','Raw MLP\nnormalized','Whitened\nnormalized','Balanced\nnormalized'],rows,[52,92,70,96,96,WIDTH-406])
        self.p('PCA variances are floored at max(10^-6 × largest variance, 10^-8). Full-rank whitening rotates/scales the interpolant and predicts transformed velocity; balanced loss only reweights error in the same PCA basis. Their comparison tests the coordinate parameterization under the shared variance-weighted objective at this fixed optimization budget. Neither removes PCs.','small')
        self.p('Source-specific velocity labels and covariance spectra differ. A smaller velocity MSE under one source is not directly equivalent to a better forecast; read this page together with its CRPS, energy and roughness table. The original rank-two acceptance gate and two-sided 20% rule were not imposed on these broader real-data/multivariate experiments.','small')

    def dictionary_geometry(self):
        self.title('05 | Dictionary-rank follow-up','Separate sparsity from a rank restriction')
        self.p('The main suite uses 64 learned velocity templates (dictionary atoms). That can span the 48-dimensional GP future and 24-dimensional OT future, but it cannot span a 168-dimensional ETTh1 joint future or a 128-dimensional Exchange future. This limitation applies to both dense and TopK coefficients.')
        self.p('**Why rank matters.** With velocity v = D a, the ODE can change only directions in the column span of D. Every component orthogonal to that span remains equal to its initial source value. A low-rank velocity dictionary does not make the complete generated endpoint low rank: unused source variation survives.','box')
        rows = []
        for dataset in ('etth1_multi','exchange_multi'):
            for atoms in (64,192):
                ranks = [r['dictionary_rank'] for r in self.rank_runs if r['dataset']==dataset and r['atoms']==atoms]
                dimensions = [r['orthogonal_dimension'] for r in self.rank_runs if r['dataset']==dataset and r['atoms']==atoms]
                rows.append([SHORT_DATASETS[dataset],str(atoms),f'{min(ranks)}-{max(ranks)}',f'{min(dimensions)}-{max(dimensions)}'])
        self.compact_table(['Dataset','Learned templates','Learned rank range','Unchangeable directions'],rows,[160,100,115,WIDTH-375])
        self.p('Ranges cover both sources, both coefficient heads and all three seeds. Learned rank is checked from each saved dictionary, not inferred from its number of columns. The independent audit also checks the generated endpoint orthogonal component against the exact initial source samples.','small')
        self.p('A declared 24-fit structural control','h2')
        self.p('We added 192-template dense and TopK heads on ETTh1 (7 channels) and Exchange (8 channels), with both sources and three seeds. No main result was overwritten. All data, optimizer settings, training steps, validation selection, forecast contexts and solver settings match the 64-template experiment. TopK still keeps at most eight coefficients per example.')
        rows = []
        for dataset in ('etth1_multi','exchange_multi'):
            for atoms in (64,192):
                cell = next(r for r in self.supplement['aggregate'] if r['dataset']==dataset and r['atoms']==atoms)
                rows.append([SHORT_DATASETS[dataset],str(atoms),str(cell['width']),f"{cell['parameters']['allocated_trainable']:,}",f'8/{atoms}'])
        self.compact_table(['Dataset','Templates','Predictor width','Stored parameters','TopK budget'],rows,[155,70,95,115,WIDTH-435])
        self.p('The approximately 64k parameter budget makes the coefficient predictor narrower when the dictionary grows. Thus 192-versus-64 changes capacity allocation as well as removing the template-count rank cap. Within each size, dense and TopK use identical widths and parameter counts. This is an exploratory follow-up designed after inspecting the main architecture, not a preregistered pure rank intervention.','small')
        self.p('A TopK head selects eight templates at each state, but its active set can change along the trajectory. Its field is therefore not permanently rank eight. Conversely, a full-rank 192-template dictionary does not guarantee that the coefficient model actually uses most templates.','small')

    def dictionary_results(self):
        self.title('06 | Dictionary-rank follow-up','Measured 64-versus-192 comparisons')
        index = {(r['dataset'],r['atoms'],r['head'],r['source']):r for r in self.supplement['aggregate']}
        for dataset in ('etth1_multi','exchange_multi'):
            self.p(SHORT_DATASETS[dataset],'h2')
            rows = []
            for atoms in (64,192):
                for head in ('dictionary_dense','dictionary_topk'):
                    for source in ('white','gp'):
                        r = index[(dataset,atoms,head,source)]
                        values = [f"{r['metrics'][m]['mean']:.4g} ± {r['metrics'][m]['sd']:.2g}" for m in ('fair_crps','joint_energy_scaled','roughness_ratio')]
                        rows.append([f"{atoms} / {'dense' if head=='dictionary_dense' else 'TopK8'}",source,*values])
            self.compact_table(['Templates / coefficients','Source','CRPS ↓','Joint energy ↓','Roughness ratio'],rows,[119,48,112,112,WIDTH-391])
        topk_worse = sum(index[(d,z,'dictionary_topk',s)]['metrics']['fair_crps']['mean'] > index[(d,z,'dictionary_dense',s)]['metrics']['fair_crps']['mean'] for d in ('etth1_multi','exchange_multi') for z in (64,192) for s in ('white','gp'))
        self.p(f'**The larger vocabulary does not rescue TopK at this budget.** TopK has higher mean CRPS than its dense control in {topk_worse}/8 dataset/source/size comparisons. Mean ± sample SD uses three seeds; all 48 runs are retained. Paired seed differences and context-block intervals are supplied separately.','small')
        rows = []
        for dataset in ('etth1_multi','exchange_multi'):
            changes = [index[(dataset,192,h,s)]['metrics']['fair_crps']['mean']-index[(dataset,64,h,s)]['metrics']['fair_crps']['mean'] for h in ('dictionary_dense','dictionary_topk') for s in ('white','gp')]
            usage = [r['validation_used_atom_union'] for r in self.rank_runs if r['dataset']==dataset and r['atoms']==192 and r['head']=='dictionary_topk']
            rows.append([SHORT_DATASETS[dataset],f'{sum(v<0 for v in changes)}/4',f'{min(changes):+.4g} to {max(changes):+.4g}',f'{min(usage)}-{max(usage)} / 192'])
        self.compact_table(['Dataset','192 lowers CRPS','CRPS change range','Templates visited'],rows,[155,100,135,WIDTH-390])
        self.p('Template usage is the union over the complete validation split, separately for each fitted TopK model. It can be much smaller than the dictionary despite full learned matrix rank. Improved span and actual template utilization are separate questions; these descriptive counts do not establish a universal benefit of a larger or sparser dictionary.','small')

    def verification(self):
        self.title('07 | Reliability and scope','What is checked, and what remains uncertain')
        final_count = sum(r['best_step']==self.cfg['steps'] for r in self.records if r['head'] in self.cfg['heads'])
        self.p(f'**Finite training budget.** {final_count}/264 new runs select the final checked step. The separate 24-fit full-rank dictionary follow-up is reported on the preceding pages. This flags possible optimization limits without proving convergence or failure. The suite uses one learning rate and fixed auxiliary-loss strengths; a weak sparse result does not establish that the method family cannot work.')
        rows = []
        for dataset in self.cfg['datasets']:
            solvers = [r['solver_sensitivity'] for r in self.records if r['dataset']==dataset
                       and r['head'] in self.cfg['heads'] and r['solver_sensitivity']]
            assert len(solvers)==22
            rows.append([SHORT_DATASETS[dataset],f"{max(s['endpoint_rmse'] for s in solvers):.4g}",
                         f"{max(abs(s['refined']['fair_crps']-s['base']['fair_crps']) for s in solvers):.4g}"])
        self.compact_table(['Euler 64 to 128, new heads','Max endpoint RMSE','Max |CRPS change|'],rows,[225,135,WIDTH-360])
        self.p('Solver checks use seed 31, the first eight selected contexts and all 11 new heads with both sources. They are a limited sensitivity check, not proof of convergence on every forecast.','small')
        self.p('Independent numerical and data checks','h2')
        self.p(f'Independent audits cover all 264 main and 24 supplementary fits and rescore all 96 frozen controls. The checks reconstruct training scalers, splits, source covariance and PCA; verify artifact hashes and checkpoint selection; replay full validation and test velocities on CPU; and recompute proper scores, roughness and saved doubled-Euler arrays. They also reconstruct Elastic Net predictions and training residuals. This exporter separately checked {self.aggregate_checks:,} aggregate mean/SD pairs against saved run records.')
        fwd = max(r['nonzero_forward']['max_abs_difference'] for r in self.parity['rows'])
        grad = max(g['max_abs_difference'] for r in self.parity['rows'] for g in r['gradient_comparisons'])
        update = max(r['post_adamw_step']['max_abs_difference'] for r in self.parity['rows'])
        self.p(f'The complete 122-test suite passed on MPS. All 36 actual-MPS model checks passed: nine sparse/dense variants at 1, 3, 7 and 8 channels. Maximum CPU-MPS difference was {fwd:.3g} in forward output, {grad:.3g} in gradients including auxiliary terms, and {update:.3g} after an AdamW update. Sparse supports agreed. Stochastic gates have private checkpointed RNG; forward calls leave training-data RNG untouched.','small')
        self.p('Data provenance and scientific limits','h2')
        self.p('ETTh1 uses 48 observed and 24 future hours, with 12/4/4-month row splits. Exchange uses 32 observed and 16 future steps, with a 70/10/20 chronological split. The synthetic problem uses 32 to 16 steps and three correlated channels. These protocols differ from the original papers. Windows and train-standardized scales match the frozen extension controls.')
        self.p('Context bootstrap intervals are exploratory: iid resampling for independent GP contexts, circular blocks of four and eight for real contexts. They condition on the chosen periods and trained seeds; block sensitivity and multiple comparisons limit generalization. A model can have favorable marginal CRPS while missing cross-channel structure, calibration or low-variance components.','small')
        self.p('Primary methods and reproducibility','h2')
        self.p('[TopK SAE](https://arxiv.org/abs/2406.04093) and its [official code](https://github.com/openai/sparse_autoencoder); [CRsAE author implementation](https://github.com/btolooshams/crsae); [hard-concrete L0 regularization](https://arxiv.org/abs/1712.01312); [Time-MoE](https://github.com/Time-MoE/Time-MoE); [MultiTask Elastic Net documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html). These motivate small independent adaptations; this report does not claim to reproduce their published results.','small')
        self.p('Run `scripts/run_sparse.sh`, or reproduce its stages with `fm_stress.sparse_experiment`, `fm_stress.sparse_baselines_qr`, `scripts/check_sparse_extension.py`, `fm_stress.analyze_sparse` and this builder. Configuration: `configs/sparse_extension.json`. Saved artifacts: `results/sparse-extension`; complete tables and paired contrasts: `output/sparse-analysis`. Model adaptations are documented in `docs/sparse_model_design.md`.','small')


def footer(canvas,doc):
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont('RS',8)
    canvas.drawString(40,25,'FM stress test | Whitening and sparse representations')
    canvas.drawRightString(A4[0]-40,25,str(doc.page))
    canvas.restoreState()


def main():
    register_fonts()
    report = SparseReport()
    report.opening()
    report.design()
    for dataset in report.cfg['datasets']:
        report.dataset(dataset)
    report.diagnostics()
    for dataset in report.cfg['datasets']:
        report.pc_metrics(dataset)
    report.dictionary_geometry()
    report.dictionary_results()
    report.verification()
    destination = ROOT / 'output/pdf/sparse-whitening-extension.pdf'
    SimpleDocTemplate(str(destination),pagesize=A4,leftMargin=40,rightMargin=40,
                      topMargin=38,bottomMargin=42,
                      title='FM stress test: joint whitening and sparse representations',
                      author='Egor Serov; Kirill Frolov; Daniil Koblov; Vasilii Lyamin').build(
                          report.story,onFirstPage=footer,onLaterPages=footer)
    metadata = subprocess.run(['pdfinfo',str(destination)],check=True,capture_output=True,text=True).stdout
    pages = int(re.search(r'^Pages:\s+(\d+)',metadata,re.MULTILINE).group(1))
    if pages != 14:
        raise AssertionError(f'Expected 14 intentional pages; got {pages}. Inspect layout before delivery.')
    print(destination)


if __name__ == '__main__':
    main()
