#!/usr/bin/env python3
"""Render the audited matched-history oracle findings (no model evaluation)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/followup-oracle';OUT.mkdir(parents=True,exist_ok=True)
s=json.loads((ROOT/'results/followup-oracle/summary.json').read_text())
a=json.loads((ROOT/'results/followup-oracle/independent_verification.json').read_text())
heads=['mlp','s4','whitened_mlp'];labels=['MLP','S4','Whitened MLP'];colors=['#2867a5','#d5653e','#338875']
fig,axs=plt.subplots(1,3,figsize=(12,4.1))
values=[[s['summary']['gp|'+h]['matched']['residual']['mean_relative_error'] for h in heads],
        [s['summary']['gp|'+h]['matched']['residual']['covariance_weighted_relative_error'] for h in heads],
        [s['summary']['gp|'+h]['matched']['fresh_velocity']['oracle_discrepancy']['residual'] for h in heads]]
for ax,v,title in zip(axs,values,['Equal weight to every input direction','Weight by conditional input covariance','Velocity error against matched oracle']):
 ax.bar(labels,v,color=colors,width=.65)
 ax.set_title(title,loc='left',fontsize=10)
 ax.set_ylim(0,max(v)*1.2)
 for i,x in enumerate(v):ax.text(i,x+max(v)*.025,f'{x:.4f}',ha='center',fontsize=10)
 ax.tick_params(axis='x',labelsize=9);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axs[0].set_ylabel('Residual Jacobian relative error');axs[1].set_ylabel('Weighted residual Jacobian error');axs[2].set_ylabel('Residual velocity MSE')
fig.suptitle('GP source: the derivative metric changes the head ranking',x=.02,ha='left',fontsize=14)
fig.text(.02,.01,'25 saved checkpoints. Derivatives: 256 histories × 5 times. Velocity: 2,048 fresh histories. Covariance weighting is a post-hoc local diagnostic.',fontsize=8,color='#555555')
fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'jacobian_metric_sensitivity.png',dpi=180);plt.close(fig)
rows=[]
for h,label in zip(heads,labels):
 r=s['summary']['gp|'+h]
 rows.append('<tr><td>'+label+'</td>'+''.join(f'<td>{v:.5f}</td>' for v in [r['privileged']['residual']['mean_relative_error'],r['matched']['residual']['mean_relative_error'],r['matched']['residual']['covariance_weighted_relative_error'],r['matched']['fresh_velocity']['label_mse']['residual'],r['matched']['fresh_velocity']['oracle_discrepancy']['residual']])+'</tr>')
support=[r for r in a['perturbation_support'] if r['source']=='gp']
html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Information-matched oracle follow-up</title><style>
body{font:17px/1.55 system-ui,sans-serif;color:#23323b;max-width:1120px;margin:45px auto;padding:0 24px;background:#fbfcfd}h1{font-size:34px;line-height:1.15}h2{font-size:23px;margin-top:35px}p{max-width:1000px}.intro{font-size:21px;color:#235b70;border-left:4px solid #338875;padding-left:18px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{text-align:right;padding:12px;border-bottom:1px solid #dbe2e6}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto;background:white}code{font-size:14px}small{color:#596b76}a{color:#2867a5}</style>
<h1>Same information, different Jacobian metric</h1><small>25 October 2026 · Supplementary audit of all 25 primary checkpoints · No retraining</small>
<p class="intro">Giving the reference exactly the history information seen by the networks preserves S4’s advantage under the earlier unweighted Jacobian metric. But weighting directions by their conditional variance reverses that ranking—and agrees with the residual velocity errors. A Jacobian ranking alone does not establish which head learns the relevant flow better.</p>
<h2>What was corrected</h2><p>The earlier Gaussian reference knew each window’s original history, mean and scale. The networks see only centered and standardized history. We integrated out the unknown mean and scale analytically. The resulting future distribution is Student-t, and its flow field is generally nonlinear: the interpolated point changes the posterior weights of its Gaussian scale components. The derivative of those weights must be included.</p>
<h2>GP-source results</h2><p>Lower is better. Jacobian entries average five optimization seeds and five flow times. The velocity columns use a separate set of 2,048 fresh histories. These are observations from the original trained models, not new model-selection results.</p>
<table><tr><th>Head</th><th>Privileged Jacobian error</th><th>Matched Jacobian error</th><th>Covariance-weighted Jacobian error</th><th>Residual label MSE</th><th>Residual matched-oracle discrepancy</th></tr>'''+''.join(rows)+'''</table>
<img src="jacobian_metric_sensitivity.png" alt="Changing Jacobian weighting changes the head ranking">
<p>S4 is closer under equal weighting of all raw-coordinate directions in every seed, yet has larger residual velocity error. The covariance-weighted derivative diagnostic instead ranks whitened MLP, raw MLP, then S4. This secondary weighting was added after the first oracle comparison and is explicitly exploratory. It measures local sensitivity to covariance-scaled perturbations; it is not generated forecast quality or a new assignment success criterion.</p>
<h2>Why apparently small perturbations need care</h2><p>A perturbation of Euclidean norm 0.1, uniformly oriented in velocity PCs3–16, has expected root-mean-square conditional Mahalanobis radius between '''+f"{min(r['rms_mahalanobis_radius_mean'] for r in support):.1f} and {max(r['rms_mahalanobis_radius_mean'] for r in support):.1f}"+''' across the five GP-source flow times. A covariance-scaled 16-dimensional draw has RMS radius 4. The conditional distribution is a mixture, so this is a covariance geometry diagnostic, not a Gaussian probability statement. It nevertheless shows that a small Euclidean perturbation can heavily probe low-variance directions.</p>
<h2>Independent checks</h2><ul><li>Six tests: independent raw-history conditional simulation, observable small-scale branch, endpoint fields/noise, quadrature convergence, finite-difference derivatives, and truncated-scale integration.</li><li>An independent derivation uses an orthonormal basis of centered history and the jointly Gaussian pair (centered history, future minus history mean). It reproduces means, scale rates and covariance.</li><li>Adaptive scalar integration with direct covariance solves reproduces the eigensystem/Gauss–Laguerre implementation.</li><li>All 250 residual Jacobian summaries replay from saved arrays; source/checkpoint hashes match.</li><li>64→128 quadrature changes are below 8×10⁻¹³ relative RMS. Finite-difference derivative disagreement is below 1.6×10⁻⁶. No probe uses the small-scale branch; its largest omitted posterior probability is below 6×10⁻¹⁸.</li></ul>
<h2>What this establishes—and what remains open</h2><p>The earlier raw-history reference had an information advantage, but that alone did not create the unweighted S4/MLP ordering. The stronger conclusion that the architecture’s optimization failure has been identified is still unsupported: the ranking depends on the derivative metric and disagrees with actual residual field errors. The matched oracle sees the complete normalized history; a learned encoder may discard some of that information. No causal claim about optimizer conditioning, no universal head recommendation, and no claim about published TimeFlow or TSFlow follows from this audit.</p>
<p><a href="oracle_information_comparison.png">Reference-information comparison</a> · <a href="matched_oracle_pc_response.png">Per-PC response gains</a></p>
<p><small>Reproduce: scripts/study_normalized_history_oracle.py, scripts/check_normalized_history_oracle.py, scripts/report_normalized_history_oracle.py. Full arrays, per-seed results, numerical checks and SHA-256 hashes: results/followup-oracle/. Mathematical protocol: docs/followup_oracle_protocol.md.</small></p></html>'''
(OUT/'report.html').write_text(html)
print(OUT/'report.html')
