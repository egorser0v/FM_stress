"""Descriptive source-mixture analysis with data draws kept separate from seeds.

The mixture parameter is a covariance weight, not a categorical mixture of
white and GP trajectories. Cross-source velocity PC scores use different bases
and must not be interpreted as the same regression target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

HEAD_LABELS = {"mlp": "MLP", "s4": "S4", "whitened_mlp": "Whitened MLP", "balanced_mlp": "MLP / balanced loss", "whitened_raw_mlp": "Whitened MLP / raw loss"}
COLORS = {"mlp": "#137b85", "s4": "#d27437", "whitened_mlp": "#7763b7", "balanced_mlp": "#417b41", "whitened_raw_mlp": "#ba5196"}
METRICS = ["pc1_mse", "pc2_mse", "residual_mse", "residual_normalized_mse",
           "roughness_ratio", "fair_crps",
           "target_pca_tail_ratio"]


def sample_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite sample")
    return {"mean": float(values.mean()), "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max()), "n": len(values)}


def summarize_cells(rows, metrics=METRICS):
    """Equal-weight draw means; never call six fits six independent datasets."""
    cells = []
    for alpha, head in sorted({(r["alpha"], r["head"]) for r in rows}):
        selected = [r for r in rows if r["alpha"] == alpha and r["head"] == head]
        by_draw = []
        for draw in sorted({r["data_seed"] for r in selected}):
            samples = [r for r in selected if r["data_seed"] == draw]
            by_draw.append({"data_seed": draw, "seeds": sorted(r["seed"] for r in samples),
                            "metrics": {m: sample_summary([r[m] for r in samples]) for m in metrics if all(m in r for r in samples)},
                            "literal_pass_count": sum(bool(r.get("literal_success", False)) for r in samples)})
        common = set.intersection(*(set(x["metrics"]) for x in by_draw))
        aggregate = {}
        for metric in sorted(common):
            draw_means = [x["metrics"][metric]["mean"] for x in by_draw]
            aggregate[metric] = {"mean": float(np.mean(draw_means)), "draw_means": draw_means,
                                 "draw_min": min(draw_means), "draw_max": max(draw_means)}
        cells.append({"alpha": alpha, "head": head, "runs": len(selected),
                      "data_draws": len(by_draw), "by_draw": by_draw, "metrics": aggregate})
    return cells


def paired_contrasts(rows, metrics=METRICS):
    """Return per-draw paired effects, plus source-by-head interactions.

    A sign means a numerical increase or decrease; roughness closer to one is
    preferable and its signed difference is not a quality ordering.
    """
    index = {(r["data_seed"], r["seed"], r["alpha"], r["head"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate run identity")
    heads = sorted({r["head"] for r in rows})
    main_heads = [h for h in heads if h not in {"balanced_mlp", "whitened_raw_mlp"}]
    whitened = next((h for h in heads if h in {"whitened_mlp"}), None)
    pairs = [("s4", "mlp")] + ([(whitened, "mlp")] if whitened else [])
    alpha_values = sorted({r["alpha"] for r in rows})
    draws = sorted({r["data_seed"] for r in rows})
    output = []

    def append(kind, alpha, first, second, metric, terms):
        per_draw = []
        for draw in draws:
            seed_values = []
            for seed in sorted({r["seed"] for r in rows if r["data_seed"] == draw}):
                value = 0.0
                for weight, source_alpha, head in terms:
                    record = index.get((draw, seed, source_alpha, head))
                    if record is None:
                        raise ValueError(f"Unpaired contrast: draw={draw}, seed={seed}, alpha={source_alpha}, head={head}")
                    if metric not in record:
                        return
                    value += weight * record[metric]
                seed_values.append({"seed": seed, "difference": value})
            per_draw.append({"data_seed": draw, "paired_seed_effects": seed_values,
                             **sample_summary([x["difference"] for x in seed_values])})
        output.append({"kind": kind, "alpha": alpha, "first": first, "second": second,
                       "metric": metric, "by_draw": per_draw,
                       "equal_draw_mean_difference": float(np.mean([x["mean"] for x in per_draw]))})

    for alpha in alpha_values:
        for metric in metrics:
            for first, second in pairs:
                if first not in heads or second not in heads:
                    continue
                append("head_difference", alpha, first, second, metric,
                       [(1, alpha, first), (-1, alpha, second)])
                if alpha != 0 and 0 in alpha_values:
                    append("head_by_source_interaction_vs_white", alpha, first, second, metric,
                           [(1, alpha, first), (-1, alpha, second), (-1, 0, first), (1, 0, second)])
            if alpha != 0 and 0 in alpha_values:
                for head in main_heads:
                    append("source_difference_vs_white", alpha, head, head, metric,
                           [(1, alpha, head), (-1, 0, head)])
    # The extra alpha=1 heads form a 2x2 input-coordinate / loss-weighting control.
    factorial = [("balanced_mlp", "mlp"), ("whitened_mlp", "whitened_raw_mlp"),
                 ("whitened_raw_mlp", "mlp"), ("whitened_mlp", "balanced_mlp")]
    for first, second in factorial:
        if first in heads and second in heads:
            for metric in metrics:
                append("gp_coordinate_loss_control", 1., first, second, metric,
                       [(1, 1., first), (-1, 1., second)])
    if all(h in heads for h in ["balanced_mlp", "whitened_raw_mlp", "whitened_mlp", "mlp"]):
        for metric in metrics:
            append("gp_coordinate_by_loss_interaction", 1., "whitened_mlp", "mlp", metric,
                   [(1, 1., "whitened_mlp"), (-1, 1., "whitened_raw_mlp"),
                    (-1, 1., "balanced_mlp"), (1, 1., "mlp")])
    return output


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(rows, cells, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "savefig.facecolor": "white"})
    alphas = sorted({r["alpha"] for r in rows})
    draws = sorted({r["data_seed"] for r in rows})
    heads = [h for h in ("mlp", "s4", "whitened_mlp") if any(r["head"] == h for r in rows)]
    xs = np.arange(len(alphas))
    labels = [f"{a:g}" for a in alphas]
    outputs = []
    specifications = [
        ("forecast_scores.png", [("fair_crps", "Marginal CRPS ↓", False), ("fair_energy_score", "Joint path energy score ↓", False)]),
        ("original_metrics.png", [("pc1_mse", "PC1 velocity MSE ↓", True), ("residual_mse", "Residual PC3–16 MSE ↓", True), ("roughness_ratio", "Generated / target roughness", False)]),
        ("common_basis.png", [("fixed_target_residual_mse", "Velocity error in fixed target PCs3–16 ↓", True), ("target_pca_tail_ratio", "Generated / target PC7–16 second moment", True)]),
    ]
    for filename, panels in specifications:
        if not all(any(metric in r for r in rows) for metric, _, _ in panels):
            continue
        fig, axes = plt.subplots(len(draws), len(panels), figsize=(5.0 * len(panels), 3.5 * len(draws)), squeeze=False)
        for row_index, draw in enumerate(draws):
            for column_index, (metric, title, log) in enumerate(panels):
                ax = axes[row_index, column_index]
                for head in heads:
                    means, lows, highs = [], [], []
                    for alpha in alphas:
                        values = [r[metric] for r in rows if r["data_seed"] == draw and r["alpha"] == alpha and r["head"] == head]
                        means.append(np.mean(values)); lows.append(min(values)); highs.append(max(values))
                    ax.plot(xs, means, marker="o", lw=2, color=COLORS[head], label=HEAD_LABELS[head])
                    ax.fill_between(xs, lows, highs, color=COLORS[head], alpha=.1)
                if metric.endswith("ratio"):
                    ax.axhline(1., color="#566775", ls="--", lw=1)
                if log:
                    ax.set_yscale("log")
                ax.set_xticks(xs, labels)
                ax.set_xlabel("GP covariance weight α (levels spaced equally)")
                ax.set_title(title)
                ax.grid(alpha=.15)
                if column_index == 0:
                    ax.set_ylabel(f"Independent dataset {row_index + 1}")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, ncol=len(heads), loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.01))
        fig.text(.5, .005, "Lines: mean over training seeds. Bands: seed minimum–maximum within one dataset; not confidence intervals.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .025, 1, .965))
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
        outputs.append(filename)
    return outputs


def format_number(value):
    if value is None:
        return "—"
    if value != 0 and (abs(value) < .0001 or abs(value) >= 10000):
        return f"{value:.2e}"
    return f"{value:.4g}"


def measured_findings(summary):
    """Describe declared source comparisons without choosing an optimal alpha."""
    cells = {(c["alpha"], c["head"]): c for c in summary["cells"]}
    draws = summary["data_seeds"]
    def value(alpha, head, draw, metric="fair_crps"):
        return next(d["metrics"][metric]["mean"] for d in cells[(alpha,head)]["by_draw"] if d["data_seed"]==draw)
    def paired_values(alpha, head, metric="fair_crps", digits=4):
        return " / ".join(f"{value(alpha,head,d,metric):.{digits}f}" for d in draws)
    statements = []
    if all((a,"mlp") in cells for a in (0.,.5,1.)):
        statements.append(f"The 50:50 GP–white covariance mixture helps ordinary MLP in both datasets: CRPS is {paired_values(.5,'mlp')}, versus {paired_values(0.,'mlp')} for white noise and {paired_values(1.,'mlp')} for pure GP. These values are dataset 1 / dataset 2. This observed comparison does not establish an optimal mixing weight.")
    if all((a,"mlp") in cells for a in (0.,.99,1.)):
        statements.append(f"A 99% GP covariance is not equivalent to pure GP for small-variance output directions. MLP’s generated/target PC7–16 second-moment ratio is {paired_values(.99,'mlp','target_pca_tail_ratio',0)} at α=0.99, versus {paired_values(1.,'mlp','target_pca_tail_ratio',3)} at pure GP and {paired_values(0.,'mlp','target_pca_tail_ratio',1)} at white noise. The target energy in this tail is tiny: these large ratios do not mean the whole trajectory has thousands of times too much variance. Aggregate CRPS alone largely hides this change.")
    if all((1.,h) in cells for h in ("mlp","s4")):
        residual_count=sum(value(1.,"s4",d,"residual_mse")<value(1.,"mlp",d,"residual_mse") for d in draws)
        roughness_count=sum(abs(value(1.,"s4",d,"roughness_ratio")-1)<abs(value(1.,"mlp",d,"roughness_ratio")-1) for d in draws)
        statements.append(f"S4 gives lower GP-source forecast CRPS ({paired_values(1.,'s4')}) than MLP ({paired_values(1.,'mlp')}). But it has lower raw residual velocity error in {residual_count}/{len(draws)} datasets and roughness closer to target in {roughness_count}/{len(draws)}. The original claim that the unstructured head uniquely fails on residual structure is therefore not supported by these declared diagnostics; the modest forecast-score advantage is a different result.")
    if all((1.,h) in cells for h in ("mlp","whitened_mlp")):
        statements.append(f"Full whitening reduces MLP’s GP-source residual velocity error from {paired_values(1.,'mlp','residual_mse',5)} to {paired_values(1.,'whitened_mlp','residual_mse',5)}, while CRPS changes from {paired_values(1.,'mlp')} to {paired_values(1.,'whitened_mlp')}. Better small-direction regression does not automatically give a better probabilistic forecast. The coordinate/loss controls below identify which parts of whitening help and which hurt.")
    return statements


def html_report(summary, output):
    esc = html.escape
    cells = summary["cells"]
    alpha_values = sorted({c["alpha"] for c in cells})
    draws = summary["data_seeds"]
    cfg = summary.get("config", {})
    parts = []
    for metric, title in [("fair_crps", "CRPS"), ("residual_mse", "Residual velocity MSE"), ("roughness_ratio", "Roughness ratio")]:
        body = []
        for cell in cells:
            if cell["head"] not in {"mlp", "s4", "whitened_mlp"} or metric not in cell["metrics"]:
                continue
            numbers = [format_number(d["metrics"][metric]["mean"]) for d in cell["by_draw"]]
            body.append("<tr><td>" + f'{cell["alpha"]:g}' + "</td><td>" + esc(HEAD_LABELS[cell["head"]]) + "</td>" + "".join(f"<td>{n}</td>" for n in numbers) + "</tr>")
        parts.append(f'<section class="table-card"><h3>{title}</h3><table><thead><tr><th>α</th><th>Head</th>' + "".join(f"<th>Dataset {i+1}</th>" for i in range(len(draws))) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></section>")
    factorial_rows = []
    for cell in cells:
        if cell["alpha"] != 1:
            continue
        for d in cell["by_draw"]:
            if cell["head"] == "s4":
                continue
            factorial_rows.append("<tr>" + f'<td>{esc(HEAD_LABELS[cell["head"]])}</td><td>{draws.index(d["data_seed"])+1}</td>' + "".join(f'<td>{format_number(d["metrics"].get(m,{}).get("mean"))}</td>' for m in ["fair_crps", "pc1_mse", "residual_mse", "roughness_ratio"]) + "</tr>")
    captions = {"source_geometry.png": "Training PCA evaluated on independent validation draws. Intermediate mixtures are reported without requiring the endpoint rank-two gate.",
                "common_basis.png": "The target PCA is fixed across sources within each dataset. PC7–16 second moments use the shared training-target mean and do not use a variance floor; this measures marginal geometry, not conditional calibration.",
                "velocity_noise_reference.png": "The prior changes the stochastic velocity-regression problem itself. The reference conditions on full raw history, stronger information than the model receives. The saved decomposition retains its signed cross term; subtracting this reference from model MSE is not an unbiased measure of approximation error.",
                "actual_objective_gradients.png": "The same selected checkpoint is probed with raw error and its actual training objective. For balanced-loss heads these are different decompositions; the raw-loss heads serve as equality controls.",
                "gradient_diagnostics.png": "Raw-loss parameter-gradient diagnostics at selected checkpoints. Balanced-loss models are also probed using raw loss, which differs from their training objective."}
    def rendered_figure(p):
        return f'<figure><img src="{esc(p)}" alt="{esc(p.replace("_", " "))}"><figcaption class="small">{esc(captions.get(p,""))}</figcaption></figure>'
    main_figures = [p for p in summary.get("figures",[]) if p in {"forecast_scores.png","original_metrics.png"}]
    path_plots = [p for p in summary.get("figures",[]) if p.startswith("learned_paths_")]
    diagnostics = [p for p in summary.get("figures",[]) if p not in main_figures + path_plots + ["source_examples.png","actual_objective_gradients.png"]]
    figures = ''.join(rendered_figure(p) for p in main_figures)
    if path_plots:
        figures += '<details><summary>Inspect fixed forecast examples for both datasets</summary>' + ''.join(rendered_figure(p) for p in path_plots) + '</details>'
    diagnostic_figures = '<details><summary>Inspect geometry, raw-loss gradients and the privileged noise reference</summary>' + ''.join(rendered_figure(p) for p in diagnostics) + '</details>' if diagnostics else ''
    objective_figure = '<details><summary>Inspect gradients of the raw error and the actual training objective</summary>' + rendered_figure("actual_objective_gradients.png") + '</details>' if "actual_objective_gradients.png" in summary.get("figures",[]) else ''
    source_example = '<figure><img src="source_examples.png" alt="Gaussian source curves at all covariance mixture levels"></figure>' if "source_examples.png" in summary.get("figures",[]) else ""
    literal = sum(int(r["literal_success"]) for r in summary["rows"])
    final_count = sum(r["best_step"] == r["steps"] for r in summary["rows"])
    geometry_rows = []
    for alpha in alpha_values:
        entries = [g for g in summary.get("geometry",[]) if g["alpha"] == alpha]
        if entries:
            geometry_rows.append("<tr>" + f"<td>{alpha:g}</td><td>{entries[0]['theoretical_source_roughness']:.4f}</td>" + "".join(f"<td>{g['velocity_top2_energy']:.4f}</td><td>{g['floored_directions']}</td>" for g in entries) + "</tr>")
    geometry_table = ('<table><thead><tr><th>α</th><th>Expected source roughness</th>' + "".join(f"<th>Dataset {i+1}: E₂</th><th>Floored PCs</th>" for i in range(len(draws))) + '</tr></thead><tbody>' + ''.join(geometry_rows) + '</tbody></table>') if geometry_rows else ''
    target_roughness = ", ".join(f"dataset {i+1}: {next(g['validation_target_roughness'] for g in summary.get('geometry',[]) if g['data_seed']==d):.4f}" for i,d in enumerate(draws)) if summary.get('geometry') else ''
    findings = "".join("<li>" + esc(x) + "</li>" for x in measured_findings(summary))
    interactions = [c for c in summary.get("contrasts",[]) if c["kind"]=="head_by_source_interaction_vs_white" and c["first"]=="s4" and c["metric"]=="fair_crps"]
    interaction_rows = "".join(f"<tr><td>{c['alpha']:g}</td>"+"".join(f"<td>{d['mean']:+.5f}</td>" for d in c["by_draw"])+"</tr>" for c in interactions)
    interaction_table = ('<h3>Does the mixture change S4’s gap to MLP?</h3><table><thead><tr><th>α</th>' + ''.join(f'<th>Dataset {i+1}: shift in CRPS gap</th>' for i in range(len(draws))) + '</tr></thead><tbody>'+interaction_rows+'</tbody></table><p class="small">Each value is [(S4 − MLP) at α] − [(S4 − MLP) at white noise], paired within training seed and then averaged. A negative shift favors S4 relative to MLP; it does not necessarily mean S4 has lower CRPS. These are mean differences, not significance tests.</p>') if interactions else ''
    cell_index={(c["alpha"],c["head"]):c for c in cells}
    coordinate_finding=""
    def gp_numbers(head,metric):
        return " / ".join(f"{d['metrics'][metric]['mean']:.4f}" for d in cell_index[(1.,head)]["by_draw"])
    if all((1.,h) in cell_index for h in ["mlp","whitened_mlp","whitened_raw_mlp","balanced_mlp"]):
        coordinate_finding = f"<p>Changing only the coordinates improves CRPS in both datasets: {gp_numbers('mlp','fair_crps')} → {gp_numbers('whitened_raw_mlp','fair_crps')}. Changing only the loss sharply worsens it to {gp_numbers('balanced_mlp','fair_crps')}. The full whitening intervention gives {gp_numbers('whitened_mlp','fair_crps')}: its lower residual error comes with slightly worse forecast CRPS than ordinary MLP. Coordinates and error weighting therefore have distinct effects under this fixed training budget.</p>"
    solver_changes=[abs(r["solver_sensitivity"]["score_changes"]["fair_crps"]) for r in summary.get("solver_sensitivity",[])]
    solver_note = (f"<li>Doubling Euler steps from 64 to 128 on the first eight contexts of seed 41 changes mean CRPS by at most {max(solver_changes):.5f} across checked cells. This is a limited solver sensitivity check, not a global convergence test.</li>" if solver_changes else "")
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Source smoothness and forecasting heads</title><style>
:root{{--ink:#18303e;--muted:#526570;--teal:#137b85;--line:#dce6e9;--pale:#edf5f5}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f8;color:var(--ink);font:17px/1.6 system-ui,-apple-system,sans-serif}}main{{max-width:1260px;margin:auto;background:white;padding:60px 64px}}h1{{font-size:48px;line-height:1.1;letter-spacing:-1.5px;margin:12px 0 20px;max-width:930px}}h2{{font-size:29px;line-height:1.2;margin-top:48px;border-top:1px solid var(--line);padding-top:28px}}h3{{font-size:19px;line-height:1.3}}p{{max-width:1050px}}.kicker{{color:var(--teal);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:13px}}.lede{{font-size:22px;color:var(--muted)}}.strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:34px 0}}.stat{{border-top:4px solid var(--teal);padding:14px 0}}.stat b{{display:block;font-size:33px}}.stat span{{font-size:14px;color:var(--muted)}}.formula{{background:var(--pale);border-left:4px solid var(--teal);padding:20px 24px;font-size:23px}}.note{{background:#f6f2e9;border-left:4px solid #c39442;padding:15px 22px}}table{{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;width:100%}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:var(--pale);font-weight:650}}.tables{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;align-items:start}}figure{{margin:30px 0}}img{{display:block;width:100%;height:auto}}a{{color:var(--teal)}}details{{border:1px solid var(--line);border-radius:8px;padding:15px 20px;margin:20px 0}}summary{{cursor:pointer;color:var(--teal);font-weight:650}}.small{{font-size:14px;color:var(--muted)}}li{{margin:9px 0}}code{{font-size:.88em}}@media(max-width:900px){{main{{padding:28px 20px}}h1{{font-size:36px}}.tables{{display:block}}.strip{{grid-template-columns:repeat(2,1fr)}}}}@media print{{body{{background:white}}main{{padding:0}}figure,table,.stat{{break-inside:avoid}}h2{{break-after:avoid}}}}
</style><main><div class="kicker">Flow-matching stress test · measured follow-up</div>
<h1>How much does source smoothness change the head comparison?</h1>
<p class="lede">A controlled path from white noise to a smooth Gaussian-process source, with the original synthetic forecasting geometry restored.</p>
<div class="strip"><div class="stat"><b>{summary['verified_runs']}</b><span>completed MPS fits</span></div><div class="stat"><b>{len(alpha_values)}</b><span>source covariance weights</span></div><div class="stat"><b>{len(draws)} × {len(summary['seeds'])}</b><span>independent datasets × training seeds</span></div><div class="stat"><b>2 × 2</b><span>coordinate / loss control at GP endpoint</span></div></div>
<h2>What the measured results show</h2><ul>{findings}</ul><p class="small">These are prespecified comparisons of dataset-specific three-seed means. Percent differences describe this experiment, without a population significance claim or selection of an optimal α.</p>
{interaction_table}
<h2>One source family, five levels of correlation</h2>
<div class="formula">C<sub>α</sub> = (1 − α) I + α C<sub>GP</sub>, &nbsp; α ∈ {{{', '.join(f'{a:g}' for a in alpha_values)}}}</div>
<p>At α = 0 the source is white noise; at α = 1 it is the matched-kernel RBF Gaussian process. Intermediate values keep unit marginal variance while adding temporal correlation. Each source draw is Gaussian with this covariance. Its marginal variance is exactly one at every α; the GP endpoint divides the original kernel by 1.000001 to keep that variance fixed. This is a covariance mixture, not a random choice between two kinds of trajectory.</p>
<p>The target remains a smooth univariate GP, with lookback 32 and horizon 16. Context-only Sundial instance normalization is applied as in the original assignment. MLP and S4 use the same tiny encoder architecture and matched initial encoder weights; their encoders then train separately. The source is independent of the target conditional on the context. “Matched kernel” does not mean its covariance equals that of the normalized target.</p>
{source_example}
{geometry_table}<p class="small">E₂ is validation velocity variance in the first two training PCs. Expected source roughness is computed analytically from adjacent Gaussian differences. Normalized target validation roughness: {target_roughness}. Source covariance is fixed in normalized coordinates and need not match the normalized target covariance.</p>
<h2>Results by independent dataset</h2>
<p>Each entry is the mean of {len(summary["seeds"])} training seeds within one dataset. CRPS uses {cfg.get("n_ensemble_contexts",128)} held-out contexts with {cfg.get("ensemble_size",32)} forecasts each; raw velocity errors use all {cfg.get("n_test",8192):,} test windows, and the original roughness diagnostic uses {cfg.get("n_paths",1024):,} generated paths. Lower CRPS and velocity MSE are better; roughness should be near 1. The two dataset columns are deliberately kept separate. There are two independent training-data draws, not six.</p>
<div class="tables">{''.join(parts)}</div>
{figures}
<h2>What does whitening change?</h2>
<p>The GP endpoint crosses two choices: raw versus whitened model coordinates, and raw versus variance-balanced velocity loss. This separates the effect of coordinates from the effect of weighting small-variance errors. All reported errors and generated trajectories are mapped back to the original observation coordinates.</p>
{coordinate_finding}
<table><thead><tr><th>Model / objective</th><th>Dataset</th><th>CRPS ↓</th><th>PC1 MSE ↓</th><th>Residual MSE ↓</th><th>Roughness ≈ 1</th></tr></thead><tbody>{''.join(factorial_rows)}</tbody></table>
<p class="small">“Whitened MLP” uses balanced loss; “MLP / balanced loss” changes only the loss; “Whitened MLP / raw loss” changes only the coordinates. Full-dimensional whitening uses a declared eigenvalue floor, without discarding principal components.</p>
{objective_figure}
{diagnostic_figures}
<h2>How to read the evidence</h2>
<ul><li>Within one source, paired head contrasts use identical datasets, optimization seeds and evaluation draws. Source-by-head interactions compare how these paired gaps change from white noise.</li>
<li>Velocity PCA is source-specific. Across α, both the regression target and its PC axes change. Common target-PCA diagnostics and probabilistic forecast scores provide complementary comparisons in a shared output space.</li>
<li>The source-sweep gradient plot measures raw-observation-MSE gradients, including for models trained with balanced loss. A separate GP-endpoint plot compares this raw-error probe with the gradient of each model’s actual training objective. PC1, PC2 and residual losses use the same batch-size × horizon denominator, so their gradients sum to the raw-MSE gradient. Encoder parameters are excluded. Gradient norms are diagnostic measurements in each model’s parameterization. A large PC1/residual ratio does not by itself prove gradient competition causes forecast failure, and norms from different architectures are not invariant quantities.</li>
<li>The original literal rule is retained: residual/PC1 error and generated/target roughness must both lie in [0.8, 1.2]. {literal}/{summary['verified_runs']} runs pass. The rule can reject a model merely because residual error is much smaller than PC1 error; it is not a sufficient forecast-quality criterion.</li>
<li>{final_count}/{summary['verified_runs']} selected checkpoints are at the final update. These are fixed-budget comparisons, not a proof of convergence or an optimal architecture ranking. The additional study is exploratory and does not replace the original assignment results.</li>{solver_note}</ul>
<p class="small">Software-smoke checks used dataset seed 137041, so their 128 test windows overlap the beginning of main dataset 1. The complete main configuration was fixed before those checks; no hyperparameters, source weights, schedule or numerical code were changed in response to smoke outcomes. Model checkpoints are selected only on validation data.</p>
<div class="note">Uncertainty is shown as training-seed variation within each dataset and agreement or disagreement between the two datasets. No population confidence interval is claimed from two data draws. All planned cells, including failures, remain in the report.</div>
<h2>Reproducible results</h2><p><a href="runs.csv">Every fit</a> · <a href="contrasts.csv">Paired differences and interactions</a> · <a href="summary.json">Complete machine-readable summary</a></p>
<p class="small">The analyzer checks completion and hashes, recomputes saved-array metrics, and preserves dataset-specific means. For the independent execution audit and the exact fixed protocol, see the project’s results/prior-mixture directory. Report generated from saved numerical results; no outcome is inferred from schematic curves.</p></main></html>'''
    (output / "report.html").write_text(text)


def load_verified_rows(root):
    from .experiment import config_hash
    from .prior_mixture import identities, source_name
    from .metrics import PCA, evaluate, velocity_metrics
    from .ensemble_evaluation import context_scores
    cfg = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "independent_verification.json").read_text())
    if not audit.get("complete") or audit["config_sha256"] != config_hash(cfg):
        raise ValueError("A complete independent audit of this configuration is required")
    if audit["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Audit and manifest fingerprints differ")
    expected = set(identities(cfg))
    paths = sorted(root.glob("dataset_*/runs/*.json"))
    if audit.get("expected_runs") != len(expected) or audit.get("verified_runs") != len(expected):
        raise ValueError("Audit run counts do not match the complete plan")
    if set(audit["record_sha256"]) != {str(p.relative_to(root)) for p in paths}:
        raise ValueError("Audit record coverage does not match the current run files")
    if len(paths) != len(expected):
        raise ValueError(f"Expected {len(expected)} records, found {len(paths)}")
    for relative, digest in audit["geometry_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Audited geometry changed: {relative}")
    for relative, digest in audit["record_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Audited record changed: {relative}")
    rows, records, record_hashes = [], [], {}
    geometry = {d: json.loads((root / f"dataset_{d}" / "geometry.json").read_text()) for d in cfg["data_seeds"]}
    observed = set()
    for path in paths:
        r = json.loads(path.read_text())
        identity = (r["data_seed"], float(r["alpha"]), r["head"], r["seed"])
        if identity not in expected or identity in observed or r["status"] != "complete" or r["device"] != "mps":
            raise ValueError(f"Unexpected, duplicate or incomplete run: {path}")
        observed.add(identity)
        if r["config_sha256"] != config_hash(cfg) or r["fingerprint"] != manifest["fingerprint"]:
            raise ValueError(f"Stale run provenance: {path}")
        record_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset_dir = path.parent.parent
        samples = dataset_dir / "samples" / (r["run"] + ".npz")
        checkpoint = dataset_dir / "checkpoints" / (r["run"] + ".pt")
        for file, key in [(samples, "samples_sha256"), (checkpoint, "checkpoint_sha256")]:
            if hashlib.sha256(file.read_bytes()).hexdigest() != r[key]:
                raise ValueError(f"Changed artifact: {file}")
        geo = geometry[r["data_seed"]]
        def pca_from(p):
            return PCA(np.array(p["mean"]), np.array(p["components"]), np.array(p["eigenvalues"]), p["floor"])
        vpca = pca_from(geo["sources"][source_name(r["alpha"])]["pca"])
        tpca = pca_from(geo["target_pca"])
        with np.load(samples) as a:
            m = evaluate(a["prediction"], a["velocity"], a["generated"], a["target"][:len(a["generated"])], vpca)
            fixed = velocity_metrics(a["prediction"], a["velocity"], tpca)
            scores = context_scores(a["ensemble"], a["ensemble_target"])
            for stored, recomputed in [(r["metrics"], m), (r["fixed_target_basis_metrics"], fixed)]:
                for key, value in recomputed.items():
                    if isinstance(value, dict):
                        if value != stored[key]:
                            raise ValueError(f"Rule mismatch: {path} / {key}")
                    else:
                        np.testing.assert_allclose(value, stored[key], rtol=1e-10, atol=1e-12)
            for key, values in scores.items():
                np.testing.assert_allclose(values, r["per_context"][key], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(values.mean(), r["forecast_means"][key], rtol=1e-10, atol=1e-12)
            generated_moments = np.mean(tpca.project(a["generated"]) ** 2, axis=0)
            target_moments = np.mean(tpca.project(a["target"][:len(a["generated"])]) ** 2, axis=0)
            tail_ratio = float(generated_moments[6:].sum() / target_moments[6:].sum())
        row = {k: r[k] for k in ["run", "data_seed", "alpha", "source", "head", "seed", "steps", "best_step", "parameters", "seconds"]}
        row.update({k: v for k, v in m.items() if np.isscalar(v)})
        row.update({k: float(v.mean()) for k, v in scores.items() if k not in {"generated_roughness", "target_roughness"}})
        row.update({"fixed_target_" + k: v for k, v in fixed.items() if np.isscalar(v)})
        row.update(roughness_ratio=m["success"]["roughness_ratio"], literal_success=m["success"]["works"],
                   target_pca_tail_ratio=tail_ratio)
        rows.append(row)
        records.append(r)
    if observed != expected:
        raise ValueError("Missing planned identities")
    return cfg, manifest, rows, records, geometry, record_hashes


def additional_figures(records, geometry, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outputs = []
    draws = sorted(geometry)
    sources_for_example = sorted(geometry[draws[0]]["sources"].values(), key=lambda x: x["alpha"])
    horizon = len(sources_for_example[0]["covariance"])
    normals = np.random.default_rng(50371).standard_normal((4, horizon))
    fig, axes = plt.subplots(1, len(sources_for_example), figsize=(3.0 * len(sources_for_example), 3.2), sharex=True, sharey=True, squeeze=False)
    for ax, source in zip(axes[0], sources_for_example):
        covariance = np.asarray(source["covariance"])
        examples = normals @ np.linalg.cholesky(covariance).T
        for curve in examples:
            ax.plot(range(1,horizon+1), curve, alpha=.8, lw=1.5)
        ax.set_title(f"α = {source['alpha']:g}")
        ax.set_xlabel("Future position")
        ax.grid(alpha=.15)
    axes[0,0].set_ylabel("Source amplitude")
    fig.suptitle("Source draws from the declared covariance: white noise → smooth GP", fontsize=13)
    fig.text(.5,.01,"Four common Gaussian random vectors, transformed by each covariance.\nIllustrative source draws; no model predictions or measured outcomes.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.92)); fig.savefig(output / "source_examples.png",dpi=180);plt.close(fig)
    outputs.append("source_examples.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, draw in enumerate(draws):
        sources = sorted(geometry[draw]["sources"].values(), key=lambda x: x["alpha"])
        alphas = [x["alpha"] for x in sources]
        axes[0].plot(range(len(alphas)), [x["validation_velocity"]["top2_energy"] for x in sources], marker="o", label=f"Dataset {i+1}")
        axes[1].plot(range(len(alphas)), [x["floored_directions"] for x in sources], marker="o", label=f"Dataset {i+1}")
    axes[0].axhline(.9, ls="--", lw=1, color="#65747c", label="Original GP endpoint gate")
    axes[0].set_title("Held-out velocity variance in top two PCs")
    axes[0].set_ylabel("Explained variance fraction")
    axes[1].set_title("Directions affected by the whitening floor")
    axes[1].set_ylabel("Number of velocity PCs")
    for ax in axes:
        ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
        ax.set_xlabel("GP covariance weight α (levels spaced equally)")
        ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "source_geometry.png", dpi=180); plt.close(fig)
    outputs.append("source_geometry.png")
    # Selected-checkpoint gradients are raw-loss contributions on fixed probes.
    g_rows = []
    for r in records:
        for g in r["gradient_diagnostics"]:
            pc1 = g["groups"]["pc1"]["gradient_norm"]
            residual = g["groups"]["residual"]["gradient_norm"]
            g_rows.append({k:r[k] for k in ["data_seed", "alpha", "head", "seed"]} | {
                "step":g["step"], "selected_checkpoint":g.get("selected_checkpoint",False),
                "pc1_gradient_norm":pc1, "residual_gradient_norm":residual,
                "pc1_over_residual_gradient_norm":pc1/residual if residual else None,
                "pc1_residual_cosine":g["gradient_cosines"]["pc1_vs_residual"]})
    write_csv(output / "gradients.csv", g_rows)
    fig, axes = plt.subplots(len(draws), 2, figsize=(11, 3.7*len(draws)), squeeze=False)
    main_heads = ["mlp", "s4", "whitened_mlp"]
    for i, draw in enumerate(draws):
        for j, (metric, title) in enumerate([("pc1_over_residual_gradient_norm", "PC1 / residual gradient norm"), ("pc1_residual_cosine", "PC1–residual gradient cosine")]):
            ax = axes[i,j]
            for head in main_heads:
                means = []
                for alpha in alphas:
                    vals = [g[metric] for g in g_rows if g["data_seed"]==draw and g["alpha"]==alpha and g["head"]==head and g["selected_checkpoint"] and g[metric] is not None]
                    means.append(float(np.mean(vals)) if vals else np.nan)
                ax.plot(range(len(alphas)), means, marker="o", color=COLORS[head], label=HEAD_LABELS[head])
            if j==0: ax.set_yscale("log")
            else: ax.axhline(0,color="#73818a",ls="--",lw=1); ax.set_ylim(-1.05,1.05)
            ax.set_xticks(range(len(alphas)),[f"{a:g}" for a in alphas]); ax.set_xlabel("GP covariance weight α")
            ax.set_title(f"Dataset {i+1}: {title}"); ax.grid(alpha=.15)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper center",frameon=False)
    fig.text(.5,.005,"Selected checkpoints; means over seeds. Within-model diagnostics.\nNot parameterization-invariant comparisons or causal proof.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96)); fig.savefig(output / "gradient_diagnostics.png",dpi=180); plt.close(fig)
    outputs.append("gradient_diagnostics.png")
    return outputs


def path_figures(root, cfg, records, output):
    """Fixed first context and first seed; no result-based example selection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .prior_mixture import source_name
    outputs = []
    seed = cfg["seeds"][0]
    alphas = cfg["alphas"]
    heads = ["mlp", "s4", "whitened_mlp"]
    for dataset_number, draw in enumerate(cfg["data_seeds"], 1):
        fig, axes = plt.subplots(len(heads), len(alphas), figsize=(3.0*len(alphas), 7.7), sharex=True, sharey=True, squeeze=False)
        all_values = []
        for i, head in enumerate(heads):
            for j, alpha in enumerate(alphas):
                run = next(r for r in records if r["data_seed"] == draw and r["seed"] == seed and r["head"] == head and r["alpha"] == alpha)
                path = root / f"dataset_{draw}" / "samples" / (run["run"] + ".npz")
                with np.load(path) as sample:
                    history = sample["ensemble_history"][0][-8:]
                    target = sample["ensemble_target"][0]
                    generated = sample["ensemble"][0, :8]
                ax = axes[i,j]
                history_x = np.arange(1-len(history), 1)
                future_x = np.arange(len(target)+1)
                ax.plot(history_x, history, color="#778590", lw=1.7, label="Observed history", zorder=4)
                for k, curve in enumerate(generated):
                    ax.plot(future_x, np.r_[history[-1],curve], color=COLORS[head], alpha=.3, lw=1.1, label="Generated draws" if k==0 else None)
                ax.plot(future_x, np.r_[history[-1],target], color="#15222b", lw=1.7, label="Held-out target draw", zorder=5)
                ax.axvline(0,color="#83939d",ls=":",lw=1)
                if i==0: ax.set_title(f"α = {alpha:g}")
                if j==0: ax.set_ylabel(HEAD_LABELS[head]+"\nNormalized value")
                if i==len(heads)-1: ax.set_xlabel("Time relative to forecast origin")
                ax.grid(alpha=.12)
                all_values.extend([history.reshape(-1), target.reshape(-1), generated.reshape(-1)])
        values = np.concatenate(all_values)
        low, high = float(values.min()), float(values.max())
        margin = max(.05, .06*(high-low))
        axes[0,0].set_ylim(low-margin,high+margin)
        axes[0,0].set_xlim(1-len(history),len(target))
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.5,.957),frameon=False)
        fig.suptitle(f"Independent dataset {dataset_number} · fixed first test context · training seed {seed}",fontsize=14,y=.995)
        fig.text(.5,.008,"First eight ensemble draws, identical axis range across all cells. The target is one possible future.\nA fixed illustrative context is not overall performance evidence; generated paths need not coincide with its target draw.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.92))
        filename=f"learned_paths_dataset{dataset_number}.png"
        fig.savefig(output/filename,dpi=180);plt.close(fig)
        outputs.append(filename)
    return outputs


def objective_gradient_report(artifact, records, fingerprint, output, record_hashes):
    """Integrate the separately recomputed selected-checkpoint objective probes."""
    if artifact is None:
        return None, []
    artifact = Path(artifact)
    report = json.loads(artifact.read_text())
    if report["status"] != "complete" or report["fingerprint"] != fingerprint or report["completed"] != len(records):
        raise ValueError("Complete matching training-objective gradient diagnostics required")
    project = Path(__file__).resolve().parents[1]
    for relative,digest in report["source_sha256"].items():
        if hashlib.sha256((project/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Gradient postprocessor source changed: {relative}")
    record_index = {(r["data_seed"],r["run"]):r for r in records}
    observed = set()
    rows = []
    for row in report["rows"]:
        identity = (row["data_seed"],row["run"])
        if identity not in record_index or identity in observed:
            raise ValueError("Duplicate or unexpected gradient diagnostic")
        observed.add(identity)
        record_path=f"dataset_{row['data_seed']}/runs/{row['run']}.json"
        if row["run_record_sha256"] != record_hashes[record_path]:
            raise ValueError("Training-objective gradient run record changed")
        if row["checkpoint_sha256"] != record_index[identity]["checkpoint_sha256"]:
            raise ValueError("Training-objective gradient checkpoint changed")
        for mode in ("raw", "training_objective"):
            stats = row[mode+"_summary"]
            rows.append({k:row[k] for k in ("data_seed","alpha","head","seed","selected_step")} | {"probe_objective":mode,**stats})
    if observed != set(record_index):
        raise ValueError("Missing gradient diagnostic identities")
    write_csv(output/"selected_objective_gradients.csv",rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    draws = sorted({r["data_seed"] for r in rows})
    heads = ["mlp","balanced_mlp","whitened_raw_mlp","whitened_mlp"]
    fig, axes = plt.subplots(len(draws),2,figsize=(12,4*len(draws)),squeeze=False)
    labels = ["Raw coordinates\nraw loss","Raw coordinates\nbalanced loss","Whitened coordinates\nraw loss","Whitened coordinates\nbalanced loss"]
    for i, draw in enumerate(draws):
        for j,(metric,title) in enumerate([("pc1_over_residual_gradient_norm","PC1 / residual gradient norm"),("pc1_residual_cosine","PC1–residual gradient cosine")]):
            ax=axes[i,j]
            for mode,color,offset,label in [("raw","#607681",-.15,"Raw-error probe"),("training_objective","#a55729",.15,"Actual training-objective probe")]:
                for k,head in enumerate(heads):
                    values=[r[metric] for r in rows if r["data_seed"]==draw and r["alpha"]==1 and r["head"]==head and r["probe_objective"]==mode and r[metric] is not None]
                    if not values:continue
                    ax.scatter(np.full(len(values),k+offset),values,color=color,s=22,alpha=.45)
                    ax.scatter(k+offset,np.mean(values),color=color,marker="D",s=42,label=label if k==0 else None)
            if j==0:ax.set_yscale("log")
            else:ax.axhline(0,color="#73818a",ls="--",lw=1);ax.set_ylim(-1.05,1.05)
            ax.set_xticks(range(len(heads)),labels,fontsize=8)
            ax.set_title(f"Dataset {i+1}: {title}");ax.grid(axis="y",alpha=.15)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,ncol=2,loc="upper center",frameon=False)
    fig.text(.5,.006,"GP endpoint: circles are training seeds; diamonds are seed means. Parameters of the velocity head only.\nRaw and actual-objective probes coincide for raw-loss models. Gradient ratios are parameterization dependent.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.955))
    fig.savefig(output/"actual_objective_gradients.png",dpi=180);plt.close(fig)
    return {"artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),"completed":report["completed"],"rows":rows},["actual_objective_gradients.png"]


def oracle_report(artifact, records, fingerprint, output, record_hashes):
    """Report source-dependent regression noise without claiming attainability."""
    if artifact is None:
        return None, []
    artifact=Path(artifact)
    report=json.loads(artifact.read_text())
    if report["status"]!="complete" or report["fingerprint"]!=fingerprint or report["completed"]!=len(records):
        raise ValueError("Complete matching privileged-oracle diagnostic required")
    project=Path(__file__).resolve().parents[1]
    for relative,digest in report["source_sha256"].items():
        if hashlib.sha256((project/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Oracle postprocessor source changed: {relative}")
    index={(r["data_seed"],r["run"]):r for r in records}
    observed=set()
    for row in report["rows"]:
        identity=(row["data_seed"],row["run"])
        if identity not in index or identity in observed:
            raise ValueError("Unexpected or duplicate oracle row")
        observed.add(identity)
        relative=f"dataset_{row['data_seed']}/runs/{row['run']}.json"
        if row["run_record_sha256"]!=record_hashes[relative] or any(row[k]!=index[identity][k] for k in ["samples_sha256","checkpoint_sha256"]):
            raise ValueError("Oracle diagnostic artifacts changed")
    if observed!=set(index):
        raise ValueError("Oracle diagnostic run coverage incomplete")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    draws=sorted({r["data_seed"] for r in records})
    alphas=sorted({r["alpha"] for r in records})
    fig,axes=plt.subplots(1,len(draws),figsize=(6*len(draws),4.7),squeeze=False,sharey=True)
    for i,draw in enumerate(draws):
        ax=axes[0,i]
        for head in ["mlp","s4","whitened_mlp"]:
            values=[np.mean([r["metrics"]["velocity_mse"] for r in records if r["data_seed"]==draw and r["alpha"]==alpha and r["head"]==head]) for alpha in alphas]
            ax.plot(range(len(alphas)),values,marker="o",color=COLORS[head],label=HEAD_LABELS[head])
        noise=[next(c["bases"]["source_velocity_pca"]["expected_privileged_conditional_noise"]["overall_mean"] for c in report["cells"] if c["data_seed"]==draw and c["alpha"]==alpha) for alpha in alphas]
        ax.plot(range(len(alphas)),noise,marker="s",lw=1.6,ls="--",color="#182e3a",label="Privileged GP noise reference")
        ax.set_xticks(range(len(alphas)),[f"{a:g}" for a in alphas]);ax.set_xlabel("GP covariance weight α")
        ax.set_title(f"Independent dataset {i+1}");ax.grid(alpha=.15)
        if i==0:ax.set_ylabel("Raw velocity MSE per coordinate")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc="upper center",bbox_to_anchor=(.5,.94),frameon=False,fontsize=9)
    fig.suptitle("Velocity regression and a source-dependent noise reference",fontsize=14,y=.995)
    fig.text(.5,.008,"The analytic reference uses full raw history, richer information than the normalized encoder.\nIt is not a forecast score or a claim that the trained heads can attain this noise level.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.87));fig.savefig(output/"velocity_noise_reference.png",dpi=180);plt.close(fig)
    return {"artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),"completed":report["completed"],"cells":report["cells"]},["velocity_noise_reference.png"]


def analyze(results, output, training_gradients=None, oracle_diagnostics=None):
    root, out = Path(results), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cfg, manifest, rows, records, geometry, record_hashes = load_verified_rows(root)
    metrics = METRICS + ["fair_energy_score", "fixed_target_residual_mse", "fixed_target_pc1_mse", "coverage_90"]
    cells = summarize_cells(rows, metrics)
    contrasts = paired_contrasts(rows, metrics)
    summary = {"complete": True, "verified_runs": len(rows), "config": cfg, "data_seeds": cfg["data_seeds"], "seeds": cfg["seeds"],
               "manifest_fingerprint": manifest["fingerprint"], "config_sha256": manifest["config_sha256"],
               "record_sha256": record_hashes, "cells": cells, "contrasts": contrasts, "rows": rows,
               "uncertainty": "Training-seed SD/range within each independent dataset; equal-weight dataset means. Only two dataset draws; no population confidence interval claimed.",
               "target_pca_tail_definition": "Generated/target second moment around shared TRAIN TARGET mean, summed PCs7–16; 1024 paired evaluation contexts, one trajectory per context. No variance floor. Marginal geometry diagnostic, not conditional calibration.",
               "solver_sensitivity": [{k:r[k] for k in ["data_seed","alpha","head","seed","solver_sensitivity"]} for r in records if r["solver_sensitivity"]]}
    write_csv(out / "runs.csv", rows)
    flat_contrasts = []
    for contrast in contrasts:
        for draw in contrast["by_draw"]:
            for effect in draw["paired_seed_effects"]:
                flat_contrasts.append({k:v for k,v in contrast.items() if k != "by_draw"} | {"data_seed":draw["data_seed"],"dataset_mean_difference":draw["mean"]} | effect)
    write_csv(out / "contrasts.csv", flat_contrasts)
    summary["geometry"] = []
    for draw in cfg["data_seeds"]:
        for src in sorted(geometry[draw]["sources"].values(),key=lambda x:x["alpha"]):
            cov = np.asarray(src["covariance"])
            adjacent_variance = np.diag(cov)[:-1] + np.diag(cov)[1:] - 2*np.diag(cov,1)
            summary["geometry"].append({"data_seed":draw,"alpha":src["alpha"],
                 "velocity_top2_energy":src["validation_velocity"]["top2_energy"],"floored_directions":src["floored_directions"],
                 "theoretical_source_roughness":float(np.mean(np.sqrt(2/np.pi*adjacent_variance))),
                 "validation_source_roughness":src["validation_source"]["roughness"],
                 "validation_target_roughness":geometry[draw]["validation_target"]["roughness"]})
    summary["training_objective_gradients"], objective_figures = objective_gradient_report(training_gradients, records, manifest["fingerprint"], out, record_hashes)
    summary["privileged_oracle"], oracle_figures = oracle_report(oracle_diagnostics, records, manifest["fingerprint"], out, record_hashes)
    summary["figures"] = draw_figures(rows, cells, out) + additional_figures(records, geometry, out) + path_figures(root,cfg,records,out) + objective_figures + oracle_figures
    dump_json(out / "summary.json", summary)
    html_report(summary, out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/prior-mixture")
    parser.add_argument("--output", default="output/prior-mixture")
    parser.add_argument("--training-gradients", default="results/diagnostics/prior_mixture_training_gradients.json", help="Complete postprocessed objective-gradient report; empty string skips this report for software smoke checks")
    parser.add_argument("--oracle-diagnostics", default="results/diagnostics/prior_mixture_oracle.json", help="Complete privileged-oracle report; empty string skips it for software smoke checks")
    args = parser.parse_args()
    result = analyze(args.results, args.output, args.training_gradients or None, args.oracle_diagnostics or None)
    print(json.dumps({"complete":result["complete"],"verified_runs":result["verified_runs"],"report":str(Path(args.output)/"report.html")}))


if __name__ == "__main__":
    main()
