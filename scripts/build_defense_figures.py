"""Rebuild fixed-context scientific figures for the defense; no model execution.

Main: first GP-MLP seed, first fresh context, members 0--3.
Capacity: same first-context selection as analyze_followup_capacity.py.
Sources: exactly reconstruct the seed/covariances used by analyze_prior_mixture.py.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/final-presentation/figures"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
TARGET = "#172B3D"
INPUTS = {}
ENTRIES = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = (ROOT / path).read_bytes()
    INPUTS[str(path)] = digest(data)
    return data


def arrays(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A9B5BC")
    ax.tick_params(colors=TARGET, labelsize=13, length=3)
    ax.grid(axis="y", color="#DBE2E7", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def member_legend():
    return [Line2D([0], [0], color=c, lw=2, label=f"Draw {i+1}")
            for i, c in enumerate(COLORS)]


def save(fig, name):
    fig.savefig(OUT / name, dpi=100, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "axes.labelcolor": TARGET, "text.color": TARGET,
                         "axes.titleweight": "bold", "axes.titlesize": 16,
                         "axes.labelsize": 14, "svg.fonttype": "none"})
    cfg = json.loads(read("results/ensemble-evaluation/config.json"))
    seed = cfg["seeds"][0]
    common = arrays(read("results/ensemble-evaluation/common_inputs.npz"))
    forecasts = arrays(read(f"results/ensemble-evaluation/samples/gp_mlp_seed{seed}.npz"))["generated"]
    history, target, samples = common["history"][0], common["target"][0], forecasts[0, :4]
    assert history.shape == (32,) and target.shape == (16,) and samples.shape == (4, 16)
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    fig.subplots_adjust(left=.08, right=.985, bottom=.17, top=.78)
    ax.axvspan(-31, .5, facecolor="#F1F5F8", zorder=0)
    ax.axvline(.5, color="#7B8790", linestyle="--", linewidth=1.3)
    ax.plot(np.arange(-31, 1), history, color=TARGET, lw=2.5, zorder=4)
    for i, curve in enumerate(samples):
        ax.plot(np.arange(17), np.r_[history[-1], curve], color=COLORS[i], lw=1.8, alpha=.95)
    ax.plot(np.arange(17), np.r_[history[-1], target], color=TARGET, lw=2.7, linestyle="--", zorder=5)
    ax.set(xlim=(-31, 16.4), xlabel="Time relative to the forecast origin", ylabel="Normalized value")
    ax.set_xticks([-30, -20, -10, 0, 8, 16])
    ax.text(.28, 1.045, "Observed history: 32 points", transform=ax.transAxes, ha="center", fontsize=15, weight="bold")
    ax.text(.83, 1.045, "Future: 16 points", transform=ax.transAxes, ha="center", fontsize=15, weight="bold")
    style(ax)
    handles = [Line2D([0], [0], color=TARGET, lw=2.5, label="History"),
               Line2D([0], [0], color=TARGET, lw=2.5, ls="--", label="Observed future")]+member_legend()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.01),
               ncol=6, frameon=False, fontsize=12, handlelength=2, columnspacing=1.3)
    save(fig, "forecast_task.png")

    manifest = json.loads(read("artifacts/followup-transfer/manifest.json"))
    capcfg = json.loads(read("configs/followup_capacity.json"))
    draw = capcfg["replicates"][0]["data_seed"]
    budget = capcfg["budgets"][-1]
    heads = ["mlp_original", "mlp_wide", "s4"]
    selections = []
    transfer = ROOT / "artifacts/followup-transfer/forecast_samples.zip"
    archive_hash = hashlib.sha256()
    with transfer.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            archive_hash.update(chunk)
    INPUTS[str(transfer.relative_to(ROOT))] = archive_hash.hexdigest()
    records = []
    with zipfile.ZipFile(transfer) as archive:
        def entry(path):
            data = archive.read(path)
            spec = next(r for r in manifest["transferred_files"]
                        if r["archive"] == "forecast_samples.zip" and r["path"] == path)
            assert digest(data) == spec["sha256"]
            ENTRIES[path] = digest(data)
            return arrays(data)
        for head in heads:
            model = next(m for m in manifest["models"] if m["family"] == "followup-capacity"
                         and m["data_seed"] == draw and m["head"] == head and m["source"] == "white")
            assert model["completed_training_budget"] == budget
            sample = entry(model["forecast_transfer"]["path"])["ensemble"][0, :4]
            observed = entry(model["shared_targets_path"])["ensemble_target"][0]
            records.append((sample, observed))
            selections.append({"model_id": model["id"], "selected_step": model["selected_step"],
                               "original_sample_source": model["sample_source"],
                               "transferred_entry": model["forecast_transfer"]["path"]})
    assert all(np.array_equal(records[0][1], r[1]) for r in records)
    lower = min(min(x.min(), y.min()) for x, y in records)
    upper = max(max(x.max(), y.max()) for x, y in records)
    margin = .08*(upper-lower)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.77, wspace=.11)
    for ax, (paths, observed), title in zip(axes, records, ["MLP: width 56", "MLP: width 72", "S4"]):
        for i, curve in enumerate(paths):
            ax.plot(np.arange(1, 65), curve, color=COLORS[i], lw=1.25, alpha=.95)
        ax.plot(np.arange(1, 65), observed, color=TARGET, lw=2.6, ls="--", zorder=5)
        ax.set(title=title, xlim=(1, 64), ylim=(lower-margin, upper+margin), xlabel="Forecast step")
        ax.set_xticks([1, 16, 32, 48, 64]); style(ax)
    axes[0].set_ylabel("Normalized value")
    fig.legend(handles=[Line2D([0], [0], color=TARGET, lw=2.5, ls="--", label="Observed future")]+member_legend(),
               loc="upper center", bbox_to_anchor=(.53, 1.015), ncol=5, frameon=False, fontsize=12)
    save(fig, "capacity_paths.png")

    # Reconstruct the already published source illustration exactly, not new forecast samples.
    geometry = json.loads(read("results/prior-mixture/dataset_137041/geometry.json"))
    normals = np.random.default_rng(50371).standard_normal((4, 16))
    sources = [normals @ np.linalg.cholesky(np.asarray(geometry["sources"][k]["covariance"])).T
               for k in ("white", "gp")]
    lo, hi = min(s.min() for s in sources), max(s.max() for s in sources)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 2.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.25, top=.81, wspace=.12)
    for ax, sample, title in zip(axes, sources, ["Independent starting noise", "Smooth starting noise"]):
        for i, curve in enumerate(sample):
            ax.plot(np.arange(1, 17), curve, color=COLORS[i], lw=1.8)
        ax.set(title=title, xlabel="Future position", xlim=(1, 16), ylim=(lo-.15, hi+.15))
        ax.set_xticks([1, 4, 8, 12, 16]); style(ax)
    axes[0].set_ylabel("Source amplitude")
    save(fig, "source_curves.png")

    for path in ("scripts/build_defense_figures.py", "scripts/analyze_followup_capacity.py", "fm_stress/analyze_prior_mixture.py"):
        read(path)
    provenance = {
        "scope": "Scientific plot reformatting from frozen arrays; no training or forecast generation",
        "forecast_task.png": {"suite": "main GP MLP fresh ensemble evaluation", "seed": seed,
            "context_index_zero_based": 0, "member_indices_zero_based": [0, 1, 2, 3],
            "euler_steps": cfg["euler_steps"], "history_length": 32, "horizon": 16,
            "pixels": [1150, 380], "selection": "First configured seed, first context, first four draws; no visual selection"},
        "capacity_paths.png": {"kernel": "rbf", "source": "white", "data_seed": draw,
            "completed_training_budget": budget, "context_index_zero_based": 0,
            "member_indices_zero_based": [0, 1, 2, 3], "horizon": 64, "euler_steps": 64,
            "pixels": [1150, 380], "models": selections,
            "architectures": {h: capcfg["architectures"][h] for h in heads},
            "selection": "Same first configured draw/context/members as output/followup-capacity/rbf_example_paths.png",
            "axis_limits": [float(lower-margin), float(upper+margin)], "shared_y_axis": True},
        "source_curves.png": {"data_seed": 137041, "rng_seed": 50371,
            "rng": "numpy.random.default_rng", "normal_shape": [4, 16], "alphas": [0.0, 1.0],
            "pixels": [1150, 250], "selection": "Exact source reconstruction used in output/prior-mixture/source_examples.png",
            "formula": "normals @ cholesky(saved_covariance).T", "shared_y_axis": True,
            "caveat": "Source illustration only; GP covariance normalized to unit marginal variance as in mixture study"},
        "interpretation": "The first context is illustrative. One realized future and four sample paths do not establish conditional calibration or a distributional ranking.",
        "input_sha256": INPUTS, "zip_entry_sha256": ENTRIES,
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "output_sha256": {p.name: digest(p.read_bytes()) for p in [OUT/name for name in ("forecast_task.png", "capacity_paths.png", "source_curves.png")]}}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("Built three fixed-selection figures in", OUT)


if __name__ == "__main__":
    main()
