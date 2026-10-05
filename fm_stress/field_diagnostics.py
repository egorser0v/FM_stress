"""Descriptive, post hoc field-error diagnostics from immutable saved arrays.

No training or checkpoint selection occurs here. PC7..16 is a fixed descriptive
tail, not a prospectively selected success metric or proof of high frequency.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import paired_bootstrap_difference

BINS = np.array([0., .1, .25, .5, .75, .9, 1.])


def summarize_errors(error, time, axes):
    """Uncentered squared errors; PCA basis is fitted to training velocities."""
    error = np.asarray(error, dtype=np.float64)
    time = np.asarray(time).reshape(-1)
    projected = error @ np.asarray(axes, dtype=np.float64).T
    pc = np.mean(projected ** 2, axis=0)
    temporal = np.mean(error ** 2, axis=0)
    if not np.allclose(pc.mean(), temporal.mean(), rtol=1e-10, atol=1e-12):
        raise ValueError("PCA error decomposition does not preserve total MSE")
    bins = []
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        mask = (time >= lo) & ((time < hi) if i < len(BINS)-2 else (time <= hi))
        if not mask.any():
            raise ValueError("Empty time bin")
        e = projected[mask]
        bins.append({"left": float(lo), "right": float(hi), "n": int(mask.sum()),
                     "pc1_mse": float(np.mean(e[:, 0] ** 2)),
                     "residual_mse": float(np.mean(e[:, 2:] ** 2)),
                     "tail_pc7_to_f_mse": float(np.mean(e[:, 6:] ** 2))})
    return {"pc_mse": pc.tolist(), "position_mse": temporal.tolist(),
            "tail_pc7_to_f_mse": float(pc[6:].mean()), "time_bins": bins}


def analyze(root, label):
    geometry = json.loads((root / "geometry.json").read_text())
    records = []
    for path in sorted((root / "runs").glob("*.json")):
        original = json.loads(path.read_text())
        source, head, seed = original["source"], original["head"], original["seed"]
        arrays = np.load(root / "samples" / (path.stem + ".npz"))
        axes = np.array(geometry["pca"][source]["components"])
        diagnostic = summarize_errors(arrays["prediction"] - arrays["velocity"], arrays["time"], axes)
        np.testing.assert_allclose(diagnostic["pc_mse"], original["metrics"]["pc_mse"], rtol=1e-10)
        # Identical TRAIN TARGET basis/centering for both sample distributions.
        # Second moments include bias; these are not conditional variances.
        target_pca = geometry["pca"]["target"]
        target_axes, target_mean = np.asarray(target_pca["components"]), np.asarray(target_pca["mean"])
        generated_pc = (arrays["generated"] - target_mean) @ target_axes.T
        observed_pc = (arrays["target"] - target_mean) @ target_axes.T
        generated_moment, target_moment = np.mean(generated_pc**2, axis=0), np.mean(observed_pc**2, axis=0)
        diagnostic["sample_target_pc_generated_second_moment"] = generated_moment.tolist()
        diagnostic["sample_target_pc_observed_second_moment"] = target_moment.tolist()
        diagnostic["sample_target_pc_tail_ratio"] = float(generated_moment[6:].mean()/target_moment[6:].mean())
        records.append({"geometry": label, "source": source, "head": head,
                        "seed": seed, "run": path.stem, **diagnostic})
    paired = []
    for source in ("white", "gp"):
        for alternative in (("s4",) if source == "white" else ("s4", "whitened_mlp")):
            left = {x["seed"]: x for x in records if x["source"] == source and x["head"] == alternative}
            right = {x["seed"]: x for x in records if x["source"] == source and x["head"] == "mlp"}
            seeds = sorted(left.keys() & right.keys())
            if len(seeds) < 2:
                continue
            selectors = [("tail_pc7_to_f_mse", lambda x: x["tail_pc7_to_f_mse"])]
            selectors += [(f"pc{j+1}_mse", lambda x, j=j: x["pc_mse"][j]) for j in range(len(axes))]
            selectors += [(f"timebin{i+1}_residual_mse", lambda x, i=i: x["time_bins"][i]["residual_mse"])
                          for i in range(len(BINS)-1)]
            for metric, select in selectors:
                a, b = [select(left[s]) for s in seeds], [select(right[s]) for s in seeds]
                interval = paired_bootstrap_difference(a, b, repetitions=10000, seed=141)
                paired.append({"geometry": label, "source": source, "comparison": alternative + " minus mlp",
                               "metric": metric, **interval})
    return records, paired


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/field-diagnostics")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, paired = [], []
    for root, label in ((Path("results/main"), "lengthscale8"), (Path("results/lengthscale6"), "lengthscale6")):
        rows, comparisons = analyze(root, label)
        records.extend(rows)
        paired.extend(comparisons)
    metadata = {"status": "post hoc descriptive diagnostics", "time_bins": BINS.tolist(),
                "tail_definition": "PCs 7 through F; fixed index subset, not necessarily high frequency",
                "interval_scope": "Paired optimization seeds on fixed data; descriptive pointwise 95% percentile intervals, no multiplicity correction",
                "records": records, "paired_differences": paired}
    (output / "diagnostics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_csv(output / "paired_differences.csv", paired)
    long_rows = []
    bin_rows = []
    sample_rows = []
    for record in records:
        identity = {k: record[k] for k in ("geometry", "source", "head", "seed")}
        for i, (pc, position) in enumerate(zip(record["pc_mse"], record["position_mse"]), 1):
            long_rows.append({**identity, "index_1based": i, "pc_mse": pc, "position_mse": position})
        bin_rows.extend({**identity, **b} for b in record["time_bins"])
        for i, (generated, target) in enumerate(zip(record["sample_target_pc_generated_second_moment"],
                                                     record["sample_target_pc_observed_second_moment"]), 1):
            sample_rows.append({**identity, "target_pc_1based": i,
                               "generated_second_moment": generated, "observed_second_moment": target})
    write_csv(output / "pc_and_position_mse.csv", long_rows)
    write_csv(output / "time_bins.csv", bin_rows)
    write_csv(output / "sample_target_pc_second_moments.csv", sample_rows)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {"mlp": "#24758b", "s4": "#c26e3b", "whitened_mlp": "#667f35"}
    for row, geometry in enumerate(("lengthscale8", "lengthscale6")):
        for head in ("mlp", "s4", "whitened_mlp"):
            subset = [r for r in records if r["geometry"] == geometry and r["source"] == "gp" and r["head"] == head]
            label = {"mlp": "MLP", "s4": "S4", "whitened_mlp": "Whitened MLP"}[head]
            for column, values, x in ((0, [r["pc_mse"] for r in subset], np.arange(1, 17)),
                                      (1, [[b["residual_mse"] for b in r["time_bins"]] for r in subset], (BINS[:-1]+BINS[1:])/2),
                                      (2, [r["position_mse"] for r in subset], np.arange(1, 17))):
                values = np.array(values)
                ax = axes[row, column]
                ax.plot(x, values.mean(0), color=colors[head], label=label, marker="o", ms=3)
                ax.fill_between(x, values.min(0), values.max(0), color=colors[head], alpha=.10)
        axes[row, 0].set_yscale("log")
        for column, (title, xlabel) in enumerate((("Error across velocity PCs", "Principal component"),
                                                 ("Residual error across flow time", "Time-bin midpoint"),
                                                 ("Error across forecast positions", "Forecast index"))):
            axes[row, column].set(title=title, xlabel=xlabel, ylabel="Raw velocity MSE")
            axes[row, column].grid(alpha=.15)
        axes[row, 0].text(.03, .04, "GP source · lengthscale " + geometry.removeprefix("lengthscale"),
                          transform=axes[row, 0].transAxes, fontsize=10)
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Saved-model field diagnostics · line = seed mean; band = seed range", fontsize=15)
    fig.savefig(output / "field_diagnostics.png", dpi=180)
    fig.savefig(output / "field_diagnostics.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, geometry in zip(axes, ("lengthscale8", "lengthscale6")):
        for head in ("mlp", "s4", "whitened_mlp"):
            subset = [r for r in records if r["geometry"] == geometry and r["source"] == "gp" and r["head"] == head]
            values = np.array([r["sample_target_pc_generated_second_moment"] for r in subset])
            label = {"mlp": "MLP", "s4": "S4", "whitened_mlp": "Whitened MLP"}[head]
            x = np.arange(1, values.shape[-1]+1)
            ax.plot(x, values.mean(0), color=colors[head], label=label, marker="o", ms=3)
            ax.fill_between(x, values.min(0), values.max(0), color=colors[head], alpha=.10)
        observed = subset[0]["sample_target_pc_observed_second_moment"]
        ax.plot(x, observed, color="#202b32", label="Observed targets", linestyle="--", linewidth=2)
        ax.set(yscale="log", xlabel="Training-target principal component", ylabel="Second moment around training mean",
               title="GP source · lengthscale " + geometry.removeprefix("lengthscale"))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    fig.suptitle("Generated sample energy · fixed train-target basis · 1,024 paired histories", fontsize=14)
    fig.savefig(output / "sample_target_pc_energy.png", dpi=180)
    fig.savefig(output / "sample_target_pc_energy.pdf")
    plt.close(fig)
    print(json.dumps({"runs": len(records), "paired_comparisons": len(paired), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
