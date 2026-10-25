"""Read-only aggregation of archived per-PC results; no training or rescoring.

Separates PC3--6, PC7, and PC8--16 and retains the finite-sample cross term.
This arithmetic audit does not independently rerun the analytic oracle.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"pc1": [0], "pc2": [1], "pc3_6": list(range(2, 6)),
          "pc7": [6], "pc8_16": list(range(7, 16)), "pc3_16": list(range(2, 16))}


def group(values):
    return {name: mean(values[i] for i in ids) for name, ids in GROUPS.items()}


def main():
    inputs = sorted((ROOT / "results/main/runs").glob("gp_*.json"))
    oracle_path = ROOT / "results/followup-oracle/summary.json"
    inputs.append(oracle_path)
    spectra_path = ROOT / "results/diagnostics/report_addendum.json"
    inputs.append(spectra_path)
    records = [json.loads(p.read_text()) for p in inputs[:-2]]
    oracle = json.loads(oracle_path.read_text())
    spectra = json.loads(spectra_path.read_text())["spectra"]
    output = {"scope": "Archived per-PC arithmetic, no new model inference or training",
              "groups": {k: [i + 1 for i in v] for k, v in GROUPS.items()},
              "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in inputs}, "main": {}, "fresh_matched_oracle": {}}
    for head in ["mlp", "s4", "whitened_mlp"]:
        rows = [r for r in records if r["head"] == head]
        assert len(rows) == 5
        output["main"][head] = {"n_optimization_seeds": 5,
            "raw_mse": group([mean(r["metrics"]["pc_mse"][i] for r in rows) for i in range(16)])}
        rows = [r["matched"] for r in oracle["velocity_rows"]
                if r["source"] == "gp" and r["kind"] == head]
        assert len(rows) == 5
        item = {}
        for metric in ["label_mse", "oracle_discrepancy", "oracle_label_mse"]:
            item[metric] = group([mean(r[metric]["per_pc"][i] for r in rows) for i in range(16)])
        for metric in ["signed_cross_term_per_pc", "expected_noise_per_pc"]:
            item[metric] = group([mean(r[metric][i] for r in rows) for i in range(16)])
        item["max_archived_decomposition_error"] = max(r["decomposition_max_abs"] for r in rows)
        output["fresh_matched_oracle"][head] = item
    output["spectrum_check"] = {
        "target_pc7_eigenvalue": spectra["target"][6],
        "target_pc8_16_range": [min(spectra["target"][7:]), max(spectra["target"][7:])],
        "normalized_nugget_level": spectra["normalized_nugget_level"],
        "target_pc7_over_normalized_nugget": spectra["target"][6] / spectra["normalized_nugget_level"],
        "interpretation": "PC7 is a transition direction; PC8--16 are nugget-dominated. Floor membership is not absence of signal."}
    path = ROOT / "results/final-pca-audit/summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
