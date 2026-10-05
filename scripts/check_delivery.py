"""Fail visibly if a submitted experimental bundle is incomplete or mismatched."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.analyze import collect
from fm_stress.experiment import config_hash
from fm_stress.metrics import PCA, evaluate


def compare_metrics(actual, expected, location):
    assert set(actual) == set(expected), location
    for key, value in actual.items():
        where = f"{location}/{key}"
        if isinstance(value, dict):
            compare_metrics(value, expected[key], where)
        elif value is None or isinstance(value, bool):
            assert value == expected[key], where
        else:
            np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-12, err_msg=where)

def main():
    checks = {}
    for name, expected in (("main",25),("lengthscale6",15)):
        config, records, groups, status = collect(ROOT/"results"/name)
        assert status["complete"], status
        assert len(records) == expected
        for record in records:
            assert record["device"] == "mps", record["run"]
            checkpoint_path = ROOT/"results"/name/"checkpoints"/(record["run"]+".pt")
            assert checkpoint_path.is_file()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            assert config_hash(checkpoint["config"]) == config_hash(config) == record["config_sha256"]
            assert checkpoint["step"] == record["best_step"]
            assert checkpoint["seed"] == record["seed"]
            assert checkpoint["kind"] == record["head"]
            assert checkpoint["source"] == record["source"]
            assert record["best_step"] <= config["steps"]
            best = min(record["curve"], key=lambda row: row["validation_objective"])
            assert best["step"] == record["best_step"]
            assert best["validation_objective"] == record["best_validation_objective"]
            arrays = np.load(ROOT/"results"/name/"samples"/(record["run"]+".npz"))
            for key in arrays.files:
                assert np.isfinite(arrays[key]).all(), (record["run"], key)
            horizon = config["data"]["horizon"]
            assert arrays["prediction"].shape == (config["n_test"], horizon)
            assert arrays["generated"].shape == (config["n_paths"], horizon)
            transform = checkpoint["pca"]
            pca = PCA(*(np.asarray(transform[key]) for key in ("mean", "components", "eigenvalues")),
                      float(transform["floor"]))
            recomputed = evaluate(arrays["prediction"], arrays["velocity"], arrays["generated"], arrays["target"], pca)
            compare_metrics(recomputed, record["metrics"], record["run"])
            arrays.close()
        checks[name] = {"runs":len(records),"all_mps":True,"complete":True,
                        "all_metrics_recomputed_from_saved_arrays":True,
                        "validation_checkpoint_selection_verified":True}
        manifest=json.loads((ROOT/"results"/name/"source_manifest.json").read_text())
        for filename, expected_hash in manifest["files"].items():
            assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected_hash, filename
    for filename in ("docs/conclusions.md", "docs/related_work.md", "docs/protocol.md",
                     "docs/model_provenance.md", "docs/defense_notes.md",
                     "results/main/robustness/solver_seed0.json", "output/pdf/related-work.pdf",
                     "output/pdf/research-report.pdf", "docs/final_audit.md"):
        assert (ROOT/filename).is_file(), filename
    sensitivity = ROOT/"results/main/optimization_sensitivity"
    extra_records = [json.loads(p.read_text()) for p in sorted((sensitivity/"runs").glob("*.json"))]
    assert len(extra_records) in (3, 9), "Incomplete bounded optimization sensitivity"
    for record in extra_records:
        assert record["device"] == "mps" and record["status"] == "complete"
        original = Path(record["source_checkpoint"])
        original = original if original.is_absolute() else ROOT/original
        assert hashlib.sha256(original.read_bytes()).hexdigest() == record["options"]["initial_checkpoint_sha256"]
        best = min(record["curve"], key=lambda row: row["validation_objective"])
        assert best["additional_step"] == record["best_additional_step"]
        assert best["validation_objective"] == record["best_validation_objective"]
        checkpoint = torch.load(sensitivity/"checkpoints"/(record["run"]+".pt"), map_location="cpu", weights_only=False)
        transform = checkpoint["pca"]
        pca = PCA(*(np.asarray(transform[key]) for key in ("mean", "components", "eigenvalues")), float(transform["floor"]))
        with np.load(sensitivity/"samples"/(record["run"]+".npz")) as arrays:
            compare_metrics(evaluate(arrays["prediction"], arrays["velocity"], arrays["generated"], arrays["target"], pca),
                            record["metrics"], "warm_restart/"+record["run"])
    checks["optimization_sensitivity"] = {"runs":len(extra_records),"all_mps":True,
                                          "original_checkpoints_verified":True,
                                          "all_metrics_recomputed_from_saved_arrays":True}
    for directory in ("output/analysis", "output/analysis-lengthscale6"):
        assert json.loads((ROOT/directory/"analysis_status.json").read_text())["complete"]
    report = json.loads((ROOT/"output/pdf/research-report.manifest.json").read_text())
    assert report["runs"] == 25 and report["lengthscale_runs"] == 15 and report["optimization_runs"] == 9
    assert report["conclusions_sha256"] == hashlib.sha256((ROOT/"docs/conclusions.md").read_bytes()).hexdigest()
    import pdfplumber
    for filename, pages in (("related-work.pdf",1),("research-report.pdf",report["pages"])):
        with pdfplumber.open(ROOT/"output/pdf"/filename) as document:
            assert len(document.pages) == pages
            assert all(len(page.extract_text() or '') > 150 for page in document.pages)
    checks["reports"] = {"one_page_synthesis":True,"report_pages":report["pages"],
                         "conclusions_match_export":True,"no_empty_pages":True}
    (ROOT/"results/delivery_verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks,indent=2))

if __name__ == "__main__":
    main()
