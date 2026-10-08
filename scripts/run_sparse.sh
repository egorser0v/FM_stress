#!/usr/bin/env bash
# Run from the repository root; completed configurations resume after hash checks.
set -euo pipefail
.venv/bin/python scripts/fetch_extension_data.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_sparse_model_device.py
.venv/bin/python -m fm_stress.sparse_baselines_qr --config configs/sparse_extension.json --output results/sparse-extension
.venv/bin/python -m fm_stress.sparse_experiment --device mps
.venv/bin/python -m fm_stress.sparse_baselines_qr --config configs/sparse_fullrank_dictionary.json --output results/sparse-fullrank-dictionary
.venv/bin/python -m fm_stress.sparse_experiment --config configs/sparse_fullrank_dictionary.json --output results/sparse-fullrank-dictionary --device mps
.venv/bin/python scripts/check_sparse_extension.py
.venv/bin/python scripts/check_sparse_extension.py --results results/sparse-fullrank-dictionary --config configs/sparse_fullrank_dictionary.json
.venv/bin/python scripts/check_dictionary_span.py
.venv/bin/python scripts/check_dictionary_span.py --results results/sparse-fullrank-dictionary --output results/diagnostics/dictionary_span_fullrank.json
.venv/bin/python -m fm_stress.analyze_sparse
.venv/bin/python scripts/check_sparse_analysis.py
.venv/bin/python -m fm_stress.analyze_dictionary_supplement --rank-diagnostics results/diagnostics/dictionary_span.json --rank-diagnostics results/diagnostics/dictionary_span_fullrank.json
.venv/bin/python scripts/check_dictionary_analysis.py
.venv/bin/python scripts/build_sparse_report.py
.venv/bin/python scripts/package_project.py --sparse
