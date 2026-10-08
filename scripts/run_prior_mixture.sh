#!/usr/bin/env bash
# Run from the repository root. Reuses only hash-compatible completed runs.
set -euo pipefail
.venv/bin/python -m fm_stress.prior_mixture --device mps
.venv/bin/python scripts/check_prior_mixture.py
.venv/bin/python scripts/check_prior_mixture_device.py
.venv/bin/python scripts/analyze_prior_training_gradients.py
.venv/bin/python scripts/analyze_prior_oracle.py
.venv/bin/python -m fm_stress.analyze_prior_mixture
.venv/bin/python scripts/check_prior_mixture_analysis.py
.venv/bin/python scripts/package_project.py --prior-mixture
