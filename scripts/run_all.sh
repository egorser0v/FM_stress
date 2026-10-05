#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${FM_PYTHON:-.venv/bin/python}"
DEVICE="${FM_DEVICE:-mps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fmstress-mpl}"

"$PYTHON" -m pytest -q
"$PYTHON" -m fm_stress.geometry_scan --config configs/main.json --output results/geometry-sensitivity --seed 37041
"$PYTHON" scripts/record_sources.py results/main
"$PYTHON" -m fm_stress.experiment geometry --config configs/main.json --output results/main
"$PYTHON" -m fm_stress.experiment train --config configs/main.json --output results/main --device "$DEVICE"
"$PYTHON" -m fm_stress.experiment oracle --config configs/main.json --output results/main
"$PYTHON" -m fm_stress.analyze --results results/main --output output/analysis --require-complete
"$PYTHON" -m fm_stress.optimization_check --config configs/main.json --results results/main --device "$DEVICE" --seeds 0
if "$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path('results/main/optimization_sensitivity/runs')
m = json.loads((p/'gp_mlp_seed0.json').read_text())
s = json.loads((p/'gp_s4_seed0.json').read_text())
before = s['initial_validation_raw_mse'] - m['initial_validation_raw_mse']
after = s['best_validation_raw_mse'] - m['best_validation_raw_mse']
expand = s['validation_improvement_fraction'] > .05 or before*after < 0
decision = {'decision': 'expand to seeds 1 and 2' if expand else 'stop after seed 0',
            'selection_information': 'validation only; no test metric used',
            'rule_source': 'docs/optimization_sensitivity.md',
            's4_validation_improvement_fraction': s['validation_improvement_fraction'],
            'ordering_reversed': before*after < 0, 'next_seeds': [1,2] if expand else []}
(p.parent/'expansion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
raise SystemExit(0 if expand else 1)
PY
then
    "$PYTHON" -m fm_stress.optimization_check --config configs/main.json --results results/main --device "$DEVICE" --seeds 1,2
fi
"$PYTHON" -m fm_stress.robustness --config configs/main.json --results results/main --device "$DEVICE" --seed 0 --n-paths 256 --steps 32,64,128,256

# Declared shorter-lengthscale repair/sensitivity; never pooled with main results.
"$PYTHON" scripts/record_sources.py results/lengthscale6
"$PYTHON" -m fm_stress.experiment geometry --config configs/lengthscale6.json --output results/lengthscale6
"$PYTHON" -m fm_stress.experiment train --config configs/lengthscale6.json --output results/lengthscale6 --device "$DEVICE"
"$PYTHON" -m fm_stress.experiment oracle --config configs/lengthscale6.json --output results/lengthscale6
"$PYTHON" -m fm_stress.analyze --results results/lengthscale6 --output output/analysis-lengthscale6 --require-complete
"$PYTHON" scripts/render_related_work.py
"$PYTHON" scripts/build_report.py
