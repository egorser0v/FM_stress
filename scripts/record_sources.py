"""Record numerical source hashes; reject changed code when reusing a run folder."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ('fm_stress/experiment.py', 'fm_stress/models.py', 'fm_stress/data.py', 'fm_stress/metrics.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--note', default='Recorded before invoking the experiment runner.')
    args = p.parse_args()
    files = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    manifest = args.output/'source_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['files'] != files:
            raise RuntimeError('Numerical source changed; use a new output directory.')
        print(f'Source hashes verified: {manifest}')
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'recorded_utc':datetime.now(timezone.utc).isoformat(),
                                   'note':args.note, 'files':files}, indent=2)+'\n')
    print(manifest)


if __name__ == '__main__':
    main()
