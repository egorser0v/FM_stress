#!/usr/bin/env python3
"""Reproduce the deterministic final label-study source/stream hash audit.

The default audit file is part of downstream provenance. An existing file must
already equal the newly reconstructed bytes; it is never silently replaced.
Use --output with a new path to demonstrate regeneration from scratch.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.experiment import config_hash
from fm_stress.followup_label_noise import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new audit destination.')
    args = parser.parse_args()
    source = ROOT / 'results/followup-label-noise'
    manifest = json.loads((source / 'manifest.json').read_text())
    config = json.loads((source / 'config.json').read_text())
    if config_hash(config) != manifest['config_sha256']:
        raise AssertionError('Frozen configuration hash changed.')
    for relative, expected in manifest['sources'].items():
        if sha(ROOT / relative) != expected:
            raise AssertionError(f'Frozen source changed: {relative}')

    groups = []
    for path in sorted(source.glob('dataset_*/*/prepared.json')):
        prepared = json.loads(path.read_text())
        if prepared['fingerprint'] != manifest['fingerprint']:
            raise AssertionError(f'Preparation fingerprint mismatch: {path}')
        for filename, expected in prepared['files'].items():
            if sha(path.parent / filename) != expected:
                raise AssertionError(f'Prepared artifact changed: {path.parent / filename}')
        groups.append({
            'group': str(path.parent.relative_to(ROOT)),
            'verified_files': len(prepared['files']),
        })
    expected_groups = len(config['replicates']) * len(config['sources'])
    if len(groups) != expected_groups:
        raise AssertionError(f'Expected {expected_groups} prepared groups; found {len(groups)}.')

    # Key order and serialization match the original final audit exactly.
    payload = {
        'complete': len(groups) == expected_groups,
        'status': 'passed',
        'source_files': len(manifest['sources']),
        'groups': groups,
        'fingerprint': manifest['fingerprint'],
    }
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode('utf-8')
    output = args.output or source / 'final_hash_verification.json'
    existed = output.exists()
    if existed:
        if output.read_bytes() != encoded:
            raise AssertionError('Existing audit differs from deterministic reconstruction; file preserved.')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    print(json.dumps({
        'status': 'passed',
        'path': str(output),
        'sha256': hashlib.sha256(encoded).hexdigest(),
        'existing_bytes_unchanged': existed,
        'verified_groups': len(groups),
    }, indent=2))


if __name__ == '__main__':
    main()
