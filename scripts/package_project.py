"""Package the reproducible project without virtual environments or scratch files."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output" / "fm-stress-complete.zip"
DIRECTORIES = ("fm_stress", "configs", "tests", "vendor", "docs", "references", "scripts")
FILES = ("README.md", "pyproject.toml", "requirements-lock.txt", ".gitignore")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store"}


def eligible(path):
    return (path.is_file() and not any(p in EXCLUDE_PARTS or p.endswith('-preview') for p in path.parts)
            and not path.name.endswith('-preview.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true', help='Include the audited S4/MLP follow-up in a separate archive')
    args = parser.parse_args()
    archive_path = ROOT / 'output/fm-stress-extended.zip' if args.extended else ARCHIVE
    for required in ("output/pdf/research-report.pdf", "docs/conclusions.md",
                     "output/analysis/analysis_status.json", "output/analysis-lengthscale6/analysis_status.json"):
        if not (ROOT / required).is_file():
            raise RuntimeError(f"Missing final deliverable: {required}")
    paths = {ROOT / name for name in FILES}
    for directory in (*DIRECTORIES, "results", "output/analysis", "output/analysis-lengthscale6", "output/pdf"):
        paths.update(p for p in (ROOT / directory).rglob("*") if eligible(p))
    if args.extended:
        for required in ('output/pdf/s4-mlp-followup.pdf', 'docs/followup_conclusions.md',
                         'results/diagnostics/followup_verification.json'):
            if not (ROOT / required).is_file():
                raise RuntimeError(f'Missing audited follow-up artifact: {required}')
        audit = json.loads((ROOT / 'results/diagnostics/followup_verification.json').read_text())
        if audit.get('status') != 'passed' or audit.get('scope') != 'both':
            raise RuntimeError('Extended packaging requires a passed audit of both complete follow-up suites')
        for directory in ('output/field-diagnostics', 'output/controlled-followup', 'output/ensemble-analysis'):
            paths.update(p for p in (ROOT / directory).rglob('*') if eligible(p))
    # Preserve the editable proposal as context, without outdated PowerPoint exports.
    for name in ("flow-matching-poster.tex", "flow-matching-poster-LaTeX-A1.pdf"):
        p = ROOT / "output/poster" / name
        if p.exists():
            paths.add(p)
    manifest = {}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            if not eligible(path):
                continue
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            manifest[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            archive.writestr("fm-stress/" + relative, content)
        archive.writestr("fm-stress/FILE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None, "ZIP integrity check failed"
        for relative, record in manifest.items():
            content = archive.read("fm-stress/" + relative)
            assert hashlib.sha256(content).hexdigest() == record["sha256"], relative
    print(json.dumps({"archive": str(archive_path), "files": len(manifest),
                      "bytes": archive_path.stat().st_size, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
