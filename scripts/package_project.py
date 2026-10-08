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
            # This describes the archive being written; including it would be self-referential.
            and path.name != 'sparse_package_verification.json'
            and not path.name.endswith('-preview.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true', help='Include the audited S4/MLP follow-up in a separate archive')
    parser.add_argument('--multivariate', action='store_true', help='Include the audited multivariate/real-data extension in a new archive')
    parser.add_argument('--sparse', action='store_true', help='Include the audited sparse/whitening follow-up in a new archive')
    args = parser.parse_args()
    if args.sparse:
        args.multivariate = True
    archive_path = (ROOT / 'output/fm-stress-sparse-whitening.zip' if args.sparse else
                    ROOT / 'output/fm-stress-multivariate.zip' if args.multivariate else
                    ROOT / 'output/fm-stress-extended.zip' if args.extended else ARCHIVE)
    for required in ("output/pdf/research-report.pdf", "docs/conclusions.md",
                     "output/analysis/analysis_status.json", "output/analysis-lengthscale6/analysis_status.json"):
        if not (ROOT / required).is_file():
            raise RuntimeError(f"Missing final deliverable: {required}")
    paths = {ROOT / name for name in FILES}
    for directory in (*DIRECTORIES, "results", "output/analysis", "output/analysis-lengthscale6", "output/pdf"):
        paths.update(p for p in (ROOT / directory).rglob("*") if eligible(p))
    if args.extended or args.multivariate:
        for required in ('output/pdf/s4-mlp-followup.pdf', 'docs/followup_conclusions.md',
                         'results/diagnostics/followup_verification.json'):
            if not (ROOT / required).is_file():
                raise RuntimeError(f'Missing audited follow-up artifact: {required}')
        audit = json.loads((ROOT / 'results/diagnostics/followup_verification.json').read_text())
        if audit.get('status') != 'passed' or audit.get('scope') != 'both':
            raise RuntimeError('Extended packaging requires a passed audit of both complete follow-up suites')
        for directory in ('output/field-diagnostics', 'output/controlled-followup', 'output/ensemble-analysis'):
            paths.update(p for p in (ROOT / directory).rglob('*') if eligible(p))
    if args.multivariate:
        audit = json.loads((ROOT / 'results/extension/independent_verification.json').read_text())
        summary = json.loads((ROOT / 'output/extension-analysis/summary.json').read_text())
        if not audit.get('complete') or summary.get('status') != 'complete' or summary.get('observed_runs') != 96:
            raise RuntimeError('Multivariate packaging requires all 96 runs and a complete independent audit')
        if audit.get('verified_runs') != 96 or audit.get('fingerprint') != summary.get('fingerprint'):
            raise RuntimeError('Multivariate audit and analysis must refer to the same complete experiment')
        analysis_audit = json.loads((ROOT/'results/diagnostics/extension_analysis_verification.json').read_text())
        if analysis_audit.get('status') != 'passed' or analysis_audit.get('summary_sha256') != hashlib.sha256((ROOT/'output/extension-analysis/summary.json').read_bytes()).hexdigest():
            raise RuntimeError('Summary must have a matching independent analysis audit')
        for required in ('output/pdf/multivariate-real-extension.pdf','docs/extension_findings.md','data/real/manifest.json'):
            p = ROOT / required
            if not p.is_file(): raise RuntimeError(f'Missing extension deliverable: {required}')
            paths.add(p)
        paths.update(p for p in (ROOT/'output/extension-analysis').rglob('*') if eligible(p))
    if args.sparse:
        sparse_root = ROOT/'results/sparse-extension'
        summary_path = ROOT/'output/sparse-analysis/summary.json'
        audit = json.loads((sparse_root/'independent_verification.json').read_text())
        summary = json.loads(summary_path.read_text())
        verification = json.loads((ROOT/'results/diagnostics/sparse_analysis_verification.json').read_text())
        if not audit.get('complete') or audit.get('verified_runs') != 264 or summary.get('status') != 'complete':
            raise RuntimeError('Sparse packaging requires all 264 fits and a complete independent audit')
        if verification.get('status') != 'passed' or verification.get('summary_sha256') != hashlib.sha256(summary_path.read_bytes()).hexdigest():
            raise RuntimeError('Sparse analysis lacks matching independent verification')
        for required in ('output/pdf/sparse-whitening-extension.pdf','docs/sparse_findings.md'):
            if not (ROOT/required).is_file():raise RuntimeError('Missing sparse artifact '+required)
        paths.update(p for p in (ROOT/'output/sparse-analysis').rglob('*') if eligible(p))
        supplement = json.loads((ROOT/'results/sparse-fullrank-dictionary/independent_verification.json').read_text())
        if not supplement.get('complete') or supplement.get('verified_runs') != 24:
            raise RuntimeError('The 24-fit dictionary supplement must also pass independent verification')
        ds_path = ROOT/'output/dictionary-supplement/summary.json'
        da = json.loads((ROOT/'results/diagnostics/dictionary_analysis_verification.json').read_text())
        if da.get('status') != 'passed' or da.get('summary_sha256') != hashlib.sha256(ds_path.read_bytes()).hexdigest():
            raise RuntimeError('Dictionary supplement analysis lacks matching verification')
        paths.update(p for p in (ROOT/'output/dictionary-supplement').rglob('*') if eligible(p))
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
