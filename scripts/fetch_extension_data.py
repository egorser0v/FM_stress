"""Fetch pinned, public ETTh1/Exchange sources; cache data outside Git."""
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'ETTh1.csv': ('zhouhaoyi/ETDataset', '1d16c8f4f943005d613b5bc962e9eeb06058cf07', 'ETT-small/ETTh1.csv'),
    'exchange_rate.txt': ('laiguokun/multivariate-time-series-data', '7f402f185cc2435b5e66aed13a3b560ed142e023', 'exchange_rate/exchange_rate.txt.gz'),
}


def main():
    dest = ROOT/'data/real'; dest.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (repo, commit, remote) in FILES.items():
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/{remote}'
        data = urllib.request.urlopen(url, timeout=120).read()
        content = gzip.decompress(data) if remote.endswith('.gz') else data
        path = dest/name
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError(f'Refusing to replace changed local dataset: {path}')
        path.write_bytes(content)
        records[name] = {'url': url, 'revision': commit, 'download_sha256': hashlib.sha256(data).hexdigest(),
                         'content_sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        print(name, len(content), records[name]['content_sha256'], flush=True)
    (dest/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
