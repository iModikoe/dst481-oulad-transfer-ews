"""Download the two public study datasets and record provenance."""
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
sources = {
    'oulad': 'https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip',
    'uci': 'https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip',
}
records = []
for name, url in sources.items():
    folder = root / 'data' / name
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'source.zip'
    print(f'Downloading {name}: {url}', flush=True)
    urllib.request.urlretrieve(url, archive)
    files = []
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            filename = Path(member.filename).name
            if not filename.endswith('.csv'):
                continue
            target = folder / filename
            with z.open(member) as source, target.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            files.append({'file': str(target.relative_to(root)),
                          'sha256': hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()})
    records.append({'dataset': name, 'url': url, 'accessed_utc': datetime.now(timezone.utc).isoformat(),
                    'licence': 'CC BY 4.0',
                    'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
                    'files': files})
    print(f'Extracted {len(files)} CSV files for {name}', flush=True)
(root / 'data_sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
