"""Refresh delivery checksums after intentional, reviewed artifact changes."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
folders=['scripts','reproducibility','figures','data','results','docs','environment','.github']
files=[ROOT/n for n in ['README.md','requirements.txt','preprocess_docs.py','.gitignore']]
for folder in folders:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'})
manifest={}
for p in sorted(files):
    raw=p.read_bytes()
    if p.suffix in {'.py','.sh','.md','.json','.txt','.yml','.tex'}:
        if re.search(rb'hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',raw):
            raise SystemExit(f'Embedded credential detected in {p.relative_to(ROOT)}')
    manifest[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
(ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Refreshed manifest for {len(manifest)} delivery files.')
