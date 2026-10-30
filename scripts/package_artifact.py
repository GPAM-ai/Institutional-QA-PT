"""Verify and package only manifest-listed files, excluding Git history and the paper."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT.parent/'Institutional-QA-PT.zip')
args=parser.parse_args()
subprocess.run([sys.executable,str(ROOT/'scripts/verify_artifact.py')],check=True)
manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())
names=sorted([*manifest,'ARTIFACT_MANIFEST.json'])
with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as z:
    for name in names:
        assert '.git/' not in name and 'paper_BDCAT/' not in name
        z.write(ROOT/name,'Institutional-QA-PT/'+name)
print(f'{args.output}\nSHA-256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}')
