"""Verify file integrity and reproduce the distributed metrics without third-party packages."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())
def check_files():
    for name,expected in manifest.items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise SystemExit(f'Artifact checksum mismatch: {name}')
check_files()
paper_figures=json.loads((ROOT/'reproducibility/paper_figure_manifest.json').read_text())
for name,expected in paper_figures.items():
    assert hashlib.sha256((ROOT/'figures/paper'/name).read_bytes()).hexdigest()==expected, name
for script in ['scripts/check_privacy.py','reproducibility/audit_results.py','scripts/summarize_results.py']:
    p=subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,text=True)
    if p.returncode:
        print(p.stdout)
        raise SystemExit(f'Reproduction failed: {script}')
check_files()
d=json.loads((ROOT/'reproducibility/verified_results.json').read_text())
assert len(d['extractive_best_validation'])==15
assert all(r['metric_agrees'] for r in d['prediction_audit'].values())
assert all(r['n_examples']==408 and r['n_questions']==103 for r in d['prediction_audit'].values())
assert not d['split']['independent_test']
assert not (ROOT/'paper_BDCAT').exists()
print(f'PASS: {len(manifest)} file checksums; split, all saved-prediction metrics, bootstrap and generated tables reproduced.')
