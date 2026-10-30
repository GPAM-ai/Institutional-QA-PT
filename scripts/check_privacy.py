"""Check the reviewed contact policy and extractive offsets; no claim of full anonymity."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
policy = json.loads((ROOT / 'reproducibility/privacy_policy.json').read_text())
allowed = set(policy['retained_institutional_emails'])
email = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
profile = re.compile(r'https?://lattes\.cnpq\.br/\d+')
phone = re.compile(r'(?:\+55\s*)?\(?\d{2}\)?[\s-]*\d{4,5}[\s-]\d{4}')
failures = []
for folder in ['data', 'results', 'reproducibility', 'docs']:
    for path in (ROOT / folder).rglob('*'):
        if not path.is_file() or path.suffix not in {'.json', '.txt', '.txt_', '.md'}:
            continue
        text = path.read_text()
        if any(m.group().lower() not in allowed for m in email.finditer(text)):
            failures.append(f'Unreviewed email: {path.relative_to(ROOT)}')
        if profile.search(text):
            failures.append(f'Individual profile URL: {path.relative_to(ROOT)}')
        if any(m.group() not in policy['retained_institutional_phones'] for m in phone.finditer(text)):
            failures.append(f'Unreviewed phone candidate: {path.relative_to(ROOT)}')

items = json.loads((ROOT / 'data/processed/corpus_completo.json').read_text())
for item in items:
    answers = item['answers']
    assert len(answers['text']) == len(answers['answer_start']), item['id']
    for answer, start in zip(answers['text'], answers['answer_start']):
        assert item['context'][start:start + len(answer)] == answer, item['id']
    if item['id'] == 'sobre_programa_coord_003':
        assert answers['text'] == [policy['student_label']], item['id']
coord = (ROOT / 'data/documents/sobre-o-programa/Coordenação.txt').read_text()
assert re.search(r'Representante discente\s+' + re.escape(policy['student_label']), coord)
if failures:
    raise SystemExit('\n'.join(failures))
print(f'PASS: reviewed contacts and {len(items)} QA answer offsets. Public identities remain; this is not an anonymity certification.')
