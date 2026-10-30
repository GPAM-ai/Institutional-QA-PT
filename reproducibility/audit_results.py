"""Reproduce camera-ready audit using only Python's standard library.

Run from any directory: python reproducibility/audit_results.py
No model loading, training, or modification of original results.
"""
import ast
import collections
import hashlib
import json
import random
import re
import statistics
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text())

# Execute the actual pure metric functions, without importing torch/transformers.
tree = ast.parse((ROOT / 'scripts/train_unifesp_qa.py').read_text())
names = {'normalize_answer', 'compute_exact', 'compute_f1'}
ns = dict(re=re, string=string, collections=collections)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<repository metrics>', 'exec'), ns)

items = read(ROOT / 'data/processed/corpus_completo.json')
random.Random(42).shuffle(items)
cut = int(.8 * len(items))
train, held = items[:cut], items[cut:]
expected = {}
for x in held:
    gold = next(iter(x['answers']['text']), '')
    expected[x['id']] = (x['id'], x['question'], gold)
    for i, q in enumerate(x.get('question_variants', [])):
        if q.strip():
            expected[f"{x['id']}_var{i}"] = (x['id'], q, gold)
assert len(expected) == 408 and len(held) == 103
(OUT/'split_manifest.json').write_text(json.dumps(dict(seed=42, training_ids=[x['id'] for x in train], holdout_ids=[x['id'] for x in held], holdout_formulation_ids=list(expected)), indent=2)+'\n')
assert not {x['id'] for x in train} & {x['id'] for x in held}

inventory = []
for p in sorted((ROOT/'results').glob('*/metrics.json')):
    d = read(p)
    if 'validation_f1' in d and 'dataset' not in p.parent.name:
        inventory.append(dict(run=p.parent.name, **d))
best = []
for m in ['mbert', 'xlmr', 'bertugues', 'bertimbau_base', 'bertimbau_large']:
    for t in ['full', 'lora', 'qlora']:
        rows = [x for x in inventory if x.get('model_type') == m and x.get('tuning_type') == t and 'dataset' not in x['run']]
        if rows:
            d = max(rows, key=lambda x: x['validation_f1'])
            best.append(dict(d, reported_memory_increment_gib=d['peak_vram_mb']/1024))

runs = [
 ('phi_gold', 'phi_4_mini_rag_sft_gold', 'predictions.json', 'llm_response'),
 ('phi_retrieved', 'phi_4_mini_rag_sft_hybrid_graph', 'predictions.json', 'llm_response'),
 ('qwen_guided', 'hybrid_sft_qwen15b_truth', 'hybrid_predictions.json', 'selected_response'),
 ('qwen_gated8', 'hybrid_gated_sft_t8', 'hybrid_predictions.json', 'selected_response'),
 ('qwen_zs_guided', 'hybrid_rag_qwen_always_baseline', 'hybrid_predictions.json', 'selected_response'),
 ('retrieved_extractor', 'hybrid_gated_sft_t8', 'hybrid_predictions.json', 'bert_span'),
 ('qwen_verify', 'hybrid_verify_always', 'hybrid_predictions.json', 'selected_response'),
 ('qwen_guarded', 'hybrid_verify_evidence', 'hybrid_predictions.json', 'selected_response'),
 ('qwen_verdict', 'hybrid_llm_verify', 'hybrid_predictions.json', 'selected_response'),
 ('llama3b', 'llama_3b_rag', 'predictions.json', 'llm_response'),
 ('qwen15b', 'qwen_1.5b_rag', 'predictions.json', 'llm_response'),
 ('smollm2', 'smollm2_rag', 'predictions.json', 'llm_response'),
 ('llama1b', 'llama_1b_rag', 'predictions.json', 'llm_response'),
 ('tucano', 'tucano_rag', 'predictions.json', 'llm_response'),
 ('qwen_dense_heavy', 'hybrid_sft_truth_a02', 'hybrid_predictions.json', 'selected_response'),
]
results, per_item = {}, {}
for label, folder, filename, field in runs:
    p = ROOT/'results'/folder/filename
    rows = read(p)
    assert len(rows) == len(expected) and len({x['id'] for x in rows}) == len(expected)
    assert {x['id'] for x in rows} == set(expected)
    scores, originals, groups, ems = {}, [], collections.defaultdict(list), []
    for x in rows:
        base, q, gold = expected[x['id']]
        assert x['question'] == q and x['gold_answer'] == gold, x['id']
        f1 = 100*ns['compute_f1'](gold, x[field])
        ems.append(100*ns['compute_exact'](gold, x[field]))
        scores[x['id']] = f1
        groups[base].append(f1)
        if x['id'] == base:
            originals.append(f1)
    group_means = {k: statistics.mean(v) for k,v in groups.items()}
    per_item[label] = group_means
    metrics = read(ROOT/'results'/folder/'rag_evaluation.json')
    key = 'reader_f1' if field == 'bert_span' else ('selected_f1' if field == 'selected_response' else 'rag_f1')
    agreement = abs(statistics.mean(scores.values())-metrics[key]) < 1e-8
    if label in {'phi_gold','phi_retrieved','qwen_guided','qwen_gated8','qwen_zs_guided','retrieved_extractor','qwen_verify','qwen_guarded','qwen_verdict'}:
        assert agreement, (label, key, statistics.mean(scores.values()),metrics[key])
    results[label] = dict(source=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        f1=statistics.mean(scores.values()), em=statistics.mean(ems),
        original_questions_f1=statistics.mean(originals), question_macro_f1=statistics.mean(group_means.values()),
        n_examples=len(rows), n_questions=len(groups), metric_agrees=agreement,
        saved_summary_f1=metrics[key], saved_summary_em=metrics['reader_exact_match' if field=='bert_span' else ('selected_exact_match' if field=='selected_response' else 'rag_exact_match')])

def quantile(xs, p):
    xs=sorted(xs); pos=(len(xs)-1)*p; lo=int(pos); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)

ids=sorted(per_item['phi_gold'])
rng=random.Random(20261028)
draws=[[rng.randrange(len(ids)) for _ in ids] for _ in range(10000)]
def interval(values):
    boot=[sum(values[i] for i in sample)/len(ids) for sample in draws]
    return [quantile(boot,.025),quantile(boot,.975)]
for k,v in per_item.items():
    results[k]['question_macro_f1_ci95']=interval([v[i] for i in ids])
paired={}
for a,b in [('phi_gold','phi_retrieved'),('qwen_gated8','qwen_guided'),('qwen_guided','retrieved_extractor')]:
    values=[per_item[a][i]-per_item[b][i] for i in ids]
    paired[f'{a} minus {b}']=dict(question_macro_difference=statistics.mean(values), ci95=interval(values))
report=dict(split=dict(total_items=len(items),training_items=len(train),evaluation_items=len(held),evaluation_examples=len(expected),
    stratified=False,independent_test=False,reason='Same seed-42 item holdout used for checkpoint selection and subsequent evaluation.'),
    extractive_best_validation=best,prediction_audit=results,paired_cluster_bootstrap=paired,
    bootstrap=dict(seed=20261028,replicates=10000,unit='original question; equal weight per question after averaging its variants',
    caveat='Conditional on selected models and reused validation set; does not correct model/threshold selection bias or training-seed variability.'))
report['extractive_inventory'] = inventory
report['saved_rag_metrics'] = {str(p.relative_to(ROOT)): read(p) for p in sorted({ROOT/'results'/folder/'rag_evaluation.json' for _,folder,_,_ in runs})}
# Lightweight reproduction does not require unpublished model weights or cluster logs.
report['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT/'data/processed/corpus_completo.json', ROOT/'scripts/train_unifesp_qa.py',
    ROOT/'scripts/train_llm_rag.py',ROOT/'scripts/evaluate_hybrid_rag.py',
    ROOT/'scripts/evaluate_generative_rag.py',ROOT/'scripts/retriever.py']}
report['reproduction_scope'] = 'Saved-prediction metric reproduction; retraining requires base models and adapters, not distributed here.'
(OUT/'verified_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(split=report['split'],predictions=results,paired=paired),indent=2))
