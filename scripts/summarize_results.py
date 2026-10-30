"""Generate readable tables from the canonical audit, without inline scores."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'reproducibility/verified_results.json').read_text())
lines=['# Resultados recalculados', '',
       'Gerado a partir de `reproducibility/verified_results.json`. Todas as estimativas usam validação reutilizada.', '',
       '## Leitores extrativos com contexto de referência', '',
       '| Modelo | Ajuste | F1 | EM | Incremento de alocação (GiB) | Treinamento (min) | Execução de origem |',
       '|---|---|---:|---:|---:|---:|---|']
for r in d['extractive_best_validation']:
    lines.append(f"| {r['model_type']} | {r['tuning_type']} | {r['validation_f1']:.2f} | {r['validation_exact_match']:.2f} | {r['reported_memory_increment_gib']:.2f} | {r['train_time_minutes']:.1f} | `{r['run']}` |")
lines += ['', '## Predições preservadas', '',
          'F1/EM usam a média de 408 formulações. Q-macro atribui peso igual às 103 perguntas originais, após calcular a média de suas variantes.', '',
          '| Identificador | F1 | EM | F1 Q-macro (intervalo de 95%) | F1 apenas originais | Origem |',
          '|---|---:|---:|---|---:|---|']
for name,r in d['prediction_audit'].items():
    lo,hi=r['question_macro_f1_ci95']
    lines.append(f"| {name} | {r['f1']:.2f} | {r['em']:.2f} | {r['question_macro_f1']:.2f} [{lo:.2f}, {hi:.2f}] | {r['original_questions_f1']:.2f} | `{r['source']}` |")
lines += ['', '## Diferenças pareadas por pergunta original', '',
          '| Contraste | Diferença Q-macro | Intervalo de 95% |', '|---|---:|---|']
for name,r in d['paired_cluster_bootstrap'].items():
    lo,hi=r['ci95']
    lines.append(f"| {name} | {r['question_macro_difference']:.2f} | [{lo:.2f}, {hi:.2f}] |")
lines += ['', 'O campo `retrieved_extractor` usa os spans extraídos da execução com roteamento. Os dois regimes de contexto BERT usam protocolos de decodificação diferentes. Limiares híbridos e checkpoints foram selecionados neste holdout. Os intervalos não eliminam o viés de seleção.', '']
(ROOT/'docs/RESULTS.md').write_text('\n'.join(lines))
print('Gerado docs/RESULTS.md a partir das métricas canônicas.')
