# Resultados recalculados

Gerado a partir de `reproducibility/verified_results.json`. Todas as estimativas usam validação reutilizada.

## Leitores extrativos com contexto de referência

| Modelo | Ajuste | F1 | EM | Incremento de alocação (GiB) | Treinamento (min) | Execução de origem |
|---|---|---:|---:|---:|---:|---|
| mbert | full | 57.30 | 43.63 | 4.73 | 4.4 | `mbert_full_bs16_lr5e-05_ep15.0` |
| mbert | lora | 48.00 | 33.82 | 2.69 | 2.8 | `mbert_lora_bs16_lr0.0002_ep15.0` |
| mbert | qlora | 47.84 | 34.80 | 0.43 | 5.6 | `mbert_qlora_bs8_lr0.0002_ep15.0` |
| xlmr | full | 50.91 | 38.73 | 3.92 | 4.3 | `xlmr_full_bs8_lr5e-05_ep15.0` |
| xlmr | lora | 42.68 | 28.19 | 2.69 | 2.2 | `xlmr_lora_bs16_lr0.0002_ep15.0` |
| xlmr | qlora | 44.01 | 25.49 | 0.83 | 3.6 | `xlmr_qlora_bs16_lr5e-05_ep15.0` |
| bertugues | full | 58.05 | 43.63 | 7.40 | 2.7 | `bertugues_full_bs32_lr0.0001_ep15.0` |
| bertugues | lora | 46.13 | 33.33 | 1.46 | 2.7 | `bertugues_lora_bs8_lr0.0002_ep15.0` |
| bertugues | qlora | 46.11 | 32.35 | 0.83 | 3.7 | `bertugues_qlora_bs16_lr0.0002_ep15.0` |
| bertimbau_base | full | 65.99 | 50.98 | 13.77 | 2.5 | `bertimbau_base_full_bs64_lr0.0002_ep15.0` |
| bertimbau_base | lora | 66.29 | 47.30 | 10.19 | 2.1 | `bertimbau_base_lora_bs64_lr0.0002_ep15.0` |
| bertimbau_base | qlora | 67.88 | 49.51 | 3.26 | 3.4 | `bertimbau_base_qlora_bs64_lr0.0002_ep15.0` |
| bertimbau_large | full | 67.87 | 49.02 | 36.11 | 6.4 | `bertimbau_large_full_bs64_lr0.0001_ep15.0` |
| bertimbau_large | lora | 73.35 | 53.43 | 26.21 | 4.7 | `bertimbau_large_lora_bs64_lr0.0001_ep15.0` |
| bertimbau_large | qlora | 72.96 | 53.68 | 5.83 | 7.9 | `bertimbau_large_qlora_bs64_lr0.0002_ep15.0` |

## Predições preservadas

F1/EM usam a média de 408 formulações. Q-macro atribui peso igual às 103 perguntas originais, após calcular a média de suas variantes.

| Identificador | F1 | EM | F1 Q-macro (intervalo de 95%) | F1 apenas originais | Origem |
|---|---:|---:|---|---:|---|
| phi_gold | 78.03 | 60.54 | 77.94 [72.28, 83.07] | 83.97 | `results/phi_4_mini_rag_sft_gold/predictions.json` |
| phi_retrieved | 53.32 | 34.56 | 53.25 [47.58, 58.76] | 65.90 | `results/phi_4_mini_rag_sft_hybrid_graph/predictions.json` |
| qwen_guided | 45.52 | 28.19 | 45.39 [39.79, 51.01] | 57.63 | `results/hybrid_sft_qwen15b_truth/hybrid_predictions.json` |
| qwen_gated8 | 48.47 | 30.64 | 48.31 [42.74, 53.90] | 60.62 | `results/hybrid_gated_sft_t8/hybrid_predictions.json` |
| qwen_zs_guided | 23.44 | 0.25 | 23.38 [19.83, 26.99] | 26.93 | `results/hybrid_rag_qwen_always_baseline/hybrid_predictions.json` |
| retrieved_extractor | 43.43 | 28.43 | 43.29 [37.18, 49.33] | 56.28 | `results/hybrid_gated_sft_t8/hybrid_predictions.json` |
| qwen_verify | 22.55 | 0.74 | 22.52 [19.01, 26.05] | 25.95 | `results/hybrid_verify_always/hybrid_predictions.json` |
| qwen_guarded | 31.77 | 12.01 | 31.73 [27.00, 36.43] | 40.92 | `results/hybrid_verify_evidence/hybrid_predictions.json` |
| qwen_verdict | 23.74 | 0.25 | 23.67 [20.03, 27.38] | 27.70 | `results/hybrid_llm_verify/hybrid_predictions.json` |
| llama3b | 44.96 | 24.51 | 45.03 [39.01, 50.79] | 59.39 | `results/llama_3b_rag/predictions.json` |
| qwen15b | 39.00 | 20.59 | 39.16 [33.43, 44.87] | 50.96 | `results/qwen_1.5b_rag/predictions.json` |
| smollm2 | 16.92 | 4.41 | 16.83 [13.29, 20.51] | 20.82 | `results/smollm2_rag/predictions.json` |
| llama1b | 27.90 | 15.93 | 27.78 [22.36, 33.45] | 39.41 | `results/llama_1b_rag/predictions.json` |
| tucano | 7.11 | 0.00 | 7.08 [5.48, 8.78] | 10.57 | `results/tucano_rag/predictions.json` |
| qwen_dense_heavy | 41.53 | 24.51 | 41.47 [35.79, 47.16] | 54.95 | `results/hybrid_sft_truth_a02/hybrid_predictions.json` |

## Diferenças pareadas por pergunta original

| Contraste | Diferença Q-macro | Intervalo de 95% |
|---|---:|---|
| phi_gold minus phi_retrieved | 24.70 | [18.65, 30.71] |
| qwen_gated8 minus qwen_guided | 2.92 | [-0.23, 6.43] |
| qwen_guided minus retrieved_extractor | 2.10 | [-2.05, 6.52] |

O campo `retrieved_extractor` usa os spans extraídos da execução com roteamento. Os dois regimes de contexto BERT usam protocolos de decodificação diferentes. Limiares híbridos e checkpoints foram selecionados neste holdout. Os intervalos não eliminam o viés de seleção.
