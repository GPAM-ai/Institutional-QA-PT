# Treinamento e inferência

A reprodução das métricas salvas está no README. Executar os modelos novamente é um experimento separado: adaptadores/checkpoints não são distribuídos e as revisões exatas dos modelos base e do ambiente original não foram preservadas. Não se promete reprodução bit a bit. A proveniência histórica de Phi/Qwen está em `reproducibility/historical_provenance.json`; `environment/current_packages.txt` descreve o ambiente observado no cluster, não necessariamente o de cada execução.

## Ambiente e caminhos

Execute os exemplos na raiz de `Institutional-QA-PT`. Instale as dependências com um PyTorch compatível com sua GPU/CUDA. As versões mínimas de `requirements.txt` não constituem um lockfile testado em todas as combinações.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Os identificadores dos modelos estão nos dicionários `MODEL_MAPPING`. Obtenha as autorizações necessárias com sua própria conta Hugging Face e configure `HF_TOKEN` de forma privada. Não registre credenciais no Git.

| Recurso | Caminho |
|---|---|
| Corpus padrão | `data/processed/corpus_completo.json` |
| Documentos para recuperação | `data/documents/` |
| Resumos e predições preservados | `results/` |
| Adaptadores gerativos criados pelo treinamento | `models/` |
| Novas avaliações dos exemplos abaixo | `results_new/` |

Use `--dense_model intfloat/multilingual-e5-small` para baixar o modelo denso da origem. O padrão local `models/multilingual-e5-small` exige que ele já esteja nessa pasta. O reranker opcional também exige um modelo local; recursos avançados não são necessários à reprodução das métricas salvas.

**Execute novos treinamentos em uma cópia de trabalho.** Os scripts gravam em `results/` e `models/`; configurações com o mesmo identificador podem sobrescrever arquivos históricos. Pesos BERT são gravados em `results/<identificador>/best_model/` e não estão distribuídos.

## Leitor extrativo

```bash
python scripts/train_unifesp_qa.py --model_type bertimbau_large --tuning_type lora --batch_size 64 --lr 0.0001 --epochs 15
```

Configurações selecionadas, taxas de aprendizado, lotes, tempos e memória estão em `extractive_best_validation` na auditoria. `extractive_inventory` conserva os resumos elegíveis usados na seleção.

## Adaptação supervisionada do gerador

```bash
python scripts/train_llm_rag.py --model_type phi_4_mini --tuning_type qlora --epochs 5 --lr 0.0002 --batch_size 4 --gradient_accumulation_steps 4 --max_seq_length 1024
```

Para Qwen, use `--model_type qwen_1.5b`. O adaptador Phi selecionado é salvo em `models/phi_4_mini_lora_rag/`; o nome histórico usa `lora` mesmo no treinamento quantizado.

## Contexto recuperado e de referência

Após treinar ou fornecer o adaptador correspondente:

```bash
python scripts/evaluate_generative_rag.py --model_type phi_4_mini --peft_model_path models/phi_4_mini_lora_rag --retriever_type hybrid --dense_model intfloat/multilingual-e5-small --k 5 --alpha 0.5 --output_dir results_new/phi_retrieved
python scripts/evaluate_generative_rag.py --model_type phi_4_mini --peft_model_path models/phi_4_mini_lora_rag --retriever_type hybrid --dense_model intfloat/multilingual-e5-small --k 5 --alpha 0.5 --gold_context --output_dir results_new/phi_reference
```

`--output_dir` separa as novas avaliações. Sem essa opção, o avaliador mantém o diretório histórico padrão e pode sobrescrever uma execução anterior.

A invocação histórica de Phi solicitou `--use_graph`, mas o componente estava indisponível. Os exemplos usam a rota efetiva sem grafo. Exemplos fixos no prompt permanecem habilitados: ausência de adaptação no domínio não significa ausência de exemplos no prompt.

Para os cinco geradores adicionais, omita `--peft_model_path` e use `llama_3b`, `qwen_1.5b`, `llama_1b`, `smollm2` ou `tucano`, com os mesmos argumentos de recuperação e um `--output_dir` diferente.

## Estratégia híbrida

Após treinar os adaptadores BERT e Qwen:

```bash
python scripts/evaluate_hybrid_rag.py --bert_model_type bertimbau_large --bert_tuning_type lora --bert_bs 64 --bert_lr 0.0001 --bert_epochs 15.0 --llm_model_type qwen_1.5b --llm_adapter_path models/qwen_1.5b_lora_rag --retriever_type hybrid --dense_model intfloat/multilingual-e5-small --k 5 --alpha 0.5 --best_rag --router_mode confidence_gated --confidence_threshold 8 --span_guidance truth --output_dir results_new/gated_t8
```

Use `--router_mode always_llm` para geração sempre orientada. `truth` é o nome histórico da orientação pelo span predito, não acesso à resposta de referência. O código conserva prompts, normalização e decodificação. As opções `--help` exigem as dependências de modelos para importar os scripts.

A reorganização não incluiu treinamento ou inferência novos com GPU. Textos revisados por privacidade podem gerar resultados diferentes em novas execuções, mesmo quando as métricas salvas coincidem com o artigo.

## Direitos dos documentos e modelos

As condições dos documentos de origem, as licenças dos modelos e as restrições de acesso continuam aplicáveis. Este pacote não atribui uma nova licença a material de terceiros. Leia `docs/PRIVACIDADE.md` antes de compartilhar dados.
