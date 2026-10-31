# Institutional-QA-PT

Código e artefatos de reprodução do artigo **Local Institutional QA under Resource and Privacy Constraints: Benchmarking Extractive, Generative, and Verified Hybrid Pipelines in Portuguese**.

O projeto compara leitores extrativos, geradores com recuperação de documentos (RAG) e estratégias híbridas para responder perguntas sobre um programa de pós-graduação da UNIFESP. Os guias estão em português; identificadores dos modelos e interfaces do código foram preservados.

## Comece aqui

Use Python 3.10 ou superior e execute na raiz do projeto:

```bash
python scripts/verify_artifact.py
```

Não é necessário instalar dependências, baixar modelos ou usar GPU para essa verificação. O comando verifica os arquivos, a política de contatos, as posições das respostas, a divisão dos dados, EM/F1, 186 resumos extrativos elegíveis, tabelas e 10.000 reamostragens bootstrap pareadas por pergunta original (semente 20261028).

**Escopo verificado:** métricas das predições salvas. Os adaptadores não estão distribuídos e não há registro completo das revisões e do ambiente original. Não se promete reproduzir o treinamento bit a bit.

Consulte as [instruções de treinamento e inferência](docs/TRAINING.md), as [tabelas de resultados](docs/RESULTS.md), a [galeria de figuras](docs/FIGURAS.md) e a [revisão de privacidade](docs/PRIVACIDADE.md).

## Organização

```text
Institutional-QA-PT/
├── README.md
├── requirements.txt
├── preprocess_docs.py       # Leitura, limpeza e segmentação dos documentos
├── data/
│   ├── documents/           # Documentos institucionais para recuperação
│   └── processed/           # Corpus de perguntas e respostas revisado
├── scripts/                 # Treinamento, avaliação, figuras e verificações
├── results/                 # Predições e resumos históricos preservados
├── reproducibility/         # Auditoria, divisão, proveniência e política de contatos
├── figures/                 # Gráficos em português, PDF/PNG e pipeline
│   └── paper/               # Cópias exatas das figuras do manuscrito local
├── docs/                    # Guias, tabelas e revisão de privacidade
├── environment/             # Pacotes observados no cluster
├── .github/workflows/       # Verificação automática no GitHub
└── ARTIFACT_MANIFEST.json   # Checksums SHA-256 dos arquivos distribuídos
```

O corpus está em `data/processed/corpus_completo.json`. Os scripts encontram os documentos em `data/documents/` a partir da localização do próprio projeto. Os nomes históricos em `results/` foram preservados; use os mapeamentos de `reproducibility/audit_results.py` para interpretá-los.

Checkpoints, caches, logs do cluster, datasets alternativos e fontes do manuscrito não fazem parte do pacote. Alguns caminhos em `historical_provenance.json` são referências a arquivos históricos não distribuídos, e não entradas necessárias à verificação.

## Figuras

Em um ambiente de visualização com NumPy e Matplotlib:

```bash
python -m pip install numpy matplotlib
python scripts/generate_figures.py
```

O script gera seis gráficos em português a partir de `reproducibility/verified_results.json`: comparação extrativa, memória/F1, geradores sem adaptação no domínio, contexto de referência versus recuperado, variantes híbridas e intervalos das diferenças pareadas. Veja as explicações e limites na [galeria](docs/FIGURAS.md).

![F1 dos leitores extrativos com contexto de referência](figures/extractive_f1.png)

![Diferenças pareadas e intervalos bootstrap](figures/paired_differences.png)

Os gráficos usam validação reutilizada. Os intervalos dos ganhos híbridos incluem zero; os resultados não estabelecem superioridade geral entre arquiteturas.

As figuras originais do paper estão em `figures/paper/` e não são sobrescritas por esse comando. Para recompilar o diagrama editável, instale LaTeX com TikZ, standalone e FontAwesome5:

```bash
python scripts/build_pipeline.py
```

Metadados de PDFs podem variar entre plataformas. A reprodução numérica usa a auditoria JSON, não igualdade binária entre figuras regeneradas.

## Dados e interpretação

O corpus contém 514 perguntas originais e 1.507 paráfrases não vazias. A divisão 80/20 por pergunta original (semente 42) contém 411/103 originais, expandidas após a divisão para 1.613/408 formulações. Os IDs estão em `reproducibility/split_manifest.json`. As execuções históricas de recuperação indexaram 318 segmentos.

**Não há conjunto de teste independente:** o mesmo holdout selecionou checkpoints e limiares. Os resultados são estimativas exploratórias de validação. O bootstrap agrupa variantes pela pergunta original, mas não elimina viés de seleção nem variação entre sementes. BERT com contexto de referência e recuperado usa protocolos de decodificação diferentes. A memória registrada é o incremento de alocação de treinamento em GiB, não o pico total do dispositivo nem energia consumida. O grafo estava indisponível nas execuções históricas de Phi. Os intervalos dos ganhos híbridos incluem zero.

## Dataset e citação

O corpus de perguntas e respostas está publicado no Zenodo como **Institutional-QA-PT**, versão **1.0.0**: [doi:10.5281/zenodo.23839298](https://doi.org/10.5281/zenodo.23839298). Esse depósito contém `corpus_completo.json`; os documentos de origem, a divisão dos dados, o código e os artefatos de avaliação estão neste repositório.

> Nina Capujra, M. M., Paye Mamani, J. E., & Vega Oliveros, D. A. (2026). *Institutional-QA-PT: A Portuguese Question Answering Dataset for Institutional Information* (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23839298

## Licenças

- **Código original deste projeto:** [MIT](LICENSE).
- **Dataset `data/processed/corpus_completo.json`:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), conforme o depósito no Zenodo. A licença MIT não se aplica ao dataset.
- **Documentos institucionais, modelos e demais materiais de terceiros:** permanecem sujeitos aos direitos e às condições de suas fontes. A licença MIT não concede direitos sobre esses materiais, nem atribui automaticamente uma licença aos demais artefatos não classificados como código.

## Privacidade e direitos de uso

Foram substituídos o nome do representante discente, 21 e-mails individuais e links individuais de currículo. Contatos gerais da universidade, nomes profissionais públicos de docentes necessários ao benchmark e referências institucionais permanecem. **O dataset não é totalmente anônimo.** Leia a [revisão de privacidade](docs/PRIVACIDADE.md).

As métricas salvas continuaram iguais após as substituições; isso não garante resultados idênticos em um novo treinamento com os textos revisados. O corpus revisado foi publicado no Zenodo na versão 1.0.0. As condições das fontes e as licenças dos modelos continuam aplicáveis. Este pacote não concede uma nova licença sobre material de terceiros.

## Principais scripts

| Arquivo | Finalidade |
|---|---|
| `scripts/train_unifesp_qa.py` | Treinar leitores extrativos Full/LoRA/QLoRA |
| `scripts/qa_utils.py` | Tokenização, janelas e decodificação de spans |
| `scripts/train_llm_rag.py` | Adaptar geradores por treinamento supervisionado |
| `scripts/evaluate_generative_rag.py` | Avaliar com contexto recuperado ou de referência |
| `scripts/evaluate_hybrid_rag.py` | Avaliar orientação por span, fallback e roteamento |
| `scripts/retriever.py` | Recuperação BM25, densa e híbrida |
| `reproducibility/audit_results.py` | Recalcular evidências sem carregar modelos |
| `scripts/generate_figures.py` | Gerar gráficos a partir da auditoria |
| `scripts/check_privacy.py` | Verificar contatos e posições das respostas |
| `scripts/verify_artifact.py` | Verificar integridade e reprodução numérica |
