# Privacidade e preparação para compartilhamento

Esta distribuição passou por minimização direcionada de dados pessoais. **Não é um dataset totalmente anônimo.**

## Informações preservadas

- Nome da UNIFESP, informações acadêmicas e páginas institucionais de origem.
- Contatos gerais do programa e dos serviços universitários presentes nas fontes. A lista revisada está em `reproducibility/privacy_policy.json`; grafias históricas não foram corrigidas nem verificadas como contatos atualmente ativos.
- Os três telefones publicados no contexto da secretaria, não números pessoais.
- Nomes profissionais públicos de docentes, suas áreas e vínculos acadêmicos, necessários às perguntas sobre orientação, disciplinas e projetos. Afiliação institucional não significa ausência de dados pessoais.
- Referências bibliográficas e assinaturas públicas em normas institucionais.

## Informações substituídas

- O nome do representante discente foi substituído de modo consistente por `Estudante 001` nos documentos, no corpus e nos arquivos de resultados em que ocorria.
- Foram substituídos 21 endereços individuais de e-mail por rótulos distintos `contato-individual-NNN-removido`, inclusive endereços com domínio institucional.
- Links individuais de currículo Lattes foram substituídos por `[PERFIL_PESSOAL_REMOVIDO]`.
- Não foram encontrados telefones particulares ou endereços residenciais na revisão realizada; isso não equivale a uma garantia automática sobre todos os textos livres.

As posições `answer_start` foram recalculadas e verificadas em todas as 514 perguntas. A correspondência entre rótulos e identidades não faz parte da distribuição.

## Limites e resultados do artigo

Funções, datas, áreas de pesquisa e páginas institucionais ainda podem permitir identificação indireta. A substituição do nome do estudante é seudonimização, não garantia de anonimato. A presença de nomes públicos de docentes deve ser considerada na decisão do responsável pelo projeto sobre disponibilização do dataset.

A auditoria das predições foi executada novamente depois das substituições. F1, EM, Q-macro, intervalos bootstrap, divisões e tabelas numéricas permaneceram iguais aos registros do artigo. Os hashes dos arquivos alterados foram atualizados. Isso não implica que um novo treinamento com os textos modificados produzirá os mesmos modelos ou resultados.

Execute `python scripts/check_privacy.py` para verificar a política de contatos e as posições das respostas. A verificação detecta padrões conhecidos e contatos fora da lista revisada; não substitui revisão humana de nomes, endereços e identificação indireta.

## GitHub e Zenodo

O histórico Git local antecede estas alterações e pode conter os dados substituídos. Não publique esse histórico. `python scripts/package_artifact.py` gera um ZIP apenas com os arquivos do manifesto, sem `.git`; utilize seu conteúdo em um repositório novo com histórico inicial limpo.

Arquivos ZIP anteriores à revisão não foram atualizados e não devem ser enviados. A publicação do dataset no Zenodo permanece pendente da revisão do responsável pelo projeto. Nenhum envio é realizado pelos scripts de verificação ou empacotamento.
