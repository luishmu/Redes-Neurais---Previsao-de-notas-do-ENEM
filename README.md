# Redes Neurais – Previsão de Nota do ENEM

> **Uma rede neural consegue prever a nota do ENEM a partir do perfil socioeconômico do candidato? E se for usada para tomar decisões, quem ela prejudica?**

Trabalho da disciplina de **Redes Neurais** da Pós-graduação em Ciência de Dados da **UNIFOR**.
Autor: **Luis Helder Martins Uchoa**

---

## Sumário

1. [Resumo](#resumo)
2. [Os dados e o obstáculo da LGPD](#os-dados-e-o-obstáculo-da-lgpd)
3. [Estrutura do repositório](#estrutura-do-repositório)
4. [Como executar](#como-executar)
5. [Metodologia](#metodologia)
6. [Resultados](#resultados)
7. [Do diagnóstico à ação](#do-diagnóstico-à-ação)
8. [Conclusões](#conclusões)
9. [Limitações](#limitações)
10. [Fontes](#fontes)

---

## Resumo

- Uma **MLP com embeddings (PyTorch)** prevê a nota média do ENEM 2023 (CN, CH, LC, MT e Redação) a partir de 29 variáveis de perfil e questionário socioeconômico.
- A rede **empata com o gradient boosting** (MAE ≈ 57 pontos, R² ≈ 0,35) e supera um pouco a regressão linear. O limite está na **informação disponível**, não no modelo.
- **O modelo aprende a desigualdade.** Entre alunos com nota real ≥ 700, a rede prevê **548** para quem tem renda de até 1 salário mínimo (SM) e **661** para quem tem mais de 7 SM. Usada para selecionar os 10% melhores, ela deixaria de fora ~95% dos talentos de baixa renda.
- **Os dados de treino definem o viés.** Treinada só com alunos de alta renda, a rede superestima os de baixa renda em **+60 pontos**.
- **Mais informação reduz o viés, mas não acaba com ele.** Escola, língua estrangeira, ano de conclusão e contexto do município (IDHM etc.) elevam o R² para 0,39 e reduzem a diferença de previsão de 113 para 97 pontos.
- **Uso responsável:** o modelo deve **indicar onde investir ajuda** (áreas, escolas, municípios) e **medir valor agregado**, nunca selecionar pessoas.

---

## Os dados e o obstáculo da LGPD

A ideia inicial era usar o **ENEM 2025**. Mas, desde 2024, o INEP divulga o questionário (`PARTICIPANTES`) e as notas (`RESULTADOS`) em **arquivos sem chave em comum**, para cumprir a LGPD. Não dá para ligar o perfil de uma pessoa à nota dela.

**Solução adotada:**

| Ano | Uso no projeto |
|---|---|
| **2023** | Treino e avaliação. É o último ano com perfil, questionário e notas no mesmo registro. |
| **2025** | Validação **agregada por UF**: o modelo de 2023 é aplicado aos questionários de 2025 e a média prevista é comparada com a média real de cada estado. |

Dos **3,93 milhões** de inscritos em 2023, **2,59 milhões** fizeram as 4 provas objetivas e tiveram redação válida. Foi sorteada uma amostra de **15% (387 mil candidatos)**.

---

## Estrutura do repositório

```
├── 01_preparar_dados.py        # lê o microdado 2023 direto do .zip, filtra e sorteia a amostra
├── 02_preparar_2025.py         # médias reais por UF (RESULTADOS) e amostra de questionários (PARTICIPANTES) de 2025
├── 02b_contexto_municipal.py   # baixa dados externos por município (IDHM, Gini, população, distância à capital…)
├── 03_rede_neural_enem.ipynb   # notebook principal: modelos, experimentos e conclusões
├── 03_rede_neural_enem.html    # o mesmo notebook já executado, para leitura no navegador
├── requirements.txt
└── dados/
    ├── enem2023_amostra.csv.gz               # amostra de 2023 (gerada pelo script 01)
    ├── enem2025_media_real_por_uf.csv        # gerado pelo script 02
    ├── enem2025_notas_amostra.csv.gz         # gerado pelo script 02
    ├── enem2025_participantes_amostra.csv.gz # gerado pelo script 02
    ├── contexto_municipal.csv                # gerado pelo script 02b
    └── Dicionário_Microdados_Enem_2023.xlsx
```

As amostras já estão em `dados/`, então **o notebook roda direto**, sem baixar os microdados completos.

---

## Como executar

### 1. Ambiente

```bash
pip install -r requirements.txt
```

Python 3.10+ com `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `torch` e `jupyter`.

### 2. Rodar o notebook (caminho rápido)

```bash
jupyter notebook 03_rede_neural_enem.ipynb
```

Leva cerca de **5 minutos** em CPU.

### 3. Regerar os dados do zero (opcional)

1. Baixe os microdados do **ENEM 2023** e do **ENEM 2025** no portal de dados abertos do INEP.
2. Coloque `microdados_enem_2023.zip` e a pasta descompactada `microdados_enem_2025/` **na pasta acima deste repositório** (os scripts procuram em `../`).
3. Execute:

```bash
python 01_preparar_dados.py                  # ~1 min; lê o CSV de 1,7 GB em blocos
python 02_preparar_2025.py resultados
python 02_preparar_2025.py participantes
python 02b_contexto_municipal.py             # precisa de internet (baixa do GitHub)
```

---

## Metodologia

### Harmonização 2023 ↔ 2025

O questionário mudou entre os anos (perguntas removidas, ordem diferente, contagens com teto "3 ou mais"). Cada variável foi convertida para um **código comum**, como renda em faixas de salário mínimo, escolaridade e ocupação dos pais e bens em casa. O resultado são 29 variáveis com o mesmo significado nos dois anos.

### Modelos

Divisão dos dados: **70% treino / 15% validação (early stopping) / 15% teste**.

| Modelo | Tipo | Como trata as categorias |
|---|---|---|
| Média global | referência mínima | — |
| Regressão linear (Ridge) | ML tradicional, linear | one-hot |
| Gradient boosting (`HistGradientBoostingRegressor`) | ML tradicional, não linear | suporte nativo a categóricas |
| **MLP com embeddings (PyTorch)** | rede neural | um vetor aprendido por categoria |

Arquitetura da rede:

```
[29 embeddings (+ numéricas)] → concat → Linear(·,256) → BatchNorm → ReLU → Dropout(0,2)
                                       → Linear(256,128) → BatchNorm → ReLU → Dropout(0,2)
                                       → Linear(128,1) → nota (padronizada)
```

Perda **Huber**, otimizador **AdamW** (lr 1e-3), batch 1024, **early stopping** pelo MAE de validação.

### Experimentos

| Experimento | Pergunta |
|---|---|
| **A** – viés do modelo bem treinado | Para a mesma nota real, a rede prevê notas diferentes por renda? Quem ela selecionaria para os 10% melhores? |
| **B** – treino enviesado | O que acontece se a rede for treinada só com alta renda (B1) ou só com Sul/Sudeste (B2)? Cada caso é comparado com um **controle aleatório de mesmo tamanho**. |
| **C** – mais informação | Escola, língua estrangeira, ano de conclusão e **contexto municipal** melhoram a previsão e reduzem o viés? |
| **Do diagnóstico à ação** | Em que área, escola e município agir antes da prova? |
| **Validação 2025** | O modelo de 2023 ordena corretamente as UFs de 2025? |

---

## Resultados

### Desempenho (conjunto de teste, 58 mil candidatos)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Média global | 72,5 | 89,4 | 0,000 |
| Regressão linear | 58,0 | 73,1 | 0,330 |
| Gradient boosting | 57,10 | 72,0 | 0,351 |
| **Rede neural (MLP)** | **57,06** | 72,1 | 0,349 |

A rede **empata com o boosting**, um resultado comum em dados tabulares. A relação entre perfil e nota é quase **aditiva**, o que deixa pouco espaço para modelos mais complexos.

### Experimento A: precisão média não é justiça

- Na média de cada grupo, o erro fica perto de zero.
- Para alunos com **nota real ≥ 700**, a rede prevê **548** (renda ≤ 1 SM) contra **661** (renda > 7 SM).
- Se a rede escolhesse os **10% melhores**: quem tem renda acima de 4 SM é **19%** dos candidatos, **54%** do top 10% real e **82%** dos selecionados pela rede.

### Experimento B: os dados de treino definem o viés

| Cenário | Efeito |
|---|---|
| B1 – treino só com renda > 4 SM | superestima em **+60 pontos** quem tem renda ≤ 1 SM; R² cai de 0,33 para 0,05 |
| B2 – treino só com Sul/Sudeste | efeito pequeno (Nordeste −9 pontos): o viés vem mais de **quem** está no treino do que de **onde** essas pessoas moram |

O **controle aleatório de mesmo tamanho** não apresenta esses erros: o problema é a **composição** dos dados, não o volume.

### Experimento C: mais informação ajuda, mas não resolve

| Cenário | R² | Previsão p/ nota ≥ 700 (≤ 1 SM × > 7 SM) | Talentos ≤ 2 SM selecionados |
|---|---|---|---|
| Só questionário | 0,349 | 548 × 661 (Δ 113) | 5% |
| + escola, língua, ano de conclusão | 0,385 | 562 × 667 (Δ 105) | 7% |
| + contexto municipal | 0,392 | 568 × 665 (Δ 97) | 9% |

**Falácia ecológica:** o IDHM do município tem correlação de **0,78** com a nota **média** do município, mas melhora quase nada a previsão para **pessoas** (+0,007 de R²).

### Validação em 2025

A rede de 2023, aplicada aos questionários de 2025, ordena as UFs com correlação de **0,98** com a média real. As previsões ficam ~10 pontos abaixo, porque `PARTICIPANTES` inclui faltosos, e são **comprimidas** (desvio-padrão previsto de 52 contra 86 no real).

---

## Do diagnóstico à ação

O resíduo **(nota real − nota esperada para o perfil)** vira uma métrica de **valor agregado**:

- **Áreas:** Matemática (Δ 183 pontos entre os extremos de renda) e Redação (Δ 169) concentram a desigualdade. A Redação é a área mais **treinável**.
- **Escolas:** alunos com renda ≤ 2 SM de **escolas federais** ficam **+53 pontos** acima do esperado (com efeito de seleção pela prova de ingresso); os de escolas estaduais ficam −5.
- **Municípios:** com o mesmo perfil socioeconômico, a diferença chega a dezenas de pontos (ex.: Itabaiana-SE, Acaraú-CE e Araripina-PE acima do esperado; Corrente-PI abaixo). Os valores usam ajuste bayesiano empírico, e o teste de duas metades indica que o ranking é em boa parte sinal, mas deve ser confirmado com mais de um ano.

### Ciclo de decisão proposto

1. **Diagnosticar:** onde a nota esperada é baixa e onde o resíduo é negativo.
2. **Priorizar:** necessidade × espaço de melhora × número de alunos.
3. **Aprender com quem supera o esperado:** estudar municípios e escolas com resíduo positivo.
4. **Intervir com teste:** piloto com grupo de comparação, usando a nota esperada como linha de base.
5. **Monitorar:** os `RESULTADOS` de 2024+ trazem o código da escola, o que permite acompanhar o valor agregado por escola todo ano sem identificar alunos.

> **O modelo deve priorizar onde ajudar, nunca selecionar quem merece oportunidade.**

---

## Conclusões

1. Dá para prever **parte** da nota pelo perfil socioeconômico, mas a maior parte da variação não está nos dados.
2. **Precisão média não é justiça:** um modelo preciso no agregado pode tornar invisíveis os talentos de baixa renda.
3. **Os dados de treino definem o viés**, e validar no mesmo recorte enviesado não revela o problema.
4. **Mais informação reduz o viés, mas não acaba com ele**; o contexto municipal ajuda pouco no nível individual.
5. **Em dados tabulares, a rede neural empata com o gradient boosting:** o viés vem dos dados, não da arquitetura.
6. **LGPD:** a separação dos microdados protege os participantes, mas também impede esse tipo de auditoria com dados recentes.

---

## Limitações

- O município usado é o **da prova**, não necessariamente o de residência.
- O Atlas do Desenvolvimento Humano é de **2010**.
- O tipo de escola e a dependência administrativa só estão preenchidos para quem **concluía o ensino médio em 2023**.
- Os resultados são **associações, não causalidade**: escolher inglês está associado a notas maiores, mas trocar de língua provavelmente não aumenta a nota.

---

## Fontes

- **INEP – Microdados do ENEM 2023 e 2025:** portal de dados abertos do INEP (gov.br/inep)
- **Atlas do Desenvolvimento Humano no Brasil (PNUD/IPEA/FJP), Censo 2010:** via [mauriciocramos/IDHM](https://github.com/mauriciocramos/IDHM)
- **Tabela de municípios (coordenadas, capitais, população 2021):** [mapaslivres/municipios-br](https://github.com/mapaslivres/municipios-br)
