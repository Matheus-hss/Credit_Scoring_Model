# 📊 Credit Scoring Model — Scorecard com WOE, IV e Logistic Regression

📌 Visão Geral
Este projeto apresenta a construção completa de um modelo de Credit Scoring utilizando técnicas clássicas de modelagem de risco de crédito utilizadas em bancos e fintechs.  O objetivo foi prever a probabilidade de inadimplência (default) de clientes com base em variáveis financeiras e comportamentais, utilizando:
Feature Engineering
Binning automático
Weight of Evidence (WOE)
Information Value (IV)
Regressão Logística
Scorecard tradicional
Avaliação com AUC, KS, ROC
Validação de estabilidade (PSI)

O resultado final é um Score de Crédito interpretável, capaz de ordenar clientes de acordo com seu risco de inadimplência.

![Rplot_DefaultvsScore](https://github.com/user-attachments/assets/880be137-32e2-4b25-97b2-a5ded8ddb1aa)

📂 Estrutura do Projeto

credit_scoring_project/
│
├── data/
│   ├── train_base.csv
│   └── test_base.csv
│
├── notebooks/
│   └── credit_scoring_pipeline.R
│
├── outputs/
│   ├── scorecard_table.csv
│   └── score_distribution.png
│
└── README.md

🧠 Metodologia

O pipeline de modelagem segue etapas amplamente utilizadas na indústria de risco de crédito.

1️⃣ Data Understanding

Inicialmente foi realizada uma análise exploratória das variáveis presentes no dataset.

Principais ações:
- análise de distribuição das variáveis
- identificação de valores inconsistentes
- tratamento de outliers
- tratamento de valores ausentes
- padronização das variáveis
- Exemplos de problemas encontrados:
- idades negativas
- número de cartões de crédito extremamente altos
- valores faltantes em variáveis financeiras
- Esses problemas foram tratados antes da modelagem.

2️⃣ Feature Engineering

Foram criadas novas variáveis que capturam relações financeiras importantes.

Exemplos:

- Debt-to-Income Ratio
Debt_to_Income = Outstanding_Debt / Annual_Income

Indica quanto da renda está comprometido com dívidas.

- Loans per Income
Loans_per_Income = Num_of_Loan / Annual_Income

Mede exposição a crédito.

- Investment to Income
Investiment_to_Income = Amount_invested_monthly / Annual_Income

Indica capacidade de poupança.


Indicadores comportamentais

Foram criadas variáveis binárias como:

- High_Utilization
- Young_Client

Essas variáveis ajudam a capturar padrões de risco.

3️⃣ Binning Automático

Para permitir a construção de um scorecard interpretável, foi aplicado binning automático utilizando o pacote scorecard.
O binning agrupa valores das variáveis em faixas com comportamento de risco semelhante.

| Age           | Bin   |
| ------------- | ----- |
| Age < 35      | Bin 1 |
| 35 ≤ Age < 46 | Bin 2 |
| Age ≥ 46      | Bin 3 |

4️⃣ Weight of Evidence (WOE)

Após o binning, cada faixa é transformada em WOE (Weight of Evidence).

O WOE mede a relação entre:

proporção de bons clientes
vs
proporção de inadimplentes

Fórmula:

WOE=ln⁡(%bons/%maus)

Essa transformação possui vantagens importantes:

lineariza relações
reduz impacto de outliers
melhora estabilidade do modelo
facilita interpretação

5️⃣ Information Value (IV)

O Information Value mede o poder preditivo das variáveis.

Classificação padrão:
| IV       | Força da variável |
| -------- | ----------------- |
| < 0.02   | sem poder         |
| 0.02–0.1 | fraco             |
| 0.1–0.3  | médio             |
| 0.3–0.5  | forte             |

Variáveis com IV < 0.02 foram removidas da modelagem.

6️⃣ Transformação WOE

Após seleção das variáveis, todas foram transformadas em WOE.

7️⃣ Modelagem — Regressão Logística

Foi construído um modelo baseline utilizando:

- tidymodels
- parsnip
- workflows

A regressão logística estima a P(Default), ou seja, a probabilidade de inadimplência do cliente.

📈 Avaliação do Modelo

Foram utilizadas métricas clássicas de risco de crédito.
ROC AUC
AUC = 0.80

| AUC | qualidade |
| --- | --------- |
| 0.5 | aleatório |
| 0.7 | bom       |
| 0.8 | muito bom |
| 0.9 | excelente |

KS Statistic
KS = 0.55

| KS      | interpretação |
| ------- | ------------- |
| <0.2    | fraco         |
| 0.2–0.4 | razoável      |
| 0.4–0.6 | bom           |

📉 Curva ROC
A curva ROC mostra a capacidade do modelo de separar clientes bons e ruins

![Rplot_Curva_ROC](https://github.com/user-attachments/assets/3fba6bd6-b643-48bf-a6fc-6d6a94793eeb)

📋Matrix de Confusão

![Rplot_Cor_Matrix](https://github.com/user-attachments/assets/25844a88-5898-4ec1-9733-76000b01133e)


📊 Taxa de Default por Score

Foi analisada a taxa de inadimplência por decis de score.
Score baixo → alto risco
Score alto → baixo risco
O modelo apresentou comportamento monotônico consistente.

🏦 Scorecard

Com base na regressão logística foi construído um Scorecard tradicional, utilizado para atribuir pontos aos clientes.

| Variável         | Faixa     | Pontos |
| ---------------- | --------- | ------ |
| Outstanding_Debt | <1200     | +29    |
| Outstanding_Debt | 1500–2700 | -41    |

Score final: Score = Base Score + soma dos pontos

![Rplot_CreditScore](https://github.com/user-attachments/assets/58b4bebf-ba98-46ef-bed0-48a09a714d12)

🔄 Validação de Estabilidade — PSI

Foi calculado o Population Stability Index (PSI) para verificar se a distribuição do modelo permanece estável entre treino e teste.
PSI = 0.0003

| PSI      | estabilidade |
| -------- | ------------ |
| <0.1     | estável      |
| 0.1–0.25 | atenção      |

O modelo apresentou excelente estabilidade.

![Rplot_Distribuicao_Probabilidades_TreinovsTeste](https://github.com/user-attachments/assets/d21f5488-ad7b-406d-a49a-44531cf5065f)

📊 Estratégia de Crédito

Com base no score é possível definir políticas de aprovação.

🚀 Tecnologias Utilizadas

- R
- tidymodels
- scorecard
- dplyr
- ggplot2
- data.table

📌 Conclusão

Este projeto demonstra a construção completa de um modelo de credit scoring interpretável, utilizando técnicas amplamente aplicadas na indústria financeira.

Principais resultados:
- AUC: 0.80
- KS: 0.55
- PSI: 0.0003

O modelo apresenta boa capacidade discriminatória, estabilidade e interpretabilidade, sendo adequado para aplicações de avaliação de risco de crédito.
