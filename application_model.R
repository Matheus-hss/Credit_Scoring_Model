######### Credit Scoring Model (Application) #########
#Os modelos de Credit Scoring são ferramentas estatísticas usadas por bancos e 
#financeiras para avaliar o risco de crédito de um cliente — ou seja, 
#a probabilidade de essa pessoa pagar (ou não) um empréstimo.

####📌 1️⃣ Application Scoring (Score de Admissão):####
#É o modelo utilizado no momento do pedido de crédito, antes de o cliente 
#ter qualquer histórico com a instituição.

# Para que serve?
#   
#Decidir se o crédito deve ser aprovado ou recusado
# Definir o limite de crédito
# Determinar a taxa de juro
# 
# Como funciona?
#O modelo analisa dados fornecidos na candidatura, como:
#Idade
#Profissão
#Rendimento
#Situação laboral
#Histórico de crédito (ex: registros no Banco Central)
#Número de créditos existentes
#Essas variáveis são combinadas através de um modelo estatístico 
#(normalmente Regressão Logística, mas hoje também se usam técnicas de Machine Learning 
#como árvores de decisão ou redes neuronais).
#
# 👉 O resultado é um score numérico.
# Quanto mais alto o score, menor o risco.

####🧩 Estrutura Geral do Projeto####
# 🔍 1. Entendimento do Problema
# 
# Objetivo: construir um modelo preditivo de Application Score que estime a probabilidade de um cliente ser “good” ou “bad”.
# 
# Tipo de problema: Classificação binária.
# 
# Métrica alvo: Acurácia, AUC-ROC, Gini, matriz de confusão, KS …
# 
# 🧠 2. Etapas de Modelagem
# 2.1 📌 Carregamento dos Dados
# 
# Importar o dataset original.
# 
# Explorar as variáveis e seus tipos (numéricas, categóricas).
# 
# Verificar tamanho, atributos e desbalanceamento da classe alvo.
# 
# 2.2 🧹 Limpeza e Tratamento Inicial
# 
# Tarefas típicas:
#   
# Lidar com valores ausentes (missing values).
# 
# Padronizar nomes de colunas.
# 
# Transformar variáveis categóricas em fatores.
# 
# Detectar e tratar outliers em variáveis numéricas.
# 
# Realizar análise univariada (histogramas, boxplots).
# 
# 2.3 💡 Engenharia de Atributos (Feature Engineering)
# 
# Exemplos de técnicas:
#   
# Criar variáveis derivadas (ex: binning, categorias agrupadas).
# 
# Normalização ou padronização de variáveis numéricas.
# 
# One-hot encoding de variáveis categóricas.
# 
# Análise de correlação para reduzir multicolinearidade.
# 
# 2.4 🔄 Divisão Treino / Teste
# 
# Separar dados em treino (ex: 70-80%) e teste (20-30%).
# 
# Usar stratified sampling para manter a proporção da classe alvo.
# 
# 2.5 🧪 Treinamento dos Modelos
# 
# Comparar múltiplos algoritmos:
#   
# Regressão Logística (baseline)
# 
# Random Forest
# 
# Gradient Boosting (ex: XGBoost)
# 
# SVM
# 
# K-Nearest Neighbors
# 
# Documente:
#   
# Parâmetros usados
# 
# Estratégia de validação cruzada
# 
# Overfitting / underfitting
# 
# 2.6 📊 Avaliação de Métricas
# 
# Métricas para comparar modelos:
#   
# AUC-ROC
# 
# Acurácia
# 
# KS Statistic
# 
# Matriz de Confusão
# 
# Curva ROC
# 
# Ganho/Lift Chart
# 
# 2.7 📈 Interpretação e Explicabilidade
# 
# Enfatize aspectos importantes:
#   
# Importância das variáveis
# 
# Odds ratios (para regressão logística)
# 
# SHAP values (para modelos complexos)
# 
# Comentários sobre o que impacta mais o score
# 
# 🧾 3. Documentação e Comunicação
# 📋 README.md
# 
# Deve conter:
#   
# ✔ Objetivos do projeto
# ✔ Descrição do dataset
# ✔ Metodologia usada
# ✔ Resultados e métricas
# ✔ Como rodar o código
# 
# 📊 Relatório Final (PDF ou HTML)
# 
# Use R Markdown ou Quarto para compor um relatório com:
#   
# Descrição do negócio (o que é scoring)
# 
# Gráficos de distribuição e variáveis
# 
# Curvas de performance
# 
# Explicação de decisões técnicas
# 
# Insights do modelo
# 
# 🧪 4. Considerações Técnicas Importantes
# 🔁 Validação Cruzada
# 
# Use k-fold cross validation para garantir robustez
# 
# ⚖️ Tratamento do Desbalanceamento
# 
# Se for necessário:
#   
# Oversampling (SMOTE)
# 
# Undersampling
# 
# Ajuste de threshold
# 
# 🔍 Métricas além da Acurácia
# 
# Porque, em scoring, o foco é capturar bons e maus clientes, não apenas acertar maior número de casos:
#   
# AUC-ROC
# 
# KS
# 
# Lift


#----------------------------------------------------------------------------#

#### Pacotes e bibliotecas ####
library(tidyverse)
library(writexl)
library(stringr)
library(scorecard)
library(Information)
library(woeBinning)
library(caret)
library(pROC)
library(corrplot)
library(tidymodels)
library(data.table)

#### Carregamento dos Dados e Tratamento  ####

test_base <- read_csv("C:/Users/m-hen/OneDrive/Área de Trabalho/Modelagem_Econométrica/application_scoring/test.csv")
train_base <- read_csv("C:/Users/m-hen/OneDrive/Área de Trabalho/Modelagem_Econométrica/application_scoring/train.csv")

# Entendimento dos Dados train_base

#Estrutura da Base
str(train_base)

#Dimensão da base
dim(train_base)

#Nome das Colunas
colnames(train_base)

#Primeiras e ultimas linhas
head(train_base)
tail(train_base)

#Tipo de Variaveis
#Verificar classe de variaveis
sapply(train_base, class)

#Muitas variaveis irrelevantes e com tipagem errada. Antes de continuar vamos organizar

#Removendo colunas indesejadas
train_base <- train_base |> 
   select(-ID,
          -Customer_ID,
          -Name,
          -SSN,
          -Month,
          -Type_of_Loan,
          -Credit_Mix,
          -Total_EMI_per_month,
          -Monthly_Balance,
          -Delay_from_due_date,
          -Num_of_Delayed_Payment,
          -Payment_of_Min_Amount,
          -Payment_Behaviour,
          -Changed_Credit_Limit,
          -Num_Credit_Inquiries)

#Colunas que sobraram
glimpse(train_base)
colnames(train_base)
# [1] "Age"                      "Occupation"               "Annual_Income"            "Monthly_Inhand_Salary"   
# [5] "Num_Bank_Accounts"        "Num_Credit_Card"          "Interest_Rate"            "Num_of_Loan"             
# [9] "Outstanding_Debt"         "Credit_Utilization_Ratio" "Credit_History_Age"       "Amount_invested_monthly" 
# [13] "Credit_Score"

#Corrigir Classes Erradas
# Agora vamos tratar as variáveis que deveriam ser numéricas mas estão como character.
# Problemas identificados
# Algumas possuem "_" misturado com números.
train_base <- train_base |> 
  mutate(across(
    c(Age, 
      Annual_Income, 
      Monthly_Inhand_Salary,
      Num_Bank_Accounts, 
      Num_Credit_Card, 
      Interest_Rate,
      Num_of_Loan,
      Outstanding_Debt,
      Credit_Utilization_Ratio,
      Amount_invested_monthly),
    ~ gsub("_", "", .)
  ))
#Transformar em numérico
train_base <- train_base |> 
  mutate(across(
    c(Age, 
      Annual_Income, 
      Monthly_Inhand_Salary,
      Num_Bank_Accounts, 
      Num_Credit_Card, 
      Interest_Rate,
      Num_of_Loan,
      Outstanding_Debt,
      Credit_Utilization_Ratio,
      Amount_invested_monthly),
    as.numeric
  ))
#Corrigir variaveis categóricas
train_base <- train_base |> 
  mutate(
    Occupation = as.factor(Occupation),
    Credit_History_Age = as.factor(Credit_History_Age),
    Credit_Score = as.factor(Credit_Score)
  )

#Estatisticas Descritivas 
#Variaveis Numéricas
train_base |> 
  select(where(is.numeric)) |> 
  summary()

#Tratamento das variaveis numéricas
#1) Age
# 📊 Problema observado:
# Min: -500
# Max: 8698
# Média: 110.6 (claramente distorcida)
#Vamos substituir idades menores que 18 ou maiores que 100 por NA
train_base <- train_base |> 
  mutate(Age = ifelse(Age < 18 | Age > 100, NA, Age))
summary(train_base$Age)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 18.00   26.00   34.00   34.43   42.00  100.00    8482
#Agora vamos tratar os NA´s 
# Em Credit Scoring temos 3 abordagens principais:
# 🔹 Imputar pela mediana
# 
# 🔹 Imputar pela média
# 
# 🔹 Criar flag de missing

#Como nosso modelo BaseLine vai ser o de regressão logistica, vamos usar mediana
#Criar variavel indicadora de ausência de idade
# 1 = idade ausente
# 0 = idade presente
train_base <- train_base |>
  mutate(age_missing = ifelse(is.na(Age), 1, 0))
#Imputar valores ausentes pela mediana
#Mediana foi escolhida por ser robusta a outliers
median_age <- median(train_base$Age, na.rm = TRUE)

train_base <- train_base |> 
  mutate(Age = ifelse(is.na(Age), median_age, Age))

#2) Annual_Income
#📊 Problema observado:
# Max: 24.198.062
# Média: 176.416 (muito acima da mediana 37.579)
# Há forte presença de outliers extremos.
# Não basta remover ≤ 0.
# Precisamos tratar extremos. Vamos aplicar Winsorização no percentil 99%
#Definir percentil 1% a 99%
p1 <- quantile(train_base$Annual_Income, 0.01, na.rm = TRUE)
p99 <- quantile(train_base$Annual_Income, 0.99, na.rm = TRUE)

#Limitar os valores aos extremos aceitaveis
train_base <- train_base |> 
  mutate(Annual_Income = 
           ifelse(Annual_Income < p1, p1, 
           ifelse(Annual_Income > p99, p99,
           Annual_Income)))
summary(train_base$Annual_Income)
sum(is.na(train_base$Annual_Income))

#3)Monthly_Inhand_Salary
# NA's: 15.002
# Max: 15.204 (plausível)
#Imputar salario mensal usando renda anual dividida por 12
train_base <- train_base |> 
  mutate(Monthly_Inhand_Salary = 
           ifelse(is.na(Monthly_Inhand_Salary),
                  Annual_Income/12,
                  Monthly_Inhand_Salary))
summary(train_base$Monthly_Inhand_Salary)

#4) Num_Bank_Accounts
# 📊 Problema:
# Min: -1
# Max: 1798
# Média: 17 (claramente distorcida)
#Remover valores negativos ou acima de 20 contas bancárias
train_base <- train_base |> 
  mutate(Num_Bank_Accounts = 
           ifelse(Num_Bank_Accounts < 0 | Num_Bank_Accounts > 20,
                  NA, Num_Bank_Accounts))
summary(train_base$Num_Bank_Accounts)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.000   3.000   5.000   5.369   7.000  18.000    1335
#Vamos tratar os NA´s usando a mediana
#Criar variavel indicadora de ausência
# 1 = valor ausente
# 0 = valor presente
train_base <- train_base |> 
  mutate(Num_Bank_Accounts_missing = 
           ifelse(is.na(Num_Bank_Accounts), 1, 0))

#Calcula mediana 
median_accounts <- median(train_base$Num_Bank_Accounts, na.rm = TRUE)

#Substitui NA pela mediana
train_base <- train_base |> 
  mutate(Num_Bank_Accounts = 
           ifelse(is.na(Num_Bank_Accounts), median_accounts, Num_Bank_Accounts))
sum(is.na(train_base$Num_Bank_Accounts))

#5) Num_Credit_Card
# 📊 Problema:
# Max: 1499
# Média: 22 (claramente distorcida)
#Remover valores impossiveis para numeros de cartões
train_base <- train_base |> 
  mutate(Num_Credit_Card = 
           ifelse(Num_Credit_Card < 0 | Num_Credit_Card > 20,
                  NA, Num_Credit_Card))
summary(train_base$Num_Credit_Card)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.000   4.000   5.000   5.534   7.000  20.000    2263
# Cria variável indicadora de ausência
# 1 = valor ausente
# 0 = valor observado
train_base <- train_base |>
  mutate(Num_Credit_Card_missing =
           ifelse(is.na(Num_Credit_Card), 1, 0))

# Calcula mediana da variável (robusta a outliers)
median_cards <- median(train_base$Num_Credit_Card, na.rm = TRUE)

# Substitui NA pela mediana
train_base <- train_base |>
  mutate(Num_Credit_Card =
           ifelse(is.na(Num_Credit_Card),
                  median_cards,
                  Num_Credit_Card))
sum(is.na(train_base$Num_Credit_Card))

#6) Interest_Rate
# 📊 Problema grave:
# Max: 5797%
# Média: 72%
# Mediana: 13% (muitos outliers)
# Remove taxas de juros negativas ou acima de 60%
train_base <- train_base |>
  mutate(Interest_Rate =
           ifelse(Interest_Rate < 0 |
                    Interest_Rate > 60,
                  NA,
                  Interest_Rate))
summary(train_base$Interest_Rate)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 1.00    7.00   13.00   14.53   20.00   60.00    2032
#Tratar NA´s na variavel Interes_Rate
# Cria variável indicadora de ausência
# 1 = valor ausente
# 0 = valor observado
train_base <- train_base |>
  mutate(Interest_Rate_missing =
           ifelse(is.na(Interest_Rate), 1, 0))

# Calcula mediana da variável (robusta a outliers)
median_interest <- median(train_base$Interest_Rate, na.rm = TRUE)

# Substitui NA pela mediana
train_base <- train_base |>
  mutate(Interest_Rate =
           ifelse(is.na(Interest_Rate),
                  median_interest,
                  Interest_Rate))
sum(is.na(train_base$Interest_Rate))

#7) Num_of_Loan
# 📊 Problema:
# Min: -100
# Max: 1496
# Remove valores negativos ou excessivos de empréstimos
train_base <- train_base |>
  mutate(Num_of_Loan =
           ifelse(Num_of_Loan < 0 |
                    Num_of_Loan > 20,
                  NA,
                  Num_of_Loan))
summary(train_base$Num_of_Loan)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 0.000   2.000   3.000   3.534   5.000  19.000    4345
#Tratar NA´s na variavel Num_of_Loan
# Cria variável indicadora de ausência
# 1 = valor ausente
# 0 = valor observado
train_base <- train_base |>
  mutate(Num_of_Loan_missing =
           ifelse(is.na(Num_of_Loan), 1, 0))

# Calcula mediana da variável (robusta a outliers)
median_loan <- median(train_base$Num_of_Loan, na.rm = TRUE)

# Substitui NA pela mediana
train_base <- train_base |>
  mutate(Num_of_Loan =
           ifelse(is.na(Num_of_Loan),
                  median_loan,
                  Num_of_Loan))
sum(is.na(train_base$Num_of_Loan))

#8) Outstanding_Debt
# 📊 Problema:
# Min: 0.23
# Max: 4998
# Parece plausível
#Vamos apenas garantir que não existam dividas negativas
# Remove dívidas negativas
train_base <- train_base |>
  mutate(Outstanding_Debt =
           ifelse(Outstanding_Debt < 0,
                  NA,
                  Outstanding_Debt))
summary(train_base$Outstanding_Debt)

#9) Credit_Utilization_Ratio
#Variavel Ok✔

#10) Amount_invested_monthly
# 📊 Problema:
# Max: 10.000
# Média: 637
# Mediana: 135
# Há assimetria forte → outliers.
# Winsorização no percentil 99% para reduzir impacto de outliers
p99_inv <- quantile(train_base$Amount_invested_monthly, 0.99, na.rm = TRUE)

train_base <- train_base |>
  mutate(Amount_invested_monthly =
           ifelse(Amount_invested_monthly > p99_inv,
                  p99_inv,
                  Amount_invested_monthly))
summary(train_base$Amount_invested_monthly)
# Min.  1st Qu.   Median     Mean  3rd Qu.     Max.     NA's 
# 0.00    74.53   135.93   637.41   265.73 10000.00     4479 
#Tratar NA´s na variavel Amount_invested_monthly
# Cria variável indicadora de ausência
# 1 = valor ausente
# 0 = valor observado
train_base <- train_base |>
  mutate(Amount_invested_monthly_missing =
           ifelse(is.na(Amount_invested_monthly), 1, 0))

# Calcula mediana da variável (robusta a outliers)
median_invested <- median(train_base$Amount_invested_monthly, na.rm = TRUE)

# Substitui NA pela mediana
train_base <- train_base |>
  mutate(Amount_invested_monthly =
           ifelse(is.na(Amount_invested_monthly),
                  median_invested,
                  Amount_invested_monthly))
sum(is.na(train_base$Amount_invested_monthly))

#No processo de ajuste de algumas variaveis numéricas foram criadas variáveis 
#indicadoras de ausência de informação, prática comum em modelos de risco de crédito
# Age_missing → 8,48%
# 
# Num_Bank_Accounts_missing → 1,33%
# 
# Num_Credit_Card_missing → 2,26%
# 
# Interest_Rate_missing → 2,03%
# 
# Num_of_Loan_missing → 4,34%
# 
# Amount_invested_monthly_missing → 4,47%

#Flags de missing frequentemente carregam informação preditiva, e a ausência 
#de informação pode indicar perfil de risco. Vamos rodar os modelos preditivos
#com e sem as flags e comparar o ganho de informação. Somente vamos remover as flags
#quando coeficiente não for significativo, não melhorar performance ou gerar multicolinearidade.

#Agora vamos analisar as variaveis categóricas

#Variaveis Categórias
train_base |> 
  select(where(is.factor)) |> 
  summary()

#1) Occupation
# 📊 O que o summary mostra
# Muitas categorias
# Categoria inválida: "_______" (7062 casos)
# Grande dispersão (61 mil registros em "Other")
# Provavelmente variável de alta cardinalidade
#Primeiro vamos tratar a categoria invalida
train_base <- train_base |> 
  mutate(Occupation = na_if(Occupation, "_______"))
summary(train_base$Occupation)
#Agora vamos agrupar as categorias raras. Categorias com menos de 2% ou 3% da base 
#vão ser agrupadas em “Rare_Occupations”. Vamos usar a blibioteca Forcats (parte do tidyverse)
train_base <- train_base |> 
  mutate(Occupation = 
           fct_lump_prop(Occupation, prop = 0.03, other_level = "Rare_Occupations"))
# Accountant        Architect        Developer      Doctor         Engineer     Entrepreneur       Journalist 
# 6271             6355             6235             6087             6350             6174             6085 
# Lawyer          Manager         Mechanic    Media_Manager         Musician        Scientist          Teacher 
# 6575             5973             6291             6232             5911             6299             6215 
# Writer        Rare_Occupations    NA's 
# 5885                0             7062
#Ficamos com 7062 NA´s
#Temos 3 formas de abordar o problema:
# | Estratégia                | Quando usar                         | Recomendação aqui |
# | ------------------------- | ----------------------------------- | ----------------- |
# | Remover linhas            | Missing muito pequeno (<1%)         | ❌ Não            |
# | Imputar pela moda         | Quando ausência é aleatória         | ⚠️ Não ideal      |
# | Criar categoria "Unknown" | Quando ausência pode carregar risco | ✅ Melhor opção   |

#Usar "unkown" permite que o modelo aprenda se ausência de ocupação está associada a maior risco
train_base <- train_base |>
  mutate(
    Occupation =
      fct_na_value_to_level(Occupation,
                            level = "Unknown")
  )
#Valores ausentes em variáveis categóricas foram convertidos em nível 
#explícito (‘Unknown’) utilizando forcats::fct_na_value_to_level(), permitindo 
#captura de possível sinal preditivo.

#2) Credit_History_Age
# 📊 O que o summary mostra
# Centenas de categorias diferentes
# 9.030 NA's
# Cada combinação de "X Years and Y Months" vira um nível diferente
# 🚨 Problema sério: variável categórica com alta cardinalidade artificial.

#A melhor forma de tratar essa variavel sera transforma-la em numérica continua (meses)
# 🧠 Exemplo de Conversão
# "22 Years and 1 Months"
# 22 anos = 22 × 12 = 264 meses
# 1 mês
# = 265 meses
#Converter para character pois vamos tratar o texto, variaveis factor no R
#são variaveis com alguma forma de ordenamento, como "alto", "baixo", "médio", o que
#não é o caso dessa variavel.
train_base <- train_base |> 
  mutate(Credit_History_Age = as.character(Credit_History_Age))

#Agora extraimos anos dos meses. Vamos usar stringr
train_base <- train_base |> 
  mutate(
    #Extrai o número que aparece antes de "Years"
    Credit_Years = as.numeric(str_extract(Credit_History_Age, "^[0-9]+")),
    #Extrai o número que aparece antes de "Months"
    Credit_Months = as.numeric(str_extract(Credit_History_Age, "(?<=and )[0-9]+"))
  )
# "^[0-9]+" → pega o número no início da string (anos)
# 
# "(?<=and )[0-9]+" → pega número depois de "and " (meses)
#Depois criamos a variavel total em meses
train_base <- train_base |>
  mutate(
    Credit_History_Total_Months =
      Credit_Years * 12 + Credit_Months
  )
summary(train_base$Credit_History_Total_Months)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
# 1.0   144.0   219.0   221.2   302.0   404.0    9030

#Por fim tratamos os NA´s 
# Flag de missing
train_base <- train_base |>
  mutate(Credit_History_missing =
           ifelse(is.na(Credit_History_Total_Months), 1, 0))

# Imputação pela mediana
median_ch <- median(train_base$Credit_History_Total_Months, na.rm = TRUE)

train_base <- train_base |>
  mutate(
    Credit_History_Total_Months =
      ifelse(is.na(Credit_History_Total_Months),
             median_ch,
             Credit_History_Total_Months)
  )
#removemos as variaveis auxiliares
train_base <- train_base |>
  select(-Credit_History_Age,
         -Credit_Years,
         -Credit_Months)

#3) Credit_Score (Target)
# 📊 Distribuição:
# Good: 17.828 (~18%)
# Standard: 53.174 (~53%)
# Poor: 28.998 (~29%)
# Levemente desbalanceada, mas não extremo.
summary(train_base$Credit_Score)
#Para Application Scoring realista, vamos fazer uma pequena transformação binária:

# Good + Standard → 0 (Bom)
# 
# Poor → 1 (Default)
train_base <- train_base |> 
  mutate(Default = ifelse(Credit_Score == "Poor", 1, 0))

#Valores Ausentes (confirmar)
#Quantidade de NA por coluna
colSums(is.na(train_base))

#Nova Estrutura
str(train_base)
colnames(train_base)
head(train_base, n = 20)

#Ajustes antes de continuar
#a) Variavel Default (Target) como numérica, é melhor transformar em fator
train_base <- train_base |>
  mutate(Default = as.factor(Default))
#b) Remover Credit_Score
train_base <- train_base |>
  select(-Credit_Score)

#### Feature Engineering ####
# Em modelos tradicionais de crédito (scorecard):
#   
# Não usamos variáveis “cruas” direto.
# 
# Trabalhamos com:
#   
# Razões financeiras
# 
# Transformações
# 
# Binning supervisionado
# 
# WOE encoding
# 
# ⚠️ Muito importante:
#   Feature Engineering vem antes do WOE, mas algumas transformações são substituídas pelo binning supervisionado

#1) Criar razões financeiras
#🔥 Quanto maior, maior risco potencial.
# Debt-to-Income Ratio.
train_base <- train_base |> 
  mutate(
    Debt_to_Income = Outstanding_Debt/Annual_Income
  )

# Credit Exposure Ratio: Coloquei +1 para evitar divisão por zero
train_base <- train_base |> 
  mutate(
    Card_per_Bank = Num_Credit_Card/(Num_Bank_Accounts+1)
  )

#Loan Intensity
train_base <- train_base |> 
  mutate(
    Loans_per_Income = Num_of_Loan/(Annual_Income+1)
  )

#Investimento relativo a renda (pode indicar tambem nivel de poupança)
train_base <- train_base |> 
  mutate(
    Investiment_to_Income = Amount_invested_monthly / (Monthly_Inhand_Salary + 1)
  )

#2) Variaveis no Tempo
#Histórico de crédito é crucial
train_base <- train_base |> 
  mutate(
    Credit_History_Years = Credit_History_Total_Months/12
  )

#3) Flags Estratégicas
train_base <- train_base |> 
  mutate(
    High_Utilization = ifelse(Credit_Utilization_Ratio > 40, 1, 0),
    Young_Client = ifelse(Age <25, 1, 0)
  )
#Essas flags ajudam os modelos lineares.

#4) Preparação do Weight of Evidence (WOE) e Information Value (IV)
# Antes de aplicar WOE:

# ✔ Target deve ser binária (0/1)
# ✔ Sem NA
# ✔ Variáveis categóricas como factor
# ✔ Numéricas prontas para binning
table(train_base$Default)
str(train_base)

# | IV       | Força preditiva        |
# | -------- | ---------------------- |
# | < 0.02   | Não útil               |
# | 0.02–0.1 | Fraco                  |
# | 0.1–0.3  | Médio                  |
# | 0.3–0.5  | Forte                  |
# | > 0.5    | Suspeito (overfitting) |

#5) Binning Automático com scorecard
# Criar bins supervisionados usando função woebin()
# O que acontece internamente?
# 🔹 Etapa A — Separação de Bons e Maus
# Ele identifica:
#   
# Bons = Default = 0
# 
# Maus = Default = 1
# 
# Calcula:
# % de bons em cada grupo
# 
# % de maus em cada grupo
# 
# 🔹 Etapa B — Binning Inicial
# Para variáveis numéricas:
#   
# Cria cortes automáticos
# 
# Pode usar quantis ou método tipo árvore (Chi-square merging)
# 
# Para variáveis categóricas:
#   
# Agrupa categorias com comportamento semelhante de risco
# 
# 🔹 Etapa C — Cálculo do WOE
# Para cada bin:
#   
# WOE = ln(%Good/%Bad)
# 
# Se:
#   
# WOE > 0 → mais bons que maus
# 
# WOE < 0 → mais maus que bons
# 
# Isso transforma qualquer variável em escala log-odds.
# 
# 🔹 Etapa D — Cálculo do IV
# 
# IV = \(\Sigma \)(%Good - %Bad) X WOE
# Isso mede poder preditivo da variável.
# 
# O que é retornado em bins?
#   
# bins é uma lista com uma tabela para cada variável.
# bins$Age
# | bin | count | good | bad | badprob | woe | iv |
# | --- | ----- | ---- | --- | ------- | --- | -- |
# Cada linha representa um intervalo
# 
# WOE Ajuda Logistic Regression pois ja etara na escala log-odds
# ✔ Relação linear com logit
# ✔ Estabilidade
# ✔ Interpretação simples
# ✔ Robustez contra outliers
# 
# Tratamento Automático de Missing
# 
# Se variável tem NA:
#   
# woebin() cria bin separado para missing
# 
# Calcula WOE específico para missing
# 
# Não perde informação
# 
# Monotonicidade
# 
# Modelos tradicionais exigem que:
#   
# À medida que variável aumenta, o risco aumente ou diminua monotonicamente.
# 
# woebin() tenta ajustar bins para respeitar essa regra.

#A função pode ser rodada sem o ajuste de parametros mas nesse projeto vamos ajustar
bins <- woebin(
    train_base,
    y = "Default",
    min_perc_fine_bin = 0.02,
    min_perc_coarse_bin = 0.05,
    stop_limit = 0.1
)
#🔍 1️⃣ min_perc_fine_bin: min_perc_fine_bin = 0.02
# 📌 O que significa?
# Percentual mínimo da base que cada bin inicial pode ter.
# Com 100.000 registros:
# 2% = 2.000 observações por bin mínimo.
#Se for muito pequeno pode gerar problemas de overfitting e sensibilidade a pequenas variações
#Se for muito grande gera pouca granularidade e perde poder discriminatório
# | Tamanho da base | min_perc_fine_bin |
# | --------------- | ----------------- |
# | < 10k           | 5%                |
# | 10k – 100k      | 2–5%              |
# | > 100k          | 1–2%              |

#🔍 2️⃣ min_perc_coarse_bin:  min_perc_coarse_bin = 0.05
# Esse parâmetro controla o tamanho mínimo do bin após fusão.

# 🎯 Diferença entre fine e coarse

# Fine binning → cortes iniciais

# Coarse binning → agrupamentos finais

# O algoritmo:
# Cria bins pequenos (fine)
# Junta bins semelhantes (coarse)
# 📌 5% significa:

# Cada bin final terá pelo menos 5% da base (~5.000 registros). Isso aumenta estabilidade.
# Se for muito pequeno: Bins finais com 1% da base → instabilidade

# Se for muito grande: Pode reduzir muito o poder preditivo

#🔍 3️⃣ stop_limit:stop_limit = 0.1
# Esse é o parâmetro mais técnico. Ele controla o critério de parada na fusão de bins.

# 🎯 Como funciona?
 
# Durante o binning:
 
# O algoritmo calcula estatística de separação (geralmente chi-square).

# Ele vai fundindo bins enquanto:

# A diferença entre eles for pequena ou enquanto o ganho informacional for baixo

# stop_limit define o limiar mínimo de ganho.
# | stop_limit | Comportamento                |
# | ---------- | ---------------------------- |
# | 0.01       | Muito permissivo (mais bins) |
# | 0.05       | Moderado                     |
# | 0.1        | Conservador                  |
# | 0.2        | Muito conservador            |

#Analise do output do processo de agrupamento usando WoeBin
# 🎯 1️⃣ Análise Geral: Overfitting?
#   
# ✔ Todos os bins respeitam min_perc_coarse_bin = 5%
# ✔ Nenhum bin extremamente pequeno
# ✔ WOE relativamente suave
# ✔ Não há bins com 1% ou menos
# 
# 👉 Conclusão: Parâmetros foram adequados.
# 
# 📊 2️⃣ Análise por IV (Information Value)
# 🔥 Muito fortes (IV > 0.5)
# | Variável                    | IV    | Interpretação      |
# | --------------------------- | ----- | ------------------ |
# | Outstanding_Debt            | 1.327 | Extremamente forte |
# | Interest_Rate               | 1.017 | Extremamente forte |
# | Debt_to_Income              | 0.742 | Muito forte        |
# | Num_Credit_Card             | 0.622 | Muito forte        |
# | Credit_History_Years        | 0.617 | Muito forte        |
# | Credit_History_Total_Months | 0.612 | Muito forte        |
# | Num_of_Loan                 | 0.579 | Muito forte        |
# | Num_Bank_Accounts           | 0.519 | Muito forte        |

# 🟡 Médio (0.1 – 0.3)
# | Variável              | IV                               |
# | --------------------- | -------------------------------- |
# | Annual_Income         | 0.278                            |
# | Monthly_Inhand_Salary | 0.219                            |
# | Loans_per_Income      | 0.467 (forte, quase muito forte) |
# | Card_per_Bank         | 0.157                            |

# 🔵 Fraco (< 0.1)
# | Variável                 | IV    |
# | ------------------------ | ----- |
# | Occupation               | 0.003 |
# | Credit_Utilization_Ratio | 0.011 |
# | Amount_invested_monthly  | 0.042 |
# | age_missing              | 0.022 |
# | High_Utilization         | 0.007 |
# | Young_Client             | 0.001 |
# | Credit_History_missing   | ~0    |
# | Missing flags (vários)   | 0     |

# 🚨 3️⃣ Variáveis que Devem Ser Removidas
# ❌ Remover (IV ≈ 0)
# Num_Bank_Accounts_missing
# Num_Credit_Card_missing
# Interest_Rate_missing
# Num_of_Loan_missing
# Amount_invested_monthly_missing
# Credit_History_missing
# Young_Client
# Occupation (IV muito baixo)
# 
# Não agregam poder preditivo.

#Algumas variaveis ficaram redundantes
# Credit_History_Total_Months
# 
# Credit_History_Years
# 
# São praticamente a mesma informação. Vou retirar uma
  
# Seleção de variaveis com poder preditivo IV > 0.02

#Extrair IV do objeto bins
# O IV está dentro de cada elemento da lista bins

#Extrair IV de cada variavel
iv_values <- sapply(bins, function(x) unique(x$total_iv))
#Essa função faz: iv_values["Age"] <- unique(bins$Age$total_iv) para cada data frame dentro da lista bins

#Transformar em data frame
iv_table <- data.frame(
  variable = names(iv_values),
  IV = iv_values
)

#Selecionar variáveis com IV > 0.02
select_variaveis <- iv_table$variable[iv_table$IV > 0.02]
select_variaveis

#Criar novo dataset apenas com as selecionadas
train_selected <- train_base[, c(select_variaveis, "Default")]
str(train_selected)

#Remover variavel redundante
train_selected <- train_selected |>
  select(-Credit_History_Years)

#Criar lista bins_selected apenas com as variaveis selecionadas
bins_selected <- bins[names(bins) %in% colnames(train_selected)]

#Tranformação WOE
train_woe <- woebin_ply(train_selected, bins_selected)
# 🔹 O que é woebin_ply()?
#   
#  Função do pacote scorecard.
# 
# Ela:
#   
# ✔ Usa os bins já calculados
# ✔ Substitui cada valor pelo WOE correspondente
# ✔ Retorna novo dataset
# 
# 🔹 Como funciona internamente?
#   
# Para cada variável:
#   
# Verifica em qual bin o valor cai
# 
# Substitui pelo WOE daquele bin
# 
# Exemplo:
#   
# Se:
#   
# Age = 23
# E bin <35 tem WOE = 0.15
# 
# Então:
#   
# Age vira 0.15

# O que acontece com o dataset?
#   
# Antes:
# | Age | Income | Loans | Default |
# | --- | ------ | ----- | ------- |
# 
# Depois:
#   
# | Age_woe | Income_woe | Loans_woe | Default |
#   
# Todas as variáveis agora estão:
#   
# ✔ Na mesma escala
# ✔ Suavizadas
# ✔ Robustas a outliers
# ✔ Interpretáveis
# 
# Sem WOE:
#   
# Relação pode ser não linear
# 
# Outliers influenciam
# 
# Modelo menos estável
# 
# Com WOE:
#  
# Relação quase linear com logit
# 
# Bins estabilizam extremos
# 
# Muito usado em bancos
#Resumindo: Pegue meus dados e substitua cada valor pelo risco estatístico do grupo ao qual ele pertence.
summary(train_woe)

#Analise da base transformada
# Todos os preditores agora estão:
#   
# ✔ Em escala WOE
# ✔ Com valores discretos (um por bin)
# ✔ Sem NA
# ✔ Prontos para regressão logística
# 
# Distribuição do Target:
# Default
# 0: 71002
# 1: 28998
# 
# Base levemente desbalanceada (~29% inadimplentes)
# 
# Age_woe
# Min: -1.04
# Max: 0.15
# Faixa com maior risco → WOE negativo grande
# Faixa com menor risco → WOE positivo
# 
# Outstanding_Debt_woe
# Min: -1.23
# Max: 1.74
# Quanto maior a dívida → maior WOE positivo → maior risco (alta amplitude, muito forte)
# 
# Investiment_to_Income_woe
# Min: -0.26
# Max: 0.09
# Provavelmente terá pouco impacto no modelo (baixa amplitude, fraca)

#Analise de Correlação (tiramos a variavel alvo)

#Matrix de correlação
cor_matrix <- cor(train_woe[, -1])
cor_matrix

#Visualização
corrplot(cor_matrix, method = "number", type = "upper")

#Vamos retirar variaveis com correlação forte (cor > 0.7)
# ✅ 1) Annual_Income_woe ↔ Monthly_Inhand_Salary_woe
# cor = 0.87
# Muito alta.
# ✔ Mesma informação econômica (renda anual vs mensal)
# 
# ✅ 2) Interest_Rate_woe ↔ Outstanding_Debt_woe
# cor = 0.71
# Alta.
# Pode indicar que taxa de juros reflete risco já capturado pela dívida.
# 
# ✅ 3) Num_of_Loan_woe ↔ Loans_per_Income_woe
# cor = 0.72
# Alta.
# Loans_per_Income é basicamente Num_of_Loan ajustado por renda.
# 
# ⚠️ 4) Outstanding_Debt_woe ↔ Debt_to_Income_woe
# cor = 0.70
# Limite exato.
# Não ultrapassa 0.7, mas está muito próximo.
# 
# ⚠️ 5) Debt_to_Income_woe ↔ Loans_per_Income_woe
# cor = 0.70
# Também limítrofe.

#Agora precisamos escolher qual manter em cada par

# 🔹 Par 1: Income vs Salary
# Recomendação:
# ✔ Manter Annual_Income_woe
# ❌ Remover Monthly_Inhand_Salary_woe
# Motivo:
# Mais tradicional em scorecard
# Evita duplicidade
# 
# 🔹 Par 2: Interest_Rate vs Outstanding_Debt
# Ambas têm IV alto.
# Outstanding_Debt → 1.32
# Interest_Rate → 1.01
# Recomendação:
# ✔ Manter Outstanding_Debt_woe
# ❌ Remover Interest_Rate_woe
# Motivo:
# Dívida é variável mais interpretável
# 
# 🔹 Par 3: Num_of_Loan vs Loans_per_Income
# IV:
# Num_of_Loan → 0.57
# Loans_per_Income → 0.46
# Recomendação:
# ✔ Manter Num_of_Loan_woe
# ❌ Remover Loans_per_Income_woe
# Motivo:
# Mais simples
# Menos dependente de outra variável

#Remover variaveis
train_woe_reduced <- train_woe |>
  dplyr::select(
    -Monthly_Inhand_Salary_woe,
    -Interest_Rate_woe,
    -Loans_per_Income_woe,
    -Debt_to_Income_woe
  )
str(train_woe_reduced)
#Renomear 
base_treino <- train_woe_reduced

#Exportar Base final
write_xlsx(base_treino, path = "C:/Users/m-hen/OneDrive/Área de Trabalho/Modelagem_Econométrica/application_scoring/base_treino.xlsx")


#### Tratamento Base de Teste ####
#🔹 ETAPA 1 — Replicar tratamento inicial
# aplicar no test_base:
#   
# ✔ Remoção das mesmas colunas
# ✔ Conversão de tipos
# ✔ Tratamento de outliers
# ✔ Criação das mesmas flags
# ✔ Criação das mesmas variáveis derivadas

#Estrutura da Base
str(test_base)

#Dimensão da base
dim(test_base)

#Nome das Colunas
colnames(test_base)

#Primeiras e ultimas linhas
head(test_base)
tail(test_base)

#Tipo de Variaveis
#Verificar classe de variaveis
sapply(test_base, class)

#Remover colunas
test_base <- test_base |> 
  select(-ID,
         -Customer_ID,
         -Name,
         -SSN,
         -Month,
         -Type_of_Loan,
         -Credit_Mix,
         -Total_EMI_per_month,
         -Monthly_Balance,
         -Delay_from_due_date,
         -Num_of_Delayed_Payment,
         -Payment_of_Min_Amount,
         -Payment_Behaviour,
         -Changed_Credit_Limit,
         -Num_Credit_Inquiries)
colnames(test_base)
#Retirar caracteres
test_base <- test_base |> 
  mutate(across(
    c(Age, 
      Annual_Income, 
      Monthly_Inhand_Salary,
      Num_Bank_Accounts, 
      Num_Credit_Card, 
      Interest_Rate,
      Num_of_Loan,
      Outstanding_Debt,
      Credit_Utilization_Ratio,
      Amount_invested_monthly),
    ~ gsub("_", "", .)
  ))
#Transformar em numérico
test_base <- test_base |> 
  mutate(across(
    c(Age, 
      Annual_Income, 
      Monthly_Inhand_Salary,
      Num_Bank_Accounts, 
      Num_Credit_Card, 
      Interest_Rate,
      Num_of_Loan,
      Outstanding_Debt,
      Credit_Utilization_Ratio,
      Amount_invested_monthly),
    as.numeric
  ))
#Corrigir variaveis categóricas
test_base <- test_base |> 
  mutate(
    Occupation = as.factor(Occupation),
    Credit_History_Age = as.factor(Credit_History_Age)
  )
#Variaveis numéricas
test_base |> 
  select(where(is.numeric)) |> 
  summary()

#Variaveis numéricas com o mesmo problema do dataset treino
#Vamos fazer os ajustes
#1) Age
test_base <- test_base |> 
  mutate(Age = ifelse(Age < 18 | Age > 100, NA, Age))
summary(test_base$Age)

test_base <- test_base |>
  mutate(age_missing = ifelse(is.na(Age), 1, 0))

median_age <- median(train_base$Age, na.rm = TRUE)

test_base <- test_base |> 
  mutate(Age = ifelse(is.na(Age), median_age, Age))
#Age na base de teste ajustado

#2) Annual_Income

p1 <- quantile(test_base$Annual_Income, 0.01, na.rm = TRUE)
p99 <- quantile(test_base$Annual_Income, 0.99, na.rm = TRUE)

#Limitar os valores aos extremos aceitaveis
test_base <- test_base |> 
  mutate(Annual_Income = 
           ifelse(Annual_Income < p1, p1, 
                  ifelse(Annual_Income > p99, p99,
                         Annual_Income)))
summary(test_base$Annual_Income)
sum(is.na(train_base$Annual_Income))
#Annual_Income ajustado

#3)Monthly_Inhand_Salary
test_base <- test_base |> 
  mutate(Monthly_Inhand_Salary = 
           ifelse(is.na(Monthly_Inhand_Salary),
                  Annual_Income/12,
                  Monthly_Inhand_Salary))
summary(test_base$Monthly_Inhand_Salary)
# Monthly_Inhand_Salary ajustado

#4) Num_Bank_Accounts

test_base <- test_base |> 
  mutate(Num_Bank_Accounts = 
           ifelse(Num_Bank_Accounts < 0 | Num_Bank_Accounts > 20,
                  NA, Num_Bank_Accounts))
summary(test_base$Num_Bank_Accounts)

test_base <- test_base |> 
  mutate(Num_Bank_Accounts_missing = 
           ifelse(is.na(Num_Bank_Accounts), 1, 0))

median_accounts <- median(test_base$Num_Bank_Accounts, na.rm = TRUE)

test_base <- test_base |> 
  mutate(Num_Bank_Accounts = 
           ifelse(is.na(Num_Bank_Accounts), median_accounts, Num_Bank_Accounts))
sum(is.na(test_base$Num_Bank_Accounts))
#Num_Bank_Accounts ajustado

#5) Num_Credit_Card

test_base <- test_base |> 
  mutate(Num_Credit_Card = 
           ifelse(Num_Credit_Card < 0 | Num_Credit_Card > 20,
                  NA, Num_Credit_Card))
summary(test_base$Num_Credit_Card)

test_base <- test_base |>
  mutate(Num_Credit_Card_missing =
           ifelse(is.na(Num_Credit_Card), 1, 0))

median_cards <- median(test_base$Num_Credit_Card, na.rm = TRUE)

test_base <- test_base |>
  mutate(Num_Credit_Card =
           ifelse(is.na(Num_Credit_Card),
                  median_cards,
                  Num_Credit_Card))
sum(is.na(test_base$Num_Credit_Card))
#Num_Credit_Card ajustado


#6) Interest_Rate

test_base <- test_base |>
  mutate(Interest_Rate =
           ifelse(Interest_Rate < 0 |
                    Interest_Rate > 60,
                  NA,
                  Interest_Rate))
summary(test_base$Interest_Rate)

test_base <- test_base |>
  mutate(Interest_Rate_missing =
           ifelse(is.na(Interest_Rate), 1, 0))

median_interest <- median(test_base$Interest_Rate, na.rm = TRUE)

test_base <- test_base |>
  mutate(Interest_Rate =
           ifelse(is.na(Interest_Rate),
                  median_interest,
                  Interest_Rate))
sum(is.na(test_base$Interest_Rate))
#Interest_Rate ajustado

#7) Num_of_Loan

test_base <- test_base |>
  mutate(Num_of_Loan =
           ifelse(Num_of_Loan < 0 |
                    Num_of_Loan > 20,
                  NA,
                  Num_of_Loan))
summary(test_base$Num_of_Loan)

test_base <- test_base |>
  mutate(Num_of_Loan_missing =
           ifelse(is.na(Num_of_Loan), 1, 0))

median_loan <- median(test_base$Num_of_Loan, na.rm = TRUE)

test_base <- test_base |>
  mutate(Num_of_Loan =
           ifelse(is.na(Num_of_Loan),
                  median_loan,
                  Num_of_Loan))
sum(is.na(test_base$Num_of_Loan))
#Num_of_Loan ajustado

#8) Outstanding_Debt

test_base <- test_base |>
  mutate(Outstanding_Debt =
           ifelse(Outstanding_Debt < 0,
                  NA,
                  Outstanding_Debt))
summary(test_base$Outstanding_Debt)
#Outstanding_Debt ajustado

#9) Credit_Utilization_Ratio
#Já estava Ok✔

#10) Amount_invested_monthly

p99_inv <- quantile(test_base$Amount_invested_monthly, 0.99, na.rm = TRUE)

test_base <- test_base |>
  mutate(Amount_invested_monthly =
           ifelse(Amount_invested_monthly > p99_inv,
                  p99_inv,
                  Amount_invested_monthly))
summary(test_base$Amount_invested_monthly)

test_base <- test_base |>
  mutate(Amount_invested_monthly_missing =
           ifelse(is.na(Amount_invested_monthly), 1, 0))


median_invested <- median(test_base$Amount_invested_monthly, na.rm = TRUE)

test_base <- test_base |>
  mutate(Amount_invested_monthly =
           ifelse(is.na(Amount_invested_monthly),
                  median_invested,
                  Amount_invested_monthly))
sum(is.na(test_base$Amount_invested_monthly))
# Amount_invested_monthly ajustado

#Variaveis numéricas ajustadas agora vamos ajustar as categóricas

#Variaveis Categórias
test_base |> 
  select(where(is.factor)) |> 
  summary()

#1) Occupation

test_base <- test_base |> 
  mutate(Occupation = na_if(Occupation, "_______"))
summary(test_base$Occupation)

test_base <- test_base |> 
  mutate(Occupation = 
           fct_lump_prop(Occupation, prop = 0.03, other_level = "Rare_Occupations"))

test_base <- test_base |>
  mutate(
    Occupation =
      fct_na_value_to_level(Occupation,
                            level = "Unknown")
  )
# Occupation Ajustado

#2) Credit_History_Age

test_base <- test_base |> 
  mutate(Credit_History_Age = as.character(Credit_History_Age))

test_base <- test_base |> 
  mutate(
    #Extrai o número que aparece antes de "Years"
    Credit_Years = as.numeric(str_extract(Credit_History_Age, "^[0-9]+")),
    #Extrai o número que aparece antes de "Months"
    Credit_Months = as.numeric(str_extract(Credit_History_Age, "(?<=and )[0-9]+"))
  )


test_base <- test_base |>
  mutate(
    Credit_History_Total_Months =
      Credit_Years * 12 + Credit_Months
  )
summary(test_base$Credit_History_Total_Months)

test_base <- test_base |>
  mutate(Credit_History_missing =
           ifelse(is.na(Credit_History_Total_Months), 1, 0))

median_ch <- median(test_base$Credit_History_Total_Months, na.rm = TRUE)

test_base <- test_base |>
  mutate(
    Credit_History_Total_Months =
      ifelse(is.na(Credit_History_Total_Months),
             median_ch,
             Credit_History_Total_Months)
  )
#removemos as variaveis auxiliares
test_base <- test_base |>
  select(-Credit_History_Age,
         -Credit_Years,
         -Credit_Months)

#Base teste ajustada
str(test_base)
summary(test_base)

#Verificando os niveis das variaveis de Occupation nas duas bases
setdiff(levels(test_base$Occupation), levels(train_base$Occupation))
setdiff(levels(train_base$Occupation), levels(test_base$Occupation))
#mesmos niveis

# Feature Engineering na base de teste (criando as mesmas variaveis)

#1) Criando razões financeiras
#Debt_to_Income
test_base <- test_base |> 
  mutate(
    Debt_to_Income = Outstanding_Debt/Annual_Income
  )

# Credit Exposure Ratio
test_base <- test_base |> 
  mutate(
    Card_per_Bank = Num_Credit_Card/(Num_Bank_Accounts+1)
  )

#Loan Intensity
test_base <- test_base |> 
  mutate(
    Loans_per_Income = Num_of_Loan/(Annual_Income+1)
  )

#Investimento relativo a renda (pode indicar tambem nivel de poupança)
test_base <- test_base |> 
  mutate(
    Investiment_to_Income = Amount_invested_monthly / (Monthly_Inhand_Salary + 1)
  )

#2) Variaveis no Tempo
#Histórico de crédito é crucial
test_base <- test_base |> 
  mutate(
    Credit_History_Years = Credit_History_Total_Months/12
  )

#3) Flags Estratégicas
test_base <- test_base |> 
  mutate(
    High_Utilization = ifelse(Credit_Utilization_Ratio > 40, 1, 0),
    Young_Client = ifelse(Age <25, 1, 0)
  )

#4) Preparação do Weight of Evidence (WOE) e Information Value (IV)
# Antes de aplicar WOE:

# ✔ Sem NA
# ✔ Variáveis categóricas como factor
# ✔ Numéricas prontas para binning
str(test_base)
colnames(test_base)

#Selecionando as mesmas colunas
#Para não ter nenhum risco de diferença irei selecionar as mesmas colunas
#da base de treino

predictors <- setdiff(colnames(train_selected), "Default")

test_selected <- test_base[, predictors]
test_selected

#Criando os mesmos agrupamentos
test_woe <- woebin_ply(test_selected, bins_selected)

#Validando
str(test_woe)

#Retirar variaveis com correlação alta, mesmo que fizemos na base de treino

test_woe_reduced <- test_woe |>
  dplyr::select(
    -Monthly_Inhand_Salary_woe,
    -Interest_Rate_woe,
    -Loans_per_Income_woe,
    -Debt_to_Income_woe
  )

#Validando novamente
setdiff(colnames(train_woe_reduced)[-1], colnames(test_woe_reduced))
setdiff(colnames(test_woe_reduced), colnames(train_woe_reduced)[-1])

#Renomear 
base_teste <- test_woe_reduced

#Exportar Base final
write_xlsx(base_teste, path = "C:/Users/m-hen/OneDrive/Área de Trabalho/Modelagem_Econométrica/application_scoring/base_teste.xlsx")



#### Modelagem ####
#Usando a estrutura do tidymodels o script segue a seguinte lógica:
#Dados → recipe → Modelo → workflow → fit 
# recipe(): Como os dados devem ser preparados antes de ir para o modelo?
#Modelo: definimos qual estrutura de modelo será utilizada
# workflow(): conector entre o recipe e o modelo, "Use essa preparação + esse modelo"
#fit():estima coeficientes


#Criando recipe
rec_logit <- recipe(Default ~., data = base_treino)

# Explicação:Default ~ . → target contra todos os preditores
# Não precisamos de step_normalize()
# Não precisamos de step_dummy()
# Não precisamos de step_impute()

#Definindo Modelo (Baseline)
logit_spec <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")
# Explicação:
# logistic_reg() → modelo logístico
# set_engine("glm") → usa glm tradicional
# set_mode("classification") → problema binário

#Criar Workflow
wf_logit <- workflow() |> 
  add_recipe(rec_logit) |> 
  add_model(logit_spec)

#Ajustar o modelo com fit()
modelo_logit <- fit(wf_logit, data = base_treino)

#Ver coeficientes
modelo_logit |> 
  tidy()

# term                             estimate std.error statistic   p.value
# <chr>                               <dbl>     <dbl>     <dbl>     <dbl>
# 1 (Intercept)                     -0.898      0.00832   -108.     0        
# 2 Age_woe                          0.0498     0.0252    1.98    4.81e-  2
# 3 Annual_Income_woe               -0.000248   0.0193   -0.0128  9.90e-  1
# 4 Num_Bank_Accounts_woe            0.0958     0.0139    6.88    6.05e- 12
# 5 Num_Credit_Card_woe              0.312      0.0125    24.9    6.47e-137
# 6 Num_of_Loan_woe                  0.0504     0.0144    3.50    4.73e-  4
# 7 Outstanding_Debt_woe             0.807      0.00957   84.3    0        
# 8 Amount_invested_monthly_woe      0.133      0.0481    2.77    5.58e-  3
# 9 age_missing_woe                 -0.0793     0.0529    -1.50   1.34e-  1
# 10 Credit_History_Total_Months_woe  0.0441     0.0147   3.00    2.69e-  3
# 11 Card_per_Bank_woe                0.133      0.0232   5.74    9.74e-  9
# 12 Investiment_to_Income_woe        0.468      0.0640   7.33    2.39e- 13

# Como interpretar magnitude
# Lembrete:
# Coeficiente está em log-odds.
# Para interpretar em odds ratio:
# OR=e^β (e = numero de euler ≈ 2,718)
# 
# Exemplo:
#   
# Outstanding_Debt_woe:e^0.807 ≈ 2.24
# 
# Isso significa:
# 👉 A cada aumento unitário no WOE da dívida,
# 👉 As odds de default aumentam 2.24 vezes.

#Gerando probabilidades no treino
pred_treino <- predict(modelo_logit, base_treino, type = "prob") |>
  bind_cols(base_treino |> select(Default))


#Calcular AUC
roc_auc(pred_treino, truth = Default, .pred_1, event_level = "second")
#👉 Em 80% das vezes, o modelo consegue rankear corretamente um cliente mau acima de um cliente bom.
#Classificação prática de AUC:
# | AUC         | Interpretação |
# | ----------- | ------------- |
# | 0.50        | Aleatório     |
# | 0.60 – 0.70 | Fraco         |
# | 0.70 – 0.75 | Aceitável     |
# | 0.75 – 0.80 | Bom           |
# | 0.80 – 0.85 | Muito bom     |
# | > 0.85      | Excelente     |

# A partir do AUC ja podemos estimar o Gini manualmente
# Gini = 2 x AUC - 1: 2 x 0.801 - 1 = 0.602 ≈ 60%

#Cálculo do KS (Kolmogorov-Smirnov)
#O KS mede: A maior distância entre a distribuição acumulada de bons e maus.
#Em credit scoring:
# | KS     | Interpretação |
# | ------ | ------------- |
# | < 20%  | Fraco         |
# | 20–30% | Razoável      |
# | 30–40% | Bom           |
# | 40–60% | Muito bom     |
# | > 60%  | Excelente     |

ks_table <- pred_treino |> 
  arrange(desc(.pred_1)) |> 
  mutate(
    good = ifelse(Default == "0", 1, 0),
    bad = ifelse(Default == "1", 1, 0),
    cum_good = cumsum(good)/sum(good),
    cum_bad = cumsum(bad)/sum(bad),
    ks = abs(cum_bad - cum_good)
  )
ks_table
ks_value <- max(ks_table$ks)
ks_value # 👉 ks: 0.552 ≈ 55,2% Muito Bom
#👉 No ponto ótimo de corte, a diferença entre a proporção acumulada de maus e bons é de 55%

#Gerar Curva ROC
#Aqui vamos usar o yardstick, não é preciso carrega-lo pois ele ja esta dentro do tidymodels
#Tambem vamos usar ggplot2 que esta dentro do tidyverse

roc_obj <- roc_curve(
                     data = pred_treino,#obrigatório passar um dataframe
                     truth = Default,#A variável verdadeira (real) é a coluna Default
                    .pred_1,#Usa a probabilidade da classe "1", classe do default
                     event_level = "second")
#Para cada possível cutoff, calcule a taxa de verdadeiros positivos e falsos positivos 
#considerando que Default = 1 é o evento.
roc_obj
#ATENÇÃO!!!
#Por padrão, o yardstick considera:
# O PRIMEIRO nível como evento positivo
# Mas eu quero que:
# "1" seja o evento (default)
# Como "1" é o segundo nível especificamos sempre "event_level = "second"", se não o AUC fica invertido
autoplot(roc_obj)

#Matriz de Confusão
#Ponto ótimo a partir de onde o KS é maximo
ks_table |>
  slice_max(ks, n = 1) # a partir do KS o cutoff ideal seria 0.269, muito conservador
# um cliente com apenas 27% de chance de dar default ja seria reprovado
#Porém a analise gráfica indica cutoff de 75%, o que seria muito permissivo, então vamos
#rodar 3 valores de cutoff e comparar as métricas, especialmente a sensibilidade que captura os falsos negativos
pred_class_0.26 <- pred_treino |> 
  mutate(
    pred_class = ifelse(.pred_1 >= 0.26, "1", "0") |> 
      factor(levels = c("0", "1"))
  )

pred_class_0.5 <- pred_treino |> 
  mutate(
    pred_class = ifelse(.pred_1 >= 0.5, "1", "0") |> 
      factor(levels = c("0", "1"))
  )

pred_class_0.75 <- pred_treino |> 
  mutate(
    pred_class = ifelse(.pred_1 >= 0.75, "1", "0") |> 
      factor(levels = c("0", "1"))
  )

conf_mat(pred_class_0.26, truth = Default, estimate = pred_class)
conf_mat(pred_class_0.5, truth = Default, estimate = pred_class)
conf_mat(pred_class_0.75, truth = Default, estimate = pred_class)

#Comparação entre os pontos de cutoff
# | Cutoff | Recall | Aprovação | Default na Carteira |
# | ------ | ------ | --------- | ------------------- |
# | 0.26   | 75%    | 63%       | 11%                 |
# | 0.50   | 58%    | 73%       | 17%                 |
# | 0.75   | 9%     | 96%       | 27%                 |


#Métricas adicionais
accuracy(pred_class_0.26, truth = Default, estimate = pred_class)
#78,3% das classificações estão corretas.

precision(pred_class_0.26, truth = Default, estimate = pred_class, event_level = "second")
#Entre todos que o modelo classificou como maus:
# 👉 59.9% realmente são maus.
# Interpretação prática
# Se o banco rejeitar todos os classificados como risco alto:
#   
# 60% das rejeições são corretas
# 
# 40% seriam clientes bons rejeitados
# Isso mede qualidade das rejeições.

recall(pred_class_0.26, truth = Default, estimate = pred_class, event_level = "second")
# O modelo captura 75% dos inadimplentes reais.
# 👉 Apenas 25% dos maus escapam.
# Essa é uma métrica crítica em crédito.
# Porque:
#   
# FN = inadimplente aprovado
# 
# Isso gera prejuízo direto
# 
# Esse valor é muito bom.

specificity(pred_class_0.26, truth = Default, estimate = pred_class, event_level = "second")
# O modelo aprova corretamente 79% dos bons clientes.
# 👉 Apenas 21% dos bons são rejeitados.
# Essa métrica mede eficiência comercial.

# | Métrica     | Valor | Interpretação                          |
# | ----------- | ----- | -------------------------------------- |
# | Accuracy    | 78%   | Boa performance geral                  |
# | Recall      | 75%   | Forte captura de maus                  |
# | Specificity | 79%   | Boa aprovação de bons                  |
# | Precision   | 60%   | 40% das rejeições são falsos positivos |

#### 🔎 Validação Out-of-Sample (Base Teste) ####

#Gerar probabilidades na base teste
pred_teste <- predict(modelo_logit, base_teste, type = "prob")

#Comparando treino e teste
summary(pred_treino$.pred_1)
summary(pred_teste$.pred_1)

#Comparando as médias
mean(pred_treino$.pred_1)
mean(pred_teste$.pred_1)

#Visualizando distribuição
ggplot()+
  geom_density(aes(x = pred_treino$.pred_1), color = "blue")+
  geom_density(aes(x = pred_teste$.pred_1), color = "purple")+
  labs(title = "Distribuição de Probabilidade - Treino vs Teste")

#A distribuição de probabilidade de inadimplência da base teste é estatisticamente 
#equivalente à base treino, indicando estabilidade populacional e ausência de drift

#Calculo do PSI

calc_psi <- function(base_treino, base_teste, bins = 10) {
  breaks <- quantile(base_treino, probs = seq(0, 1, length.out = bins + 1))
  
  train_bins <- cut(base_treino, breaks = breaks, include.lowest = TRUE)
  test_bins <- cut(base_teste, breaks = breaks, include.lowest = TRUE)
  
  train_dist <- prop.table(table(train_bins))
  test_dist <- prop.table(table(test_bins))
  
  psi <- sum((test_dist - train_dist) * log(test_dist / train_dist))  
  
  return(psi)
}

psi_score <- calc_psi(pred_treino$.pred_1, pred_teste$.pred_1)
psi_score
# PSI ≈ 0.00031
#O Population Stability Index (PSI) entre a base de desenvolvimento e a base de validação foi de 0.0003, 
#indicando alta estabilidade populacional e ausência de drift significativo.

#Criando Scorecard

#Parâmetros do scorecard (numeros escolhidos de forma aleatória)
pdo <- 20
base_score <- 600
base_odds <- 50

#a função scorecard espera receber um modelo do tipo glm padrão para funcionar
#porém o nosso modelo está encapsulado pelo tidymodels, precisamos extrair
modelo_glm <- modelo_logit |> 
  extract_fit_parsnip() |> 
  pluck("fit")
class(modelo_glm) #extraimos

#Criar scorecard
card <- scorecard(
  bins_selected,
  modelo_glm,
  points0 = base_score,
  odds0 = base_odds,
  pdo = pdo
)
# Agora cada faixa de variavel irá receber pontos
#Exemplo:
# Age = 28 → +12
# Debt = baixo → +40
# Loans = poucos → +20
#ScoreFinal = 600 + 12 + 40 + 20 = 672

#Calcular score dos clientes
score_clientes <- scorecard_ply(
  train_selected,
  card
)

summary(score_clientes)
hist(score_clientes$score)

ggplot(score_clientes, aes(score)) +
  geom_histogram(bins = 30, fill = "steelblue")

#Score esta variando entre 676 até 796, com maior faixa na casa dos 700

##### Finalizando #####

#1) Gerar a tabela completa do Scorecard (pontos por variável)

scorecard_table <- rbindlist(card, fill = TRUE) |>
  dplyr::select(variable, bin, woe, points)

scorecard_table

#Estrutura da Tabela do Scorecard
# | coluna       | significado                  |
# | ------------ | ---------------------------- |
# | **variable** | variável do modelo           |
# | **bin**      | faixa de valores da variável |
# | **woe**      | Weight of Evidence           |
# | **points**   | pontos atribuídos ao cliente |

#Scorebase do modelo Score = 739. Isso significa que todo cliente começa com essa pontuação
# e depois os pontos das variáveis são somados ou subtraídos.

#Outstanding_Debt -> variavel mais importante
# [-Inf,1200) → +29
# [1500,2700) → -41
#70 pontos de diferença, Quanto maior a dívida → maior risco de default.

#2) Taxa de Default por Score
#2.1) Criar faixas de score
score_clientes <- score_clientes |> 
  mutate(score_bin = ntile(-score, 10)) #fazemos um agrupamento de 10 faixas do melhor escolhe para o pior

#Calcular default rate
default_rate <- score_clientes |>
  mutate(Default = base_treino$Default) |>
  group_by(score_bin) |>
  summarise(
    clientes = n(),
    defaults = sum(Default == 1),
    default_rate = mean(Default == 1),
    score_medio = mean(score)
  )

default_rate

#Visualizar relação Score × Default
ggplot(default_rate, aes(x = score_medio, y = default_rate))+
  geom_line(color = "steelblue")+
  geom_point(size = 3)+
  theme_classic()+
  labs(
    title = "Taxa de Default por Faixa de Score",
    x = "Score Médio",
    y = "Default Rate"
  )















