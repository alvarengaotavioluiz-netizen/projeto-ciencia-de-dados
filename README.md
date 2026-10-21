# 📊 Sales Intelligence — Análise e Previsão de Vendas

Projeto acadêmico de análise exploratória de vendas e avaliação de um modelo de machine learning, com dashboard interativo em Streamlit.

## Objetivo

Investigar a evolução histórica das vendas, identificar padrões de sazonalidade e avaliar a capacidade de um modelo de regressão de estimar vendas diárias, comparando-o a referências simples (*baselines*).

## Dados

O projeto utiliza o dataset público **Store Sales Forecasting**, com série exibida no dashboard entre **01/01/2013 e 15/08/2017**. Os valores apresentados correspondem às unidades/valores de vendas registrados no dataset e **não devem ser automaticamente interpretados como faturamento em reais**. A análise é feita sobre vendas diárias agregadas.

## Tecnologias

- **Python** — processamento e aplicação.
- **Pandas** — leitura e preparação dos dados tabulares.
- **Scikit-learn** — modelagem e avaliação (modelo identificado no projeto: `HistGradientBoostingRegressor`).
- **Streamlit** — interface interativa.
- **Plotly** — visualizações.

## Organização do dashboard

### 1. Visão Executiva

Filtro de período, vendas totais, média diária, maior venda diária, dias analisados, série histórica e comparações por ano e mês. O período completo exibido contém **1.684 dias**, com vendas totais de aproximadamente **1,074 bilhão** de unidades do dataset.

### 2. Inteligência de Vendas

Filtros por ano, sazonalidade mensal e semanal e tendência das vendas. A média móvel de 30 registros ajuda a visualizar a tendência histórica, **não sendo uma previsão**.

### 3. Machine Learning

Indicadores MAE, RMSE e R² no teste, comparação com baselines, gráfico de valores reais versus previstos e importância das variáveis por permutação.

## Abordagem de modelagem

O modelo utilizado é o **HistGradientBoostingRegressor**, da família de métodos de *gradient boosting*, que combina sucessivas árvores de decisão para aprender padrões não lineares. Entre as variáveis apresentadas no dashboard estão dia da semana, dia do mês, mês, trimestre, vendas anteriores e estatísticas de janelas históricas.

A seleção foi realizada por **validação temporal**; o conjunto de teste corresponde a um período posterior aos períodos de treino e validação. A visualização mostra uma **previsão diária com histórico real atualizado**, não uma previsão recursiva de longo prazo.

> **Nota de transparência:** como os resultados do teste foram consultados durante refinamentos, os números atuais devem ser apresentados como **reavaliação exploratória**, e não como uma nova avaliação totalmente independente após esses refinamentos.

## Resultados

| Abordagem | MAE | RMSE | R² |
|---|---:|---:|---:|
| HistGradientBoostingRegressor | 68.055,42 | 94.921,82 | 0,6733 |
| Baseline: vendas de ontem | 150.483,17 | 186.805,69 | -0,2652 |
| Baseline: 7 dias | 96.732,05 | 134.722,97 | 0,3420 |

Valores arredondados conforme o CSV exibido no dashboard. O nome `Gradient_Boosting_Final` pode aparecer no CSV como identificação legada; o algoritmo indicado na aplicação é `HistGradientBoostingRegressor`.

- **MAE:** erro absoluto médio. Em média, as previsões diferiram das vendas reais em cerca de 68 mil unidades do dataset.
- **RMSE:** raiz do erro quadrático médio. Penaliza mais os erros grandes; menor é melhor.
- **R²:** coeficiente de determinação. O valor 0,6733 indica que o modelo explica aproximadamente 67,33% da variabilidade do alvo no conjunto avaliado, relativamente à referência de previsão pela média. **Não é uma taxa de acerto.**

O modelo apresentou RMSE inferior aos dois baselines, sugerindo melhor desempenho nesse conjunto de avaliação.

## Interpretação das variáveis

A importância por permutação mede o quanto o RMSE se altera quando os valores de uma variável são embaralhados. As maiores contribuições exibidas são `dia_semana`, `vendas_ontem` e `dia_mes`. Os valores **não são percentuais**, podem ser negativos e não representam relações de causa e efeito. A interpretação é exploratória, especialmente por ter sido calculada no teste.

## Como executar o dashboard

Com Python instalado, na pasta do projeto:

```bash
pip install pandas streamlit plotly scikit-learn
streamlit run dashboard.py
```

Se o arquivo utilizado for `dashboard_revisado.py`, substitua o nome no comando. A aplicação lê arquivos CSV dentro de `resultados/`, incluindo:

```text
resultados/
├── vendas_diarias.csv
├── vendas_mensais.csv
├── sazonalidade_mensal.csv
├── sazonalidade_semanal.csv
├── previsoes_teste_final.csv
├── metricas_teste_final.csv
└── importancia_variaveis.csv
```

Esses CSVs precisam ter sido gerados previamente pelo fluxo de preparação e avaliação do projeto. **Executar o dashboard não treina novamente o modelo.** O código de treinamento não foi incluído neste README porque não estava disponível para conferência.

## Limitações e próximos passos

- Métricas são específicas do período de teste e não garantem desempenho futuro.
- Erros maiores podem ocorrer em picos e mudanças inesperadas.
- O dashboard não constitui um sistema de previsão autônoma de longo prazo.
- Uma etapa futura seria validar o fluxo completo com novos períodos não utilizados em refinamentos e automatizar a atualização dos dados.

## Contexto

Projeto acadêmico de Machine Learning e análise operacional de vendas. As visualizações e resultados são apresentados para fins educacionais e de interpretação dos dados.
