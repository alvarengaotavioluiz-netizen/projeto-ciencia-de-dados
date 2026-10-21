from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================
# CONFIGURAÇÃO
# ============================================

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

st.set_page_config(
    page_title="Inteligência de Vendas | ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILO VISUAL
# ============================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetric"] {
        background: #14243A;
        border: 1px solid #29415E;
        border-radius: 12px;
        padding: 18px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARREGAMENTO DOS DADOS
# ============================================

@st.cache_data
def carregar_csv(nome):
    caminho = RESULTADOS / nome
    if not caminho.exists():
        return None
    return pd.read_csv(caminho)


def localizar_coluna(df, opcoes):
    if df is None:
        return None
    mapa = {str(c).lower(): c for c in df.columns}
    for nome in opcoes:
        if nome.lower() in mapa:
            return mapa[nome.lower()]
    return None


def preparar_serie(df):
    if df is None:
        return None

    df = df.copy()
    data = localizar_coluna(df, ["date", "data"])
    vendas = localizar_coluna(
        df, ["sales", "vendas", "total_vendas"]
    )

    if data is None or vendas is None:
        return None

    df[data] = pd.to_datetime(df[data], errors="coerce")
    df[vendas] = pd.to_numeric(df[vendas], errors="coerce")
    df = df.dropna(subset=[data, vendas])
    df = df.rename(columns={data: "date", vendas: "sales"})
    return df.sort_values("date")


def numero_br(valor, casas=0):
    """Formata números no padrão brasileiro, sem assumir moeda."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def preparar_metricas(df):
    if df is None or df.empty:
        return None
    obrigatorias = {"modelo", "MAE", "RMSE", "R2"}
    if not obrigatorias.issubset(df.columns):
        return None
    df = df.copy()
    for coluna in ("MAE", "RMSE", "R2"):
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    # Mantém compatibilidade com o nome legado exportado pelo script.
    nomes_baseline = {"Baseline_Teste", "Baseline_7_Dias"}
    candidatos = df[~df["modelo"].isin(nomes_baseline)].dropna(subset=["MAE", "RMSE", "R2"])
    if candidatos.empty:
        return None
    return df, candidatos.iloc[0]


def grafico_linha(df, x, y, titulo, cor="#38BDF8"):
    fig = px.line(df, x=x, y=y, title=titulo)
    fig.update_traces(line_color=cor, line_width=2)
    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(l=15, r=15, t=55, b=15),
        hovermode="x unified"
    )
    return fig


vendas = preparar_serie(carregar_csv("vendas_diarias.csv"))
mensais = preparar_serie(carregar_csv("vendas_mensais.csv"))
sazonal_mensal = carregar_csv("sazonalidade_mensal.csv")
sazonal_semanal = carregar_csv("sazonalidade_semanal.csv")
previsoes = carregar_csv("previsoes_teste_final.csv")
metricas = carregar_csv("metricas_teste_final.csv")
importancias = carregar_csv("importancia_variaveis.csv")

# ============================================
# MENU LATERAL
# ============================================

with st.sidebar:
    st.title("📊 Sales Intelligence")
    st.caption("Machine Learning • Operações")
    st.divider()

    pagina = st.radio(
        "Navegação",
        [
            "Visão Executiva",
            "Inteligência de Vendas",
            "Machine Learning"
        ]
    )

    st.divider()
    st.caption("Projeto acadêmico")
    st.caption("Dataset público: Store Sales Forecasting")
    st.caption("Modelo: HistGradientBoostingRegressor")

# ============================================
# PÁGINA 1 — VISÃO EXECUTIVA
# ============================================

if pagina == "Visão Executiva":

    st.title("Visão Executiva de Vendas")
    st.caption("Análise histórica e indicadores operacionais")

    if vendas is None or vendas.empty:
        st.error(
            "Não foi possível carregar resultados/vendas_diarias.csv "
            "com as colunas date e sales."
        )
        st.stop()

    data_min = vendas["date"].min().date()
    data_max = vendas["date"].max().date()

    periodo = st.date_input(
        "Selecione o período",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max
    )

    filtrado = vendas.copy()

    if isinstance(periodo, (tuple, list)) and len(periodo) == 2:
        inicio, fim = periodo
        filtrado = filtrado[
            (filtrado["date"].dt.date >= inicio) &
            (filtrado["date"].dt.date <= fim)
        ]

    if filtrado.empty:
        st.warning("Nenhum registro no período selecionado.")
        st.stop()

    total = filtrado["sales"].sum()
    media = filtrado["sales"].mean()
    maior = filtrado["sales"].max()
    dias = len(filtrado)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Vendas totais", numero_br(total))
    c2.metric("Média diária", numero_br(media))
    c3.metric("Maior venda diária", numero_br(maior))
    c4.metric("Dias analisados", numero_br(dias))

    st.divider()

    st.plotly_chart(
        grafico_linha(
            filtrado, "date", "sales",
            "Evolução das Vendas Diárias"
        ),
        use_container_width=True
    )

    esquerda, direita = st.columns(2)

    with esquerda:
        anual = (
            filtrado.assign(ano=filtrado["date"].dt.year)
            .groupby("ano", as_index=False)["sales"]
            .sum()
        )
        anual["ano"] = anual["ano"].astype(str)

        fig = px.bar(
            anual, x="ano", y="sales",
            title="Vendas por Ano",
            color_discrete_sequence=["#22C55E"]
        )
        fig.update_layout(template="plotly_dark", height=370)
        st.plotly_chart(fig, use_container_width=True)

    with direita:
        por_mes = (
            filtrado.assign(mes=filtrado["date"].dt.month)
            .groupby("mes", as_index=False)["sales"]
            .mean()
        )
        fig = px.bar(
            por_mes, x="mes", y="sales",
            title="Média Diária de Vendas por Mês",
            color_discrete_sequence=["#F59E0B"]
        )
        fig.update_layout(template="plotly_dark", height=370)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Os valores representam vendas registradas no dataset. "
        "Não são necessariamente faturamento em reais."
    )

# ============================================
# PÁGINA 2 — INTELIGÊNCIA DE VENDAS
# ============================================

elif pagina == "Inteligência de Vendas":

    st.title("Inteligência de Vendas")
    st.caption("Sazonalidade, tendências e padrões operacionais")

    if vendas is None or vendas.empty:
        st.error("Dados de vendas diárias não encontrados.")
        st.stop()

    df = vendas.copy()
    df["mes"] = df["date"].dt.month
    df["dia_semana"] = df["date"].dt.dayofweek
    df["ano"] = df["date"].dt.year

    anos = sorted(df["ano"].unique().tolist())

    selecionados = st.multiselect(
        "Filtrar anos",
        anos,
        default=anos
    )

    df = df[df["ano"].isin(selecionados)]

    if df.empty:
        st.warning("Selecione pelo menos um ano.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        meses = (
            df.groupby("mes", as_index=False)["sales"]
            .mean()
        )
        fig = px.bar(
            meses, x="mes", y="sales",
            title="Sazonalidade Mensal — Média Diária",
            color_discrete_sequence=["#38BDF8"]
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        semana = (
            df.groupby("dia_semana", as_index=False)["sales"]
            .mean()
        )
        nomes = {
            0: "Seg", 1: "Ter", 2: "Qua",
            3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"
        }
        semana["dia"] = semana["dia_semana"].map(nomes)

        fig = px.bar(
            semana, x="dia", y="sales",
            title="Sazonalidade Semanal",
            color_discrete_sequence=["#A78BFA"]
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tendência das Vendas")

    df = df.sort_values("date").copy()
    df["media_movel_30"] = (
        df["sales"].rolling(30, min_periods=7).mean()
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sales"],
        name="Vendas diárias",
        line=dict(color="#64748B", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["media_movel_30"],
        name="Média móvel de 30 registros",
        line=dict(color="#38BDF8", width=3)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=430,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "A média móvel acima é descritiva, não uma previsão. "
        "Os cálculos consideram os registros do período filtrado."
    )

# ============================================
# PÁGINA 3 — MACHINE LEARNING
# ============================================

else:
    st.title("Machine Learning — Previsão de Vendas")
    st.caption("HistGradientBoostingRegressor | avaliação temporal no conjunto de teste")

    st.subheader("Desempenho do Modelo")
    avaliacao = preparar_metricas(metricas)
    if avaliacao is None:
        st.error("Métricas ausentes ou inválidas em resultados/metricas_teste_final.csv.")
    else:
        tabela_metricas, linha_modelo = avaliacao
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE no teste", numero_br(linha_modelo["MAE"], 2))
        c2.metric("RMSE no teste", numero_br(linha_modelo["RMSE"], 2))
        c3.metric("R² no teste", numero_br(linha_modelo["R2"], 4))
        st.caption("Indicadores carregados automaticamente do CSV de avaliação final.")

        comparacao = tabela_metricas.dropna(subset=["RMSE"]).copy()
        comparacao["modelo_exibicao"] = comparacao["modelo"].replace({
            "Gradient_Boosting_Final": "HistGradientBoosting (final)",
            "HistGradientBoosting_Final": "HistGradientBoosting (final)",
            "Baseline_Teste": "Baseline: ontem",
            "Baseline_7_Dias": "Baseline: 7 dias",
        })
        st.subheader("Comparação com os baselines")
        fig_comparacao = px.bar(
            comparacao, x="modelo_exibicao", y="RMSE",
            color="modelo_exibicao", title="RMSE por modelo — menor é melhor",
            text="RMSE",
        )
        fig_comparacao.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_comparacao.update_layout(
            template="plotly_white", height=380, showlegend=False,
            xaxis_title="", yaxis_title="RMSE (unidades de vendas)",
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig_comparacao, use_container_width=True)

    st.divider()
    st.subheader("Vendas reais versus previstas")
    colunas_prev = {"date", "vendas_reais", "vendas_previstas"}
    if previsoes is None or not colunas_prev.issubset(previsoes.columns):
        st.warning("Arquivo de previsões ausente ou sem as colunas esperadas.")
    else:
        serie = previsoes.copy()
        serie["date"] = pd.to_datetime(serie["date"], errors="coerce")
        for col in ("vendas_reais", "vendas_previstas", "baseline", "baseline_7_dias"):
            if col in serie:
                serie[col] = pd.to_numeric(serie[col], errors="coerce")
        serie = serie.dropna(subset=["date", "vendas_reais", "vendas_previstas"]).sort_values("date")
        if serie.empty:
            st.warning("Não há previsões válidas para exibir.")
        else:
            incluir_baselines = st.checkbox("Exibir baselines no gráfico", value=False)
            fig = go.Figure()
            for coluna, nome, cor in (
                ("vendas_reais", "Vendas reais", "#38BDF8"),
                ("vendas_previstas", "Previsão do modelo", "#F59E0B"),
            ):
                fig.add_trace(go.Scatter(
                    x=serie["date"], y=serie[coluna], mode="lines",
                    name=nome, line=dict(color=cor, width=2),
                ))
            if incluir_baselines:
                for coluna, nome, cor in (
                    ("baseline", "Baseline: ontem", "#94A3B8"),
                    ("baseline_7_dias", "Baseline: 7 dias", "#A78BFA"),
                ):
                    if coluna in serie:
                        fig.add_trace(go.Scatter(
                            x=serie["date"], y=serie[coluna], mode="lines",
                            name=nome, line=dict(color=cor, width=1, dash="dot"),
                        ))
            fig.update_layout(
                template="plotly_white", height=440, hovermode="x unified",
                xaxis_title="Data", yaxis_title="Vendas (unidades do dataset)",
                margin=dict(l=10, r=10, t=25, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Previsão diária com histórico de vendas reais atualizado; não é uma previsão recursiva de longo prazo.")

    st.divider()
    st.subheader("Importância das variáveis")
    if importancias is None or not {"variavel", "importancia"}.issubset(importancias.columns):
        st.warning("Importância das variáveis indisponível.")
    else:
        imp = importancias.copy()
        imp["importancia"] = pd.to_numeric(imp["importancia"], errors="coerce")
        imp = imp.dropna(subset=["importancia"]).sort_values("importancia")
        if not imp.empty:
            fig_imp = px.bar(
                imp, x="importancia", y="variavel", orientation="h",
                title="Impacto da permutação de cada variável no RMSE",
                color_discrete_sequence=["#22C55E"],
            )
            fig_imp.update_layout(
                template="plotly_white", height=460,
                xaxis_title="Variação do RMSE", yaxis_title="",
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Importância por permutação calculada no teste, apenas para interpretação exploratória. "
                "Valores não são percentuais e podem ser negativos; variáveis correlacionadas "
                "podem dividir a importância entre si."
            )
            with st.expander("Ver tabela de importância"):
                st.dataframe(imp.sort_values("importancia", ascending=False),
                             use_container_width=True, hide_index=True)

    with st.expander("Métricas exportadas — dados originais"):
        if metricas is not None:
            st.dataframe(metricas, use_container_width=True, hide_index=True)
            st.caption("O CSV pode manter o nome legado Gradient_Boosting_Final; o algoritmo treinado foi HistGradientBoostingRegressor.")
        else:
            st.warning("Arquivo de métricas não encontrado.")

    st.info(
        "A seleção do modelo foi feita na validação temporal. O teste é posterior "
        "aos períodos de treino e validação. Como os resultados do teste já foram "
        "consultados durante refinamentos, esta versão deve ser descrita como "
        "reavaliação exploratória, não como nova seleção independente de modelo."
    )

# ============================================
# RODAPÉ
# ============================================

st.divider()
st.caption(
    "Sales Intelligence | Projeto acadêmico de Machine Learning "
    "| Python • Scikit-learn • Streamlit • Plotly"
)
