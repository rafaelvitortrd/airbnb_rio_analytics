# Aplicação Web para Estatística Inferencial e Probabilidade - Parte 3
# Aluno: Rafael Vitor T R Dutra

import logging
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats  
import os

st.set_page_config(page_title="Airbnb Rio - Estatística Inferencial", layout="wide")

traducoes = {
    'price': 'Preço',
    'review_scores_rating': 'Avaliação Geral',
    'review_scores_value': 'Custo-Benefício',
    'review_scores_location': 'Localização',
    'accommodates': 'Acomodações',
    'neighbourhood_cleansed': 'Bairro',
    'room_type': 'Tipo de Quarto'
}

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'listings.csv')
    
    if not os.path.exists(file_path):
        file_path = 'listings.csv'
        
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df = df[df['price'] <= 10000] 
    
    for col in ['review_scores_rating', 'review_scores_value', 'review_scores_location']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    return df

df = load_data()

if df is None:
    st.error("❌ O arquivo **listings.csv** não foi encontrado!")
    st.stop()

st.sidebar.header("⚙️ Configurações e Filtros")

lista_bairros = sorted(df['neighbourhood_cleansed'].unique())
selecao_bairros = st.sidebar.multiselect("Bairros:", options=lista_bairros, default=['Copacabana', 'Ipanema', 'Leblon'])

limite_razoavel = float(df['price'].quantile(0.95))
preco_range = st.sidebar.slider("Faixa de Preço Global (R$):", 0.0, 10000.0, (0.0, limite_razoavel))

tipo_quarto = st.sidebar.multiselect("Tipo de Quarto:", options=df['room_type'].unique(), default=df['room_type'].unique())

df_filt = df[
    (df['neighbourhood_cleansed'].isin(selecao_bairros)) &
    (df['price'] >= preco_range[0]) & (df['price'] <= preco_range[1]) &
    (df['room_type'].isin(tipo_quarto))
]

st.title("📊 Airbnb Rio de Janeiro: Análise Estatística Inferencial")

with st.expander("📖 Sobre este Projeto: Contextualização e Pergunta de Pesquisa"):
    st.markdown("""
    ### 🏨 Contextualização
    Este projeto utiliza dados do **Inside Airbnb** para modelar o comportamento de aluguel de curto prazo no Rio de Janeiro sob a ótica da **Estatística Inferencial e Probabilidade**.
    
    ### ❓ Pergunta de Pesquisa
    **Qual é a relação de dependência entre o preço, a localização e a percepção de custo-benefício dos hóspedes no mercado do Airbnb no Rio de Janeiro?**
    """)

st.markdown(f"Amostra atual filtrada para análise: **{len(df_filt)}** observações.")

if len(df_filt) > 5:
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Preço Médio", f"R$ {df_filt['price'].mean():.2f}")
    with m2: st.metric("Custo-Benefício Médio", f"{df_filt['review_scores_value'].mean():.2f} / 5.0")
    with m3: st.metric("Nota Média de Localização", f"{df_filt['review_scores_location'].mean():.2f} / 5.0")

    st.divider()

    st.write("### 1. Distribuição Geográfica de Preços")
    fig1 = px.scatter_mapbox(df_filt, lat="latitude", lon="longitude", color="price", 
                            size="accommodates", color_continuous_scale="Viridis",
                            mapbox_style="carto-positron", zoom=10, height=450,
                            labels={'price': 'Preço', 'accommodates': 'Acomodações'})
    fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 2. Modelagem Probabilística: FDP das Notas de Localização")
        df_loc_clean = df_filt.dropna(subset=['review_scores_location'])
        
        mu_loc, sigma_loc = stats.norm.fit(df_loc_clean['review_scores_location'])
        
        fig_hist = px.histogram(df_loc_clean, x='review_scores_location', nbins=20, histnorm='probability density',
                                template="plotly_white", color_discrete_sequence=['#cbd5e1'], 
                                labels={'review_scores_location': 'Nota de Localização'}, range_x=[1, 5])
        
        x_curva = np.linspace(1, 5, 200)
        y_curva = stats.norm.pdf(x_curva, mu_loc, sigma_loc)
        
        fig_hist.add_trace(go.Scatter(x=x_curva, y=y_curva, mode='lines', line=dict(color='#3b82f6', width=3), name='FDP Teórica (EMV)'))
        fig_hist.update_layout(yaxis_title="Densidade de Probabilidade", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.caption(f"""**Interpretação da FDP:** A curva azul representa a Função Densidade de Probabilidade teórica ajustada por Máxima Verossimilhança. 
        A forte concentração à direita indica que, probabilisticamente, a chance de um imóvel receber uma nota de localização excelente é esmagadora neste mercado.""")

    with col2:
        st.write("### 3. Boxplot: Variabilidade do Custo-Benefício por Bairro")
        fig2 = px.box(df_filt, x="neighbourhood_cleansed", y="review_scores_value", color="room_type",
                      template="plotly_white", height=450, labels=traducoes, range_y=[1, 5])
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Ajuste por Filtro: Mede a dispersão, mediana e a presença of outliers (pontos fora da curva) na percepção de Custo-Benefício comparando as regiões escolhidas.")

    st.divider()

    st.write("### 4. Função de Distribuição Acumulada Empírica (FDA) do Custo-Benefício")
    df_ecdf_clean = df_filt.dropna(subset=['review_scores_value'])
    fig_ecdf = px.ecdf(df_ecdf_clean, x="review_scores_value", color="room_type", 
                       template="plotly_white", height=450, labels=traducoes, range_x=[1, 5])
    fig_ecdf.update_layout(yaxis_title="Probabilidade Acumulada (F(x))", xaxis_title="Nota de Custo-Benefício")
    st.plotly_chart(fig_ecdf, use_container_width=True)
    
    st.markdown("<p style='text-align: center; font-weight: bold;'>Modelo Probabilístico Acumulado:</p>", unsafe_allow_html=True)
    st.latex(r"F(x) = P(X \le x)")

    st.divider()

    st.write("### 5. Teste de Hipótese Inferencial: Preço vs. Percepção de Custo-Benefício")
    df_reg = df_filt.dropna(subset=['price', 'review_scores_value'])
    
    if len(df_reg) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_reg['price'], df_reg['review_scores_value'])
        r_squared = r_value ** 2
        
        df_reg['predicted'] = intercept + slope * df_reg['price']
        df_reg['residuals'] = df_reg['review_scores_value'] - df_reg['predicted']
        
        c_graf5, c_txt5 = st.columns([2, 1])
        
        with c_graf5:
            # Gerar o gráfico base com os pontos dispersos
            fig_reg = px.scatter(df_reg, x="price", y="review_scores_value", opacity=0.4, 
                                 template="plotly_white", height=450, color_discrete_sequence=['#10b981'], labels=traducoes)
            
            # Ajustar os limites do eixo X dinamicamente para ancorar a reta nas extremidades exatas
            x_min, x_max = float(df_reg['price'].min()), float(df_reg['price'].max())
            x_reg = np.linspace(x_min, x_max, 100)
            y_reg = slope * x_reg + intercept
            
            # Adicionar a reta com garantia de visibilidade e largura reforçada
            fig_reg.add_trace(go.Scatter(x=x_reg, y=y_reg, mode='lines', 
                                         line=dict(color='#ef4444', width=4), 
                                         name='Reta de Regressão'))
            
            # Configuração final para garantir que o Plotly não oculte a trace
            fig_reg.update_layout(showlegend=True, yaxis=dict(autorange=True))
            st.plotly_chart(fig_reg, use_container_width=True)
            
        with c_txt5:
            st.markdown("**Modelo de Regressão Linear Populacional:**")
            st.latex(r"Y = \beta_0 + \beta_1 X + \epsilon")
            st.markdown(f"**Equação Ajustada:**")
            st.code(f"Custo-Benefício = {intercept:.4f} + ({slope:.4f}) * Preço")
            
            st.markdown("---")
            st.markdown("**Métricas de Validação Inferencial:**")
            st.metric("Variância Explicada ($R^2$)", f"{r_squared:.4f}")
            st.metric("p-valor Calculado", f"{p_value:.4e}")
            
            if p_value < 0.05:
                st.success("✔️ Rejeita-se H0: O impacto do preço na nota é estatisticamente significante (Confiança >= 95%).")
            else:
                st.warning("⚠️ Não se rejeita H0: Não há evidência estatística de dependência linear na amostra atual.")
                
            st.caption("Nota de leitura: A inclinação da reta vermelha determina se cobrar mais caro diminui (reta descendente) ou aumenta a nota dada pelo hóspede.")

        st.write("#### Diagnóstico do Modelo: Distribution dos Resíduos (Erros)")
        fig_res = px.scatter(df_reg, x="price", y="residuals", opacity=0.5, template="plotly_white", height=300,
                             color_discrete_sequence=['#64748b'], labels={'price': 'Preço', 'residuals': 'Resíduo (Erro)'})
        fig_res.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption("Validação Teórica: Para a regressão linear ser válida inferencialmente, os erros devem estar espalhados aleatoriamente ao redor da linha vermelha zero (Homocedasticidade), sem formar desenhos geométricos ou funis.")

    else:
        st.warning("Dados insuficientes para calcular a regressão linear.")

    st.divider()

    st.write("### 6. Análise de Oportunidades: Custo-Benefício vs. Localização por Bairro Selecionado")
    st.markdown("**Aviso de Filtro:** Este gráfico abaixo ignora o menu lateral esquerdo e foca exclusivamente no bairro que escolher na caixa de seleção a seguir:")
    
    bairro_exclusivo = st.selectbox("Escolha o Bairro para Mapeamento Unitário:", options=lista_bairros, index=lista_bairros.index('Copacabana') if 'Copacabana' in lista_bairros else 0)
    
    df_bairro_unico = df[
        (df['neighbourhood_cleansed'] == bairro_exclusivo) &
        (df['price'] >= preco_range[0]) & (df['price'] <= preco_range[1]) &
        (df['room_type'].isin(tipo_quarto))
    ].dropna(subset=['review_scores_location', 'review_scores_value'])
    
    st.info(f"Exibindo **{len(df_bairro_unico)}** opções ponto a ponto processadas para o bairro: *{bairro_exclusivo}*.")
    
    fig5_exclusivo = px.scatter(df_bairro_unico, x="review_scores_location", y="review_scores_value",
                              color="price", size="accommodates", color_continuous_scale="Viridis",
                              template="plotly_white", height=550, labels=traducoes,
                              range_x=[1, 5], range_y=[1, 5], hover_data=['price', 'room_type'])
    
    fig5_exclusivo.add_hline(y=4.5, line_dash="dot", annotation_text="Custo-Benefício Alvo (4.5)", line_color="gray")
    fig5_exclusivo.add_vline(x=4.5, line_dash="dot", annotation_text="Localização Ideal (4.5)", line_color="gray")
    st.plotly_chart(fig5_exclusivo, use_container_width=True)

else:
    st.warning("⚠️ Amostra excessivamente reduzida para computar análises inferenciais estáveis. Ajuste os filtros.")