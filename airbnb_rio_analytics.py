import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Airbnb Rio - Estatística Aplicada", layout="wide")

# 2. DICIONÁRIO DE TRADUÇÃO (Interface em Português)
traducoes = {
    'price': 'Preço',
    'review_scores_rating': 'Avaliação Geral',
    'review_scores_value': 'Custo-Benefício',
    'review_scores_location': 'Localização',
    'accommodates': 'Acomodações',
    'availability_365': 'Disponibilidade (Dias/Ano)',
    'number_of_reviews': 'Total de Reviews',
    'neighbourhood_cleansed': 'Bairro',
    'room_type': 'Tipo de Quarto'
}

# 3. CARREGAMENTO E LIMPEZA
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'listings.csv')
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    df = pd.read_csv(file_path)
    
    # Limpeza Estatística
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df = df[df['price'] <= 10000] # Removendo outliers extremos
    
    # Padronização de notas
    for col in ['review_scores_rating', 'review_scores_value', 'review_scores_location']:
        df[col] = df[col].replace(0, np.nan)
    
    return df

df = load_data()

# 4. PAINEL LATERAL (Configurações e Filtros)
st.sidebar.header("⚙️ Configurações e Filtros")

if not df.empty:
    lista_bairros = sorted(df['neighbourhood_cleansed'].unique())
    selecao_bairros = st.sidebar.multiselect("Bairros:", options=lista_bairros, default=['Copacabana', 'Ipanema', 'Leblon'])

    limite_razoavel = float(df['price'].quantile(0.95))
    preco_range = st.sidebar.slider("Faixa de Preço (R$):", 0.0, 10000.0, (0.0, limite_razoavel))

    tipo_quarto = st.sidebar.multiselect("Tipo de Quarto:", options=df['room_type'].unique(), default=df['room_type'].unique())

    # Aplicação dos Filtros
    df_filt = df[
        (df['neighbourhood_cleansed'].isin(selecao_bairros)) &
        (df['price'] >= preco_range[0]) & (df['price'] <= preco_range[1]) &
        (df['room_type'].isin(tipo_quarto))
    ]

    # Botão de Exportação para o Aluno/Professor
    st.sidebar.divider()
    csv = df_filt.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(label="📥 Baixar Dados Filtrados", data=csv, file_name='airbnb_filtrado.csv', mime='text/csv')
else:
    st.error("Erro: Base de dados não encontrada.")
    st.stop()

# 5. TÍTULO E INTRODUÇÃO ACADÊMICA
st.title("📊 Airbnb Rio de Janeiro: Análise Estatística")

# INTEGRANDO A CONTEXTUALIZAÇÃO E A PERGUNTA
with st.expander("📖 Sobre este Projeto: Contextualização e Pergunta de Pesquisa"):
    st.markdown("""
    ### 🏨 Contextualização
    Este projeto utiliza dados do **Inside Airbnb**, uma iniciativa que extrai dados públicos da plataforma para analisar o impacto do aluguel de curto prazo nas cidades. 
    No **Rio de Janeiro**, o mercado hoteleiro informal é fortemente influenciado pela geografia costeira e pela infraestrutura urbana, criando 
    padrões de preços e satisfação que variam drasticamente entre bairros.
    
    ### ❓ Pergunta de Pesquisa
    **Qual é a relação de dependência entre o preço, a localização e a percepção de custo-benefício dos hóspedes no mercado do Airbnb no Rio de Janeiro?**
    
    *O objetivo é identificar se o investimento em bairros de alta demanda reflete proporcionalmente na satisfação do hóspede ou se 
    existem distorções estatísticas onde bairros menos valorizados oferecem melhor retorno emocional e financeiro.*
    """)

st.markdown(f"Amostra atual: **{len(df_filt)}** acomodações filtradas.")

# 6. MÉTRICAS E GRÁFICOS
if len(df_filt) > 0:
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Preço Médio", f"R$ {df_filt['price'].mean():.2f}")
    with m2: st.metric("Custo-Benefício Médio", f"{df_filt['review_scores_value'].mean():.2f}")
    with m3: st.metric("Qtd Bairros", len(df_filt['neighbourhood_cleansed'].unique()))

    st.divider()

    # 1. Mapa
    st.write("### 1. Distribuição Geográfica de Preços")
    fig1 = px.scatter_mapbox(df_filt, lat="latitude", lon="longitude", color="price", 
                            size="accommodates", color_continuous_scale="Viridis",
                            mapbox_style="carto-positron", zoom=10, height=500,
                            labels={'price': 'Preço', 'accommodates': 'Acomodações'})
    fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # 2 e 3. Boxplot e Barras
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 2. Preços por Bairro")
        fig2 = px.box(df_filt, x="neighbourhood_cleansed", y="price", color="room_type",
                      template="plotly_white", height=450, labels=traducoes)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.write("### 3. Disponibilidade Anual")
        df_filt['availability_months'] = (df_filt['availability_365'] / 30.5).round(0).clip(lower=1, upper=12)
        df_meses = df_filt['availability_months'].value_counts().reset_index()
        df_meses.columns = ['Meses', 'Contagem']
        df_meses = df_meses.sort_values('Meses')
        fig3 = px.bar(df_meses, x="Meses", y="Contagem", color="Contagem", color_continuous_scale="Blues",
                      template="plotly_white", height=450)
        fig3.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # 4. Custo-Benefício
    st.write("### 4. Ranking de Custo-Benefício vs. Localização por Bairro")
    df_cb = df_filt.groupby('neighbourhood_cleansed')[['review_scores_value', 'review_scores_location']].mean().reset_index()
    fig4 = px.scatter(df_cb, x="review_scores_location", y="review_scores_value",
                      text="neighbourhood_cleansed", size_max=60,
                      template="plotly_white", height=500,
                      labels=traducoes,
                      color="review_scores_value", color_continuous_scale="Greens")
    fig4.update_traces(textposition='top center')
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # 5. Matriz de Correlação
    st.write("### 5. Matriz de Correlação Estatística")
    cols_int = ['price', 'review_scores_rating', 'review_scores_value', 'review_scores_location', 'accommodates']
    df_corr = df_filt[cols_int].corr().rename(index=traducoes, columns=traducoes)
    fig5 = px.imshow(df_corr, text_auto=".2f", aspect="auto",
                     color_continuous_scale='RdBu_r', template="plotly_white", height=600) 
    st.plotly_chart(fig5, use_container_width=True)

else:
    st.warning("⚠️ Nenhum dado encontrado. Ajuste os filtros no painel lateral.")