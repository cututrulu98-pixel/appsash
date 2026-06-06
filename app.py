import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard CSV", layout="wide")

st.title("📊 Dashboard de CSV")

file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Vista previa")
    st.dataframe(df)

    cols = df.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        x = st.selectbox("Eje X", cols)

    with col2:
        y = st.selectbox("Eje Y", cols)

    tipo = st.selectbox("Tipo de gráfica", ["scatter", "line", "bar"])

    if tipo == "scatter":
        fig = px.scatter(df, x=x, y=y)
    elif tipo == "line":
        fig = px.line(df, x=x, y=y)
    else:
        fig = px.bar(df, x=x, y=y)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Sube un archivo CSV para empezar")