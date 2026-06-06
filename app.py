import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Spotify Dashboard Pro", layout="wide")

st.title("🎵 Spotify Dashboard + Machine Learning")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df


file = st.file_uploader("📁 Sube tu CSV de Spotify", type=["csv"])

if file:
    df = load_data(file)

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.header("🎛️ Filtros")

    if "artist_name" in df.columns:
        artist = st.sidebar.multiselect("Artista", df["artist_name"].dropna().unique())
    else:
        artist = []

    if "genre" in df.columns:
        genre = st.sidebar.multiselect("Género", df["genre"].dropna().unique())
    else:
        genre = []

    if "country" in df.columns:
        country = st.sidebar.multiselect("País", df["country"].dropna().unique())
    else:
        country = []

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year
        year = st.sidebar.multiselect("Año", sorted(df["year"].dropna().unique()))
    else:
        year = []

    # APPLY FILTERS
    if artist:
        df = df[df["artist_name"].isin(artist)]
    if genre:
        df = df[df["genre"].isin(genre)]
    if country:
        df = df[df["country"].isin(country)]
    if year:
        df = df[df["year"].isin(year)]

    # =========================
    # KPIs
    # =========================
    st.subheader("📊 KPIs Generales")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🎵 Canciones", len(df))
    c2.metric("👨‍🎤 Artistas", df["artist_name"].nunique() if "artist_name" in df else 0)
    c3.metric("🌍 Países", df["country"].nunique() if "country" in df else 0)
    c4.metric("🔥 Popularidad media", round(df["popularity"].mean(), 2) if "popularity" in df else 0)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 General",
        "🎧 Audio",
        "👨‍🎤 Artistas",
        "🌍 Países",
        "📊 Correlación",
        "🤖 ML"
    ])

    # =========================
    # TAB 1
    # =========================
    with tab1:
        st.subheader("📊 Popularidad")

        if "popularity" in df:
            fig = px.histogram(
                df,
                x="popularity",
                color_discrete_sequence=["#00D4FF"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Distribución de la popularidad de las canciones.")

        if "year" in df:
            fig = px.histogram(
                df,
                x="year",
                color_discrete_sequence=["#FF4B4B"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Canciones publicadas por año.")

    # =========================
    # TAB 2 AUDIO
    # =========================
    with tab2:
        audio_cols = ["danceability", "energy", "tempo", "loudness"]

        colors = ["#00D4FF", "#FF4B4B", "#00CC96", "#AB63FA"]

        for i, col in enumerate(audio_cols):
            if col in df.columns:
                fig = px.histogram(
                    df,
                    x=col,
                    color_discrete_sequence=[colors[i % len(colors)]]
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"📌 Distribución de {col} en las canciones.")

    # =========================
    # TAB 3 ARTISTS
    # =========================
    with tab3:
        if "artist_name" in df and "stream_count" in df:
            top = df.groupby("artist_name")["stream_count"].sum().nlargest(10).reset_index()

            fig = px.bar(
                top,
                x="artist_name",
                y="stream_count",
                color_discrete_sequence=["#00CC96"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Top 10 artistas con más streams.")

    # =========================
    # TAB 4 COUNTRIES
    # =========================
    with tab4:
        if "country" in df and "stream_count" in df:
            topc = df.groupby("country")["stream_count"].sum().reset_index()

            fig = px.choropleth(
                topc,
                locations="country",
                locationmode="country names",
                color="stream_count",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Streams por país a nivel mundial.")

    # =========================
    # TAB 5 CORRELATION
    # =========================
    with tab5:
        numeric = df.select_dtypes(include=np.number)

        if len(numeric.columns) > 1:
            corr = numeric.corr()

            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Correlación entre variables numéricas.")

        if "danceability" in df and "popularity" in df:
            fig = px.scatter(
                df,
                x="danceability",
                y="popularity",
                color_discrete_sequence=["#00D4FF"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📌 Relación entre danceability y popularidad.")

    # =========================
    # TAB 6 ML
    # =========================
    with tab6:
        st.subheader("🤖 Machine Learning")

        @st.cache_resource
        def train(df):
            results = {}

            # REGRESION
            if "popularity" in df.columns:
                X = df.select_dtypes(include=np.number).drop(columns=["popularity"], errors="ignore").fillna(0)
                y = df["popularity"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                results["reg"] = {
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                    "R2": r2_score(y_test, pred)
                }

            # CLASSIFICATION
            if "explicit" in df.columns:
                X = df.select_dtypes(include=np.number).fillna(0)
                y = df["explicit"].astype(int)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier(n_estimators=100)
                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                results["clf"] = {
                    "accuracy": accuracy_score(y_test, pred)
                }

            return results

        res = train(df)

        if "reg" in res:
            st.write("📈 Predicción Popularidad")
            st.json(res["reg"])

        if "clf" in res:
            st.write("🚨 Clasificación Explicit")
            st.write("Accuracy:", res["clf"]["accuracy"])

else:
    st.info("📂 Sube tu dataset para comenzar")