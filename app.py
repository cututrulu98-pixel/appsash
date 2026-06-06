import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Spotify ML Dashboard", layout="wide")

st.title("🎵 Spotify Analytics + Machine Learning Dashboard")

# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


file = st.file_uploader("📁 Sube tu dataset Spotify CSV", type=["csv"])

if file:
    df = load_data(file)

    # =========================
    # CLEANING
    # =========================
    df.columns = df.columns.str.strip()

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.header("🎛️ Filtros")

    if "artist_name" in df.columns:
        artist = st.sidebar.multiselect("Artista", df["artist_name"].dropna().unique())

    if "genre" in df.columns:
        genre = st.sidebar.multiselect("Género", df["genre"].dropna().unique())

    if "country" in df.columns:
        country = st.sidebar.multiselect("País", df["country"].dropna().unique())

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year
        year = st.sidebar.multiselect("Año", sorted(df["year"].dropna().unique()))

    # APPLY FILTERS
    if "artist_name" in df.columns and artist:
        df = df[df["artist_name"].isin(artist)]

    if "genre" in df.columns and genre:
        df = df[df["genre"].isin(genre)]

    if "country" in df.columns and country:
        df = df[df["country"].isin(country)]

    if "year" in df.columns and year:
        df = df[df["year"].isin(year)]

    # =========================
    # KPI ROW
    # =========================
    st.subheader("📊 KPIs Generales")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🎵 Canciones", len(df))
    col2.metric("👨‍🎤 Artistas", df["artist_name"].nunique() if "artist_name" in df else 0)
    col3.metric("🌍 Países", df["country"].nunique() if "country" in df else 0)
    col4.metric("🔥 Popularidad media", round(df["popularity"].mean(), 2) if "popularity" in df else 0)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 General",
        "🎧 Audio Features",
        "👨‍🎤 Artistas",
        "🌍 Países",
        "📊 Correlación",
        "🤖 ML Models"
    ])

    # =========================
    # TAB 1 - GENERAL
    # =========================
    with tab1:
        st.subheader("Distribución de Popularidad")

        if "popularity" in df:
            fig = px.histogram(df, x="popularity", nbins=50)
            st.plotly_chart(fig, use_container_width=True)

        if "year" in df:
            fig = px.histogram(df, x="year")
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 2 - AUDIO
    # =========================
    with tab2:
        audio_cols = ["danceability", "energy", "tempo", "loudness", "instrumentalness"]

        for col in audio_cols:
            if col in df.columns:
                fig = px.histogram(df, x=col, title=f"Distribución de {col}")
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 3 - ARTISTS
    # =========================
    with tab3:
        if "artist_name" in df and "stream_count" in df:
            top_artists = df.groupby("artist_name")["stream_count"].sum().nlargest(10).reset_index()

            fig = px.bar(top_artists, x="artist_name", y="stream_count")
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 4 - COUNTRIES
    # =========================
    with tab4:
        if "country" in df and "stream_count" in df:
            top_countries = df.groupby("country")["stream_count"].sum().reset_index()

            fig = px.choropleth(top_countries,
                                locations="country",
                                locationmode="country names",
                                color="stream_count")
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 5 - CORRELATION
    # =========================
    with tab5:
        numeric_df = df.select_dtypes(include=np.number)

        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()

            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        if "danceability" in df and "popularity" in df:
            fig = px.scatter(df, x="danceability", y="popularity")
            st.plotly_chart(fig, use_container_width=True)

        if "energy" in df and "popularity" in df:
            fig = px.scatter(df, x="energy", y="popularity")
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 6 - MACHINE LEARNING
    # =========================
    with tab6:
        st.subheader("🤖 Modelos Predictivos")

        @st.cache_resource
        def train_models(data):

            results = {}

            # =====================
            # REGRESIÓN POPULARITY
            # =====================
            if "popularity" in data.columns:
                features = data.select_dtypes(include=np.number).drop(columns=["popularity"], errors="ignore")
                target = data["popularity"]

                features = features.fillna(0)
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                results["regression"] = {
                    "MAE": mean_absolute_error(y_test, preds),
                    "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                    "R2": r2_score(y_test, preds)
                }

            # =====================
            # CLASIFICACIÓN EXPLICIT
            # =====================
            if "explicit" in data.columns:
                features = data.select_dtypes(include=np.number).drop(columns=[], errors="ignore")
                target = data["explicit"].astype(int)

                features = features.fillna(0)
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                results["classification"] = {
                    "accuracy": accuracy_score(y_test, preds),
                    "report": classification_report(y_test, preds, output_dict=True)
                }

            return results

        results = train_models(df)

        if "regression" in results:
            st.write("### 📈 Predicción Popularidad")
            st.json(results["regression"])

        if "classification" in results:
            st.write("### 🚨 Clasificación Explicit")
            st.write("Accuracy:", results["classification"]["accuracy"])

else:
    st.info("📂 Sube tu dataset CSV para iniciar el dashboard")