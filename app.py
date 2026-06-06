import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="🎵 Spotify Dashboard Pro",
    layout="wide"
)

st.title("🎵 Spotify Dashboard Pro + ML")

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

    # LIMPIEZA BÁSICA
    df = df.fillna(0)

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.header("🎛️ Filtros")

    def safe_multiselect(label, column):
        if column in df.columns:
            return st.sidebar.multiselect(label, df[column].unique())
        return []

    artist = safe_multiselect("🎤 Artista", "artist_name")
    genre = safe_multiselect("🎧 Género", "genre")
    country = safe_multiselect("🌍 País", "country")

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year
        year = st.sidebar.multiselect("📅 Año", sorted(df["year"].dropna().unique()))
    else:
        year = []

    # FILTERS
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
    st.subheader("📊 KPIs")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🎵 Canciones", len(df))
    c2.metric("👨‍🎤 Artistas", df["artist_name"].nunique() if "artist_name" in df else 0)
    c3.metric("🌍 Países", df["country"].nunique() if "country" in df else 0)
    c4.metric("🔥 Popularidad media", round(df["popularity"].mean(), 2) if "popularity" in df else 0)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Popularidad",
        "🎧 Audio",
        "👨‍🎤 Artistas",
        "🌍 Países",
        "📊 Correlación",
        "🤖 ML"
    ])

    # =========================
    # TAB 1 - POPULARIDAD MEJORADA
    # =========================
    with tab1:
        st.subheader("📈 Distribución de Popularidad")

        if "popularity" in df.columns:

            fig = px.histogram(
                df,
                x="popularity",
                nbins=30,
                color_discrete_sequence=["#00D4FF"],
                template="plotly_dark"
            )

            fig.update_layout(
                bargap=0.1,
                title="Distribución de Popularidad",
                xaxis_title="Popularidad",
                yaxis_title="Cantidad de canciones"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 2 AUDIO (MEJORADO)
    # =========================
    with tab2:
        st.subheader("🎧 Audio Features")

        audio_cols = ["danceability", "energy", "tempo", "loudness"]
        colors = ["#00D4FF", "#FF4B4B", "#00CC96", "#AB63FA"]

        for i, col in enumerate(audio_cols):
            if col in df.columns:

                fig = px.histogram(
                    df,
                    x=col,
                    nbins=30,
                    color_discrete_sequence=[colors[i]],
                    template="plotly_dark"
                )

                fig.update_layout(
                    title=f"Distribución de {col}",
                    xaxis_title=col,
                    yaxis_title="Frecuencia",
                    bargap=0.1
                )

                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 3 ARTISTS
    # =========================
    with tab3:
        st.subheader("👨‍🎤 Top Artistas")

        if "artist_name" in df.columns and "stream_count" in df.columns:

            top = df.groupby("artist_name")["stream_count"].sum().nlargest(10).reset_index()

            fig = px.bar(
                top,
                x="stream_count",
                y="artist_name",
                orientation="h",
                color="stream_count",
                color_continuous_scale="Blues",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 4 COUNTRIES
    # =========================
    with tab4:
        st.subheader("🌍 Streams por País")

        if "country" in df.columns and "stream_count" in df.columns:

            topc = df.groupby("country")["stream_count"].sum().reset_index()

            fig = px.choropleth(
                topc,
                locations="country",
                locationmode="country names",
                color="stream_count",
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 5 CORRELATION (MEJORADO)
    # =========================
    with tab5:
        st.subheader("📊 Correlación")

        numeric = df.select_dtypes(include=np.number)

        if len(numeric.columns) > 1:

            fig = px.imshow(
                numeric.corr(),
                text_auto=True,
                color_continuous_scale="RdBu",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 6 ML (MEJORADO)
    # =========================
    with tab6:
        st.subheader("🤖 Machine Learning")

        if st.button("🚀 Entrenar modelo"):

            if "popularity" in df.columns:

                X = df.select_dtypes(include=np.number).drop(columns=["popularity"], errors="ignore")
                y = df["popularity"]

                X = X.fillna(0)

                if len(X.columns) > 0:

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    model = RandomForestRegressor(n_estimators=100)
                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    st.write("📊 Resultados:")
                    st.json({
                        "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                        "R2": r2_score(y_test, pred)
                    })

else:
    st.info("📂 Sube tu dataset para comenzar")
    