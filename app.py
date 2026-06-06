import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG PROFESIONAL
# =========================
st.set_page_config(
    page_title="Spotify Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Spotify Analytics Dashboard Pro")
st.caption("📊 Análisis completo de música + Machine Learning")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

file = st.file_uploader("📁 Sube tu dataset de Spotify (CSV)", type=["csv"])

if file:

    df = load_data(file)

    # LIMPIEZA
    df = df.fillna(0)

    # =========================
    # SIDEBAR (FILTROS PRO)
    # =========================
    st.sidebar.header("🎛️ Filtros")

    def multi(col, label):
        if col in df.columns:
            return st.sidebar.multiselect(label, df[col].unique())
        return []

    artist = multi("artist_name", "🎤 Artista")
    genre = multi("genre", "🎧 Género")
    country = multi("country", "🌍 País")

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year
        year = multi("year", "📅 Año")
    else:
        year = []

    # FILTROS
    if artist:
        df = df[df["artist_name"].isin(artist)]
    if genre:
        df = df[df["genre"].isin(genre)]
    if country:
        df = df[df["country"].isin(country)]
    if year:
        df = df[df["year"].isin(year)]

    # =========================
    # KPIs PRO
    # =========================
    st.subheader("📊 KPIs Generales")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🎵 Canciones", len(df))
    c2.metric("👨‍🎤 Artistas", df["artist_name"].nunique() if "artist_name" in df else 0)
    c3.metric("🌍 Países", df["country"].nunique() if "country" in df else 0)
    c4.metric("🔥 Popularidad media", round(df["popularity"].mean(), 2) if "popularity" in df else 0)

    # =========================
    # TABS PRO
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 General",
        "🎧 Audio Features",
        "👨‍🎤 Artistas",
        "🌍 Países",
        "📈 Correlación",
        "🤖 ML"
    ])

    # =========================
    # TAB 1 - GENERAL (CLARO Y PRO)
    # =========================
    with tab1:

        st.subheader("📊 Distribución de Popularidad")

        if "popularity" in df.columns:
            fig = px.histogram(
                df,
                x="popularity",
                nbins=30,
                color_discrete_sequence=["#00D4FF"],
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📅 Canciones por Año")
        if "year" in df.columns:
            fig = px.histogram(
                df,
                x="year",
                color_discrete_sequence=["#FF4B4B"],
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎧 Streams")
        if "stream_count" in df.columns:
            fig = px.histogram(
                df,
                x="stream_count",
                nbins=30,
                color_discrete_sequence=["#00CC96"],
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 2 - AUDIO FEATURES
    # =========================
    with tab2:

        audio = ["danceability", "energy", "tempo", "loudness"]
        colors = ["#00D4FF", "#FF4B4B", "#00CC96", "#AB63FA"]

        for i, col in enumerate(audio):
            if col in df.columns:
                st.subheader(f"🎧 {col}")

                fig = px.histogram(
                    df,
                    x=col,
                    nbins=30,
                    color_discrete_sequence=[colors[i]],
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # TAB 3 - ARTISTAS TOP
    # =========================
    with tab3:

        if "artist_name" in df.columns and "stream_count" in df.columns:

            st.subheader("👨‍🎤 Top 10 Artistas")

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
    # TAB 4 - PAÍSES
    # =========================
    with tab4:

        if "country" in df.columns and "stream_count" in df.columns:

            st.subheader("🌍 Streams por País")

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
    # TAB 5 - CORRELACIÓN
    # =========================
    with tab5:

        st.subheader("📈 Correlación entre Variables")

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
    # TAB 6 - MACHINE LEARNING PRO
    # =========================
    with tab6:

        st.subheader("🤖 Predicción de Popularidad")

        if st.button("🚀 Entrenar modelo"):

            if "popularity" in df.columns:

                X = df.select_dtypes(include=np.number).drop(columns=["popularity"], errors="ignore")
                y = df["popularity"]

                X = X.fillna(0)

                if len(X.columns) > 0:

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    model = RandomForestRegressor(n_estimators=120, random_state=42)
                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    st.success("Modelo entrenado correctamente")

                    st.json({
                        "MAE": round(mean_absolute_error(y_test, pred), 3),
                        "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 3),
                        "R2": round(r2_score(y_test, pred), 3)
                    })

else:
    st.info("📂 Sube tu archivo CSV para comenzar")