import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Spotify Mega Pro Dashboard", layout="wide")

st.title("🎵 Spotify MEGA PRO Dashboard + AI")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df


file = st.file_uploader("📁 Sube tu dataset Spotify CSV", type=["csv"])

if not file:
    st.info("📂 Sube un archivo CSV para comenzar")
    st.stop()

df = load_data(file)

# =========================
# CLEAN DATA
# =========================
if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🎛️ Filtros")

def safe_multiselect(col):
    if col in df.columns:
        return st.sidebar.multiselect(col, df[col].dropna().unique())
    return []

artist = safe_multiselect("artist_name")
genre = safe_multiselect("genre")
country = safe_multiselect("country")

if "year" in df.columns:
    year = st.sidebar.multiselect("year", sorted(df["year"].dropna().unique()))
else:
    year = []

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

c1.metric("🎵 Songs", len(df))
c2.metric("👨‍🎤 Artists", df["artist_name"].nunique() if "artist_name" in df else 0)
c3.metric("🌍 Countries", df["country"].nunique() if "country" in df else 0)
c4.metric("🔥 Avg Popularity", round(df["popularity"].mean(), 2) if "popularity" in df else 0)

# =========================
# RANKING
# =========================
st.sidebar.header("📊 Ranking")

metric = st.sidebar.selectbox(
    "Ordenar por",
    [col for col in ["popularity", "stream_count", "danceability", "energy"] if col in df.columns]
)

st.subheader(f"🏆 Top 10 por {metric}")

top = df.sort_values(metric, ascending=False).head(10)

st.dataframe(top)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 General",
    "🎧 Audio",
    "👨‍🎤 Artists",
    "🌍 Countries",
    "📊 AI Insights",
    "🤖 Machine Learning"
])

# =========================
# TAB 1 - GENERAL
# =========================
with tab1:
    if "popularity" in df.columns:
        fig = px.histogram(df, x="popularity", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if "year" in df.columns:
        fig = px.histogram(df, x="year", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2 - AUDIO FEATURES
# =========================
with tab2:
    audio_cols = ["danceability", "energy", "tempo", "loudness"]
    colors = px.colors.qualitative.Set2

    for col in audio_cols:
        if col in df.columns:
            fig = px.histogram(df, x=col, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3 - ARTISTS
# =========================
with tab3:
    if "artist_name" in df.columns and "stream_count" in df.columns:
        top_artists = df.groupby("artist_name")["stream_count"].sum().nlargest(10).reset_index()

        fig = px.bar(top_artists, x="artist_name", y="stream_count", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 4 - COUNTRIES
# =========================
with tab4:
    if "country" in df.columns and "stream_count" in df.columns:
        country_data = df.groupby("country")["stream_count"].sum().reset_index()

        fig = px.choropleth(
            country_data,
            locations="country",
            locationmode="country names",
            color="stream_count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 5 - AI INSIGHTS
# =========================
with tab5:
    st.subheader("🧠 Insights automáticos")

    if "popularity" in df.columns and "artist_name" in df.columns:
        best_artist = df.groupby("artist_name")["popularity"].mean().idxmax()
        st.success(f"🔥 Top artista por popularidad: {best_artist}")

    if "year" in df.columns and "popularity" in df.columns:
        trend = df.groupby("year")["popularity"].mean().reset_index()

        fig = px.line(trend, x="year", y="popularity", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Recomendador
    st.subheader("🎧 Recomendador tipo Spotify")

    if "danceability" in df.columns and "energy" in df.columns:

        d = st.slider("Danceability", 0.0, 1.0, 0.5)
        e = st.slider("Energy", 0.0, 1.0, 0.5)

        rec = df.copy()
        rec["score"] = abs(rec["danceability"] - d) + abs(rec["energy"] - e)

        st.dataframe(rec.sort_values("score").head(10))

# =========================
# TAB 6 - MACHINE LEARNING
# =========================
with tab6:
    st.subheader("🤖 ML Models")

    @st.cache_resource
    def train_models(df):
        results = {}

        # REGRESSION
        if "popularity" in df.columns:
            X = df[numeric_cols].drop(columns=["popularity"], errors="ignore").fillna(0)
            y = df["popularity"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            results["reg"] = {
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
                "model": model,
                "features": X.columns
            }

        # CLASSIFICATION
        if "explicit" in df.columns:
            X = df[numeric_cols].fillna(0)
            y = df["explicit"].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            pred = clf.predict(X_test)

            results["clf"] = {
                "accuracy": accuracy_score(y_test, pred),
                "model": clf,
                "features": X.columns
            }

        return results

    res = train_models(df)

    if "reg" in res:
        st.write("📈 Regression Metrics")
        st.json(res["reg"])

        fi = pd.DataFrame({
            "feature": res["reg"]["features"],
            "importance": res["reg"]["model"].feature_importances_
        }).sort_values("importance", ascending=False)

        fig = px.bar(fi, x="feature", y="importance", template="plotly_dark")
        st.plotly_chart(fig)

    if "clf" in res:
        st.write("🚨 Classification Accuracy")
        st.metric("Accuracy", round(res["clf"]["accuracy"], 3))

# =========================
# CLUSTERING (BONUS PRO)
# =========================
st.subheader("🎯 Clustering de canciones")

if "danceability" in df.columns and "energy" in df.columns:
    Xc = df[["danceability", "energy"]].fillna(0)

    scaler = StandardScaler()
    Xc_scaled = scaler.fit_transform(Xc)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(Xc_scaled)

    fig = px.scatter(df, x="danceability", y="energy", color="cluster", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# FINAL MESSAGE
# =========================
st.success("🚀 Dashboard listo — versión MEGA PRO activada")