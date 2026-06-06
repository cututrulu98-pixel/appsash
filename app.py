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
st.set_page_config(page_title="Spotify Dashboard", layout="wide")

st.title("Spotify Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

file = st.file_uploader("Sube tu dataset Spotify CSV", type=["csv"])

if not file:
    st.info("Sube un archivo CSV para comenzar")
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
st.sidebar.header("Filtros")

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
st.subheader("KPIs")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Canciones", len(df))
c2.metric("Artistas", df["artist_name"].nunique() if "artist_name" in df else 0)
c3.metric("Paises", df["country"].nunique() if "country" in df else 0)
c4.metric("Popularidad Promedio", round(df["popularity"].mean(), 2) if "popularity" in df else 0)

# =========================
# RANKING
# =========================
st.sidebar.header("Ranking")

metric = st.sidebar.selectbox(
    "Ordenar por",
    [col for col in ["popularity", "stream_count", "danceability", "energy"] if col in df.columns]
)

st.subheader(f"Top 10 por {metric}")

top = df.sort_values(metric, ascending=False).head(10)
st.dataframe(top)

# =========================
# TAB DESIGN
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "General",
    "Audio",
    "Artistas",
    "Paises",
    "Correlacion y clustering",
    "Machine Learning"
])

# =========================
# TAB 1 - GENERAL (MEJORADO)
# =========================
with tab1:
    if "popularity" in df.columns:
        fig = px.histogram(
            df,
            x="popularity",
            nbins=40,
            opacity=0.85,
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    if "year" in df.columns:
        fig = px.histogram(
            df,
            x="year",
            nbins=30,
            opacity=0.85,
            color_discrete_sequence=["#00CC96"]
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2 - AUDIO FEATURES (COLORES DIFERENTES)
# =========================
with tab2:
    audio_cols = ["danceability", "energy", "tempo", "loudness"]

    palette = ["#EF553B", "#00CC96", "#636EFA", "#AB63FA"]

    for i, col in enumerate(audio_cols):
        if col in df.columns:
            fig = px.histogram(
                df,
                x=col,
                opacity=0.75,
                color_discrete_sequence=[palette[i % len(palette)]],
            )
            fig.update_traces(marker_line_width=1, marker_line_color="black")
            st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3 - ARTISTS (MEJOR BAR)
# =========================
with tab3:
    if "artist_name" in df.columns and "stream_count" in df.columns:
        top_artists = df.groupby("artist_name")["stream_count"].sum().nlargest(10).reset_index()

        fig = px.bar(
            top_artists,
            x="artist_name",
            y="stream_count",
            color="stream_count",
            color_continuous_scale="Turbo"
        )

        fig.update_traces(marker_line_width=1, marker_line_color="black")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 4 - COUNTRIES (MEJOR MAPA)
# =========================
with tab4:
    if "country" in df.columns and "stream_count" in df.columns:
        country_data = df.groupby("country")["stream_count"].sum().reset_index()

        fig = px.choropleth(
            country_data,
            locations="country",
            locationmode="country names",
            color="stream_count",
            color_continuous_scale="Plasma"
        )

        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 5 - AI INSIGHTS (LINEA PRO)
# =========================
with tab5:
    st.subheader("Prespectivas automáticas")

    if "popularity" in df.columns and "artist_name" in df.columns:
        best_artist = df.groupby("artist_name")["popularity"].mean().idxmax()
        st.success(f"Top artistas: {best_artist}")

    if "year" in df.columns and "popularity" in df.columns:
        trend = df.groupby("year")["popularity"].mean().reset_index()

        fig = px.line(
            trend,
            x="year",
            y="popularity",
            markers=True
        )

        fig.update_traces(line=dict(width=3, color="#00CC96"))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 6 - ML
# =========================
with tab6:
    st.subheader("ML Modelos")

    @st.cache_resource
    def train_models(df):
        results = {}

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
        st.write("Feature Importance")

        fi = pd.DataFrame({
            "feature": res["reg"]["features"],
            "importance": res["reg"]["model"].feature_importances_
        }).sort_values("importance", ascending=False)

        fig = px.bar(
            fi,
            x="feature",
            y="importance",
            color="importance",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(fig)

    if "clf" in res:
        st.metric("Accuracy", round(res["clf"]["accuracy"], 3))

# =========================
# CLUSTERING
# =========================
st.subheader("Clustering")

if "danceability" in df.columns and "energy" in df.columns:
    Xc = df[["danceability", "energy"]].fillna(0)

    scaler = StandardScaler()
    Xc_scaled = scaler.fit_transform(Xc)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(Xc_scaled)

    fig = px.scatter(
        df,
        x="danceability",
        y="energy",
        color="cluster",
        opacity=0.7
    )

    fig.update_traces(marker_line_width=1, marker_line_color="black")

    st.plotly_chart(fig, use_container_width=True)

st.success("Dashboard Spotify")