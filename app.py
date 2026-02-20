"""
Climate-Adjusted Yield Resilience Dashboard
=============================================
Interactive Streamlit dashboard for barley yield analysis.

Launch:
    .venv/bin/streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import requests

from constants.path import GOLD_PATH
from src.data.silver_to_gold import climate_features_year

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="ClientCo ESG — Yield Resilience",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #0f4c3a 0%, #1a6b4f 50%, #2d8b6e 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(15, 76, 58, 0.3);
}
.main-header h1 {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: rgba(255,255,255,0.8);
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafb 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.kpi-delta {
    font-size: 0.8rem;
    margin-top: 0.3rem;
}
.kpi-delta.positive { color: #16a34a; }
.kpi-delta.negative { color: #dc2626; }

/* Winner/Loser badges */
.winner-badge {
    display: inline-block;
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #166534;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}
.loser-badge {
    display: inline-block;
    background: linear-gradient(135deg, #fef2f2, #fecaca);
    color: #991b1b;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_data(show_spinner=False)
def load_gold():
    """Load historical gold dataset."""
    return pd.read_parquet(GOLD_PATH / "gold.parquet")


@st.cache_data(show_spinner=False)
def load_departments():
    """Load department code → name mapping."""
    return pd.read_parquet(GOLD_PATH / "department.parquet")


@st.cache_data(show_spinner=False)
def load_projections(scenario: str):
    """Load pre-computed projection features."""
    return pd.read_parquet(GOLD_PATH / f"projections_{scenario}.parquet")


@st.cache_data(show_spinner=False)
def load_france_geojson():
    """Load GeoJSON of French departments."""
    url = (
        "https://france-geojson.gregoiredavid.fr/"
        "repo/departements.geojson"
    )
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained XGBoost model."""
    with open(MODELS_DIR / "xgb_yield.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def kpi_card(label: str, value: str, delta: str = "", positive: bool = True):
    """Render a styled KPI card."""
    delta_html = ""
    if delta:
        cls = "positive" if positive else "negative"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def prepare_features_for_prediction(proj_df, feature_cols, departments):
    """
    Build the full feature matrix for projection data so it matches
    the trained model's expected columns.
    """
    # The projection df has code_dep + year + climate features.
    # We need to replicate the same feature engineering as training:
    #   1. One-hot encode code_dep
    #   2. Create climate × department interactions
    #   3. Drop raw department dummies
    #   4. Align columns with feature_cols

    df = proj_df.copy()

    # One-hot encode
    df = pd.get_dummies(df, columns=["code_dep"], drop_first=False)

    # Interaction features
    CLIMATE_INTERACT = ["heat_days", "hot_dry_days", "total_precip", "gdd"]
    dep_dummies = [c for c in df.columns if c.startswith("code_dep_")]
    for v in CLIMATE_INTERACT:
        if v in df.columns:
            for d in dep_dummies:
                df[f"{v}_x_{d}"] = df[v] * df[d]
    df = df.drop(columns=dep_dummies, errors="ignore")

    # Drop non-feature columns
    df = df.drop(columns=["year"], errors="ignore")

    # Align to training columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols].astype(np.float32)

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature grouping for SHAP-style explainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEATURE_GROUPS = {
    "Baseline (Lags / Dept)": [
        "code_dep", "_x_code_dep",
    ],
    "Water (Precipitation)": [
        "precip", "rain", "wet", "dry",
    ],
    "Heat (Tmax > 30°C / ZD49)": [
        "heat", "tmax", "gdd", "hot_dry", "frost",
        "m4_", "m5_",
    ],
}


def classify_feature(feat_name: str) -> str:
    """Classify a feature into one of the SHAP groups."""
    feat_lower = feat_name.lower()
    # Check Baseline first (interaction terms)
    for kw in FEATURE_GROUPS["Baseline (Lags / Dept)"]:
        if kw in feat_lower:
            return "Baseline (Lags / Dept)"
    # Check Heat
    for kw in FEATURE_GROUPS["Heat (Tmax > 30°C / ZD49)"]:
        if kw in feat_lower:
            return "Heat (Tmax > 30°C / ZD49)"
    # Check Water
    for kw in FEATURE_GROUPS["Water (Precipitation)"]:
        if kw in feat_lower:
            return "Water (Precipitation)"
    # Default to Heat for temperature features
    if "temp" in feat_lower or "tmean" in feat_lower:
        return "Heat (Tmax > 30°C / ZD49)"
    return "Baseline (Lags / Dept)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gold_df = load_gold()
dept_df = load_departments()
dept_map = dict(zip(dept_df["code_dep"], dept_df["nom_dep"]))

st.sidebar.markdown("## 🌾 Yield Resilience")
st.sidebar.markdown("---")

view = st.sidebar.radio(
    "📌 Navigation",
    ["Historical", "Projections"],
    index=0,
)

st.sidebar.markdown("---")

# Department filter
dept_names = sorted(dept_df["nom_dep"].tolist())
filter_level = st.sidebar.selectbox(
    "🔎 Filter Level",
    ["National"] + dept_names,
    index=0,
)

# Scenario selector (projections only)
scenario = None
if view == "Projections":
    st.sidebar.markdown("---")
    scenario = st.sidebar.selectbox(
        "🎯 Climate Scenario",
        ["Middle", "Worst"],
        index=0,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
subtitle = "Historical Analysis (1982–2018)" if view == "Historical" \
    else f"Projections (2018–2050) — {scenario} Scenario"

st.markdown(f"""
<div class="main-header">
    <h1>🌾 Climate-Adjusted Yield Resilience</h1>
    <p>{subtitle} · {'National' if filter_level == 'National' else filter_level}</p>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HISTORICAL VIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if view == "Historical":

    # ── Filter data ───────────────────────────────────────
    if filter_level == "National":
        df_view = gold_df.groupby("year").agg(
            {"yield": "mean", "production": "sum", "area": "sum"}
        ).reset_index()
    else:
        code = dept_df[dept_df["nom_dep"] == filter_level]["code_dep"].iloc[0]
        df_view = gold_df[gold_df["code_dep"] == code].copy()

    df_view = df_view.sort_values("year")

    # ── KPI Cards ─────────────────────────────────────────
    latest = df_view[df_view["year"] == df_view["year"].max()].iloc[0]
    prev_row = df_view[df_view["year"] == df_view["year"].max() - 1]

    yield_val = latest["yield"]
    prod_val = latest["production"]
    area_val = latest["area"]

    # Compute deltas
    if len(prev_row) > 0:
        prev = prev_row.iloc[0]
        y_delta = (yield_val - prev["yield"]) / prev["yield"] * 100
        p_delta = (prod_val - prev["production"]) / prev["production"] * 100
        a_delta = (area_val - prev["area"]) / prev["area"] * 100
    else:
        y_delta = p_delta = a_delta = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            kpi_card(
                "Yield (t/ha)",
                f"{yield_val:.2f}",
                f"{abs(y_delta):.1f}% vs prev year",
                y_delta >= 0,
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            kpi_card(
                "Production (tonnes)",
                f"{prod_val:,.0f}",
                f"{abs(p_delta):.1f}% vs prev year",
                p_delta >= 0,
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            kpi_card(
                "Area (ha)",
                f"{area_val:,.0f}",
                f"{abs(a_delta):.1f}% vs prev year",
                a_delta >= 0,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Yield vs Area Chart with 2011 crash highlight ─────
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_view["year"],
            y=df_view["yield"],
            name="Yield (t/ha)",
            line=dict(color="#0ea5e9", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(14,165,233,0.08)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_view["year"],
            y=df_view["area"],
            name="Area (ha)",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    # 2011 crash annotation
    fig.add_vrect(
        x0=2010.5, x1=2011.5,
        fillcolor="rgba(220,38,38,0.12)",
        layer="below",
        line=dict(width=0),
    )
    fig.add_annotation(
        x=2011,
        y=df_view[df_view["year"] == 2011]["yield"].values[0]
            if 2011 in df_view["year"].values else df_view["yield"].mean(),
        text="<b>2011 Crash</b><br>Heat + Drought",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#dc2626",
        font=dict(color="#dc2626", size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#dc2626",
        borderwidth=1,
        borderpad=6,
        ax=40,
        ay=-50,
    )

    fig.update_layout(
        title=dict(
            text="Yield & Area Evolution",
            font=dict(size=18, color="#0f172a"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        margin=dict(l=60, r=60, t=60, b=40),
        height=450,
    )
    fig.update_xaxes(
        title="Year", gridcolor="rgba(0,0,0,0.05)",
        dtick=5,
    )
    fig.update_yaxes(
        title_text="Yield (t/ha)", secondary_y=False,
        gridcolor="rgba(0,0,0,0.05)",
    )
    fig.update_yaxes(
        title_text="Area (ha)", secondary_y=True,
        gridcolor="rgba(0,0,0,0.03)",
    )

    st.plotly_chart(fig, width="stretch")

    # ── France Choropleth Map ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Yield Map by Department")

    geojson = load_france_geojson()
    years = sorted(gold_df["year"].unique())
    map_year = st.slider(
        "Select year",
        min_value=int(years[0]),
        max_value=int(years[-1]),
        value=int(years[-1]),
        step=1,
        key="map_year_hist",
    )

    # Average yield per department for selected year
    map_data = (
        gold_df[gold_df["year"] == map_year]
        .groupby("code_dep", as_index=False)["yield"]
        .mean()
    )
    # Merge department names
    map_data = map_data.merge(dept_df, on="code_dep", how="left")
    map_data["nom_dep"] = (
        map_data["nom_dep"].str.replace("_", " ")
    )

    fig_map = px.choropleth(
        map_data,
        geojson=geojson,
        locations="code_dep",
        featureidkey="properties.code",
        color="yield",
        color_continuous_scale=[
            [0.0, "#fef3c7"],
            [0.3, "#fbbf24"],
            [0.5, "#84cc16"],
            [0.7, "#22c55e"],
            [1.0, "#047857"],
        ],
        hover_name="nom_dep",
        hover_data={"yield": ":.2f", "code_dep": False},
        labels={"yield": "Yield (t/ha)"},
    )
    fig_map.update_geos(
        fitbounds="locations",
        visible=False,
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
        title=dict(
            text=f"Average Barley Yield — {map_year}",
            font=dict(size=16, color="#0f172a"),
            x=0.5,
        ),
        coloraxis_colorbar=dict(
            title="t/ha",
            thickness=15,
            len=0.7,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig_map, width="stretch")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECTION VIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif view == "Projections":

    model, feature_cols = load_model()
    proj_df = load_projections(scenario.lower())
    departments = dept_df.copy()

    # ── Generate predictions ──────────────────────────────
    proj_with_meta = proj_df.copy()
    X_proj = prepare_features_for_prediction(proj_df, feature_cols, departments)
    proj_with_meta["predicted_yield"] = model.predict(X_proj)

    # ── Historical baseline (last known year) ─────────────
    hist_last = gold_df[gold_df["year"] == gold_df["year"].max()].copy()
    hist_national = hist_last.groupby("year").agg(
        {"yield": "mean"}
    ).reset_index()

    # ── Forecast Chart ────────────────────────────────────
    st.markdown("### 📈 Yield Forecast")

    if filter_level == "National":
        proj_national = (
            proj_with_meta
            .groupby("year")
            .agg({"predicted_yield": "mean"})
            .reset_index()
        )
        # Prepend historical tail for continuity
        hist_tail = (
            gold_df.groupby("year")
            .agg({"yield": "mean"})
            .reset_index()
            .rename(columns={"yield": "predicted_yield"})
        )
        hist_tail = hist_tail[hist_tail["year"] >= 2015]
        chart_df = pd.concat([hist_tail, proj_national], ignore_index=True)
        chart_df = chart_df.sort_values("year")
    else:
        code = dept_df[dept_df["nom_dep"] == filter_level]["code_dep"].iloc[0]
        proj_dept = proj_with_meta[proj_with_meta["code_dep"] == code].copy()
        proj_chart = proj_dept[["year", "predicted_yield"]].sort_values("year")

        hist_dept = (
            gold_df[gold_df["code_dep"] == code][["year", "yield"]]
            .rename(columns={"yield": "predicted_yield"})
        )
        hist_dept = hist_dept[hist_dept["year"] >= 2015]
        chart_df = pd.concat([hist_dept, proj_chart], ignore_index=True)
        chart_df = chart_df.sort_values("year")

    fig_proj = go.Figure()

    # Historical segment
    hist_seg = chart_df[chart_df["year"] <= 2018]
    proj_seg = chart_df[chart_df["year"] >= 2018]

    fig_proj.add_trace(go.Scatter(
        x=hist_seg["year"],
        y=hist_seg["predicted_yield"],
        name="Historical",
        line=dict(color="#64748b", width=2),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    scenario_color = "#0ea5e9" if scenario == "Middle" else "#ef4444"
    fig_proj.add_trace(go.Scatter(
        x=proj_seg["year"],
        y=proj_seg["predicted_yield"],
        name=f"Forecast ({scenario})",
        line=dict(color=scenario_color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({14 if scenario == 'Middle' else 239},"
                  f"{165 if scenario == 'Middle' else 68},"
                  f"{233 if scenario == 'Middle' else 68},0.08)",
        mode="lines",
    ))

    # Divider at 2018
    fig_proj.add_vline(
        x=2018, line_dash="dash", line_color="#94a3b8",
        annotation_text="Today", annotation_position="top",
    )

    fig_proj.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(l=60, r=40, t=40, b=40),
        height=400,
        yaxis_title="Predicted Yield (t/ha)",
        xaxis_title="Year",
    )
    fig_proj.update_xaxes(gridcolor="rgba(0,0,0,0.05)", dtick=5)
    fig_proj.update_yaxes(gridcolor="rgba(0,0,0,0.05)")

    st.plotly_chart(fig_proj, width="stretch")

    # ── Strategic Ranking ─────────────────────────────────
    st.markdown("### 🏆 Strategic Department Ranking")

    # Compare 2020 yield vs 2050 yield per department
    early = proj_with_meta[proj_with_meta["year"] <= 2022].copy()
    late = proj_with_meta[proj_with_meta["year"] >= 2045].copy()

    early_avg = (
        early.groupby("code_dep")["predicted_yield"]
        .mean().rename("yield_early")
    )
    late_avg = (
        late.groupby("code_dep")["predicted_yield"]
        .mean().rename("yield_late")
    )
    ranking = pd.DataFrame({"yield_early": early_avg, "yield_late": late_avg})
    ranking["change_pct"] = (
        (ranking["yield_late"] - ranking["yield_early"])
        / ranking["yield_early"] * 100
    )
    ranking = ranking.dropna()
    ranking = ranking.merge(
        dept_df, left_index=True, right_on="code_dep", how="left"
    )
    ranking = ranking.sort_values("change_pct", ascending=False)

    col_w, col_l = st.columns(2)

    with col_w:
        st.markdown("#### 🟢 Winners — Resilient Departments")
        winners = ranking.head(5)
        for i, row in winners.iterrows():
            name = row["nom_dep"].replace("_", " ")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:0.5rem 0.8rem;margin:0.3rem 0;'
                f'background:#f0fdf4;border-radius:8px;border-left:3px solid #22c55e;">'
                f'<span style="font-weight:600;color:#15803d;">{name}</span>'
                f'<span class="winner-badge">+{row["change_pct"]:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_l:
        st.markdown("#### 🔴 Losers — De-prioritize")
        losers = ranking.tail(5).iloc[::-1]
        for i, row in losers.iterrows():
            name = row["nom_dep"].replace("_", " ")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:0.5rem 0.8rem;margin:0.3rem 0;'
                f'background:#fef2f2;border-radius:8px;border-left:3px solid #ef4444;">'
                f'<span style="font-weight:600;color:#991b1b;">{name}</span>'
                f'<span class="loser-badge">{row["change_pct"]:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP-style Feature Importance Explainer ───────────
    st.markdown("### 🔬 Feature Importance Explainer")
    st.caption(
        "Feature importances from the XGBoost model, "
        "grouped into **Baseline**, **Water**, and **Heat** categories."
    )

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    )

    # Group importances
    group_imp = {}
    for feat, imp_val in importances.items():
        grp = classify_feature(feat)
        group_imp.setdefault(grp, 0.0)
        group_imp[grp] += imp_val

    # Normalise to percentages
    total = sum(group_imp.values())
    group_pct = {k: v / total * 100 for k, v in group_imp.items()}

    # Sort for display
    groups_sorted = sorted(group_pct.items(), key=lambda x: x[1], reverse=True)

    colors_map = {
        "Baseline (Lags / Dept)": "#8b5cf6",
        "Water (Precipitation)": "#0ea5e9",
        "Heat (Tmax > 30°C / ZD49)": "#ef4444",
    }

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        y=[g[0] for g in groups_sorted],
        x=[g[1] for g in groups_sorted],
        orientation="h",
        marker=dict(
            color=[colors_map.get(g[0], "#94a3b8") for g in groups_sorted],
            line=dict(width=0),
        ),
        text=[f"{g[1]:.1f}%" for g in groups_sorted],
        textposition="outside",
        textfont=dict(size=13, color="#0f172a"),
    ))

    fig_shap.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=200, r=80, t=20, b=40),
        height=220,
        xaxis_title="Share of Total Importance (%)",
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12),
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.05)",
            range=[0, max(g[1] for g in groups_sorted) * 1.25],
        ),
    )

    st.plotly_chart(fig_shap, width="stretch")

    # ── Top features within each group ────────────────────
    with st.expander("🔍 Top features per group"):
        for grp_name, _ in groups_sorted:
            feats_in_group = {
                f: v for f, v in importances.items()
                if classify_feature(f) == grp_name
            }
            top5 = sorted(
                feats_in_group.items(), key=lambda x: x[1], reverse=True
            )[:5]
            st.markdown(f"**{grp_name}**")
            for f_name, f_imp in top5:
                bar_width = int(f_imp / importances.max() * 100)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'margin:2px 0;">'
                    f'<code style="min-width:280px;font-size:0.8rem;">'
                    f'{f_name}</code>'
                    f'<div style="background:{colors_map.get(grp_name, "#94a3b8")};'
                    f'height:12px;width:{bar_width}%;border-radius:3px;'
                    f'min-width:4px;"></div>'
                    f'<span style="font-size:0.8rem;color:#64748b;">'
                    f'{f_imp:.4f}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("")


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.8rem;">'
    'ClientCo ESG · Climate-Adjusted Yield Resilience Model · 2025'
    '</div>',
    unsafe_allow_html=True,
)
