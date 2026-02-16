"""
Streamlit Dashboard
====================
Interactive dashboard for the Climate-Adjusted Yield Resilience Model.

Launch with:
    streamlit run src/streamlit/app.py
"""

import streamlit as st

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="ClientCo ESG — Yield Resilience",
    page_icon="🌾",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.title("🌾 Yield Resilience")
st.sidebar.markdown("Navigate between views using the pages below.")

# ── Main page ────────────────────────────────────────────
st.title("Climate-Adjusted Yield Resilience Model")
st.markdown(
    """
    Welcome to the **ClientCo ESG** dashboard.

    Use the sidebar to explore:
    - **EDA** — Explore historical yield and climate data
    - **Model Results** — View predictions and performance metrics
    """
)
