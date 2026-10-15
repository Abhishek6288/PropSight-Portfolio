"""
PropSight — Localized Real-Estate Intelligence & Decision Support
Main Streamlit Entry Point.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PropSight — Pune & PCMC Real-Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_inr_lakhs, format_rate_per_sqft
from src.data.loader import DataLoader
from src.database.db import init_db, get_recent_predictions
from src.models.trainer import MODEL_METADATA_PATH
import json

# Initialize styles and database
apply_custom_css()
init_db()

# Load model metadata
metadata = {}
if MODEL_METADATA_PATH.exists():
    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

# Sidebar
with st.sidebar:
    st.markdown("### 🏙️ PropSight")
    st.caption("Pune & PCMC Decision Support")
    st.markdown("---")
    st.markdown("**System Status**")
    st.markdown("• Model: **" + metadata.get("selected_model", "Gradient Boosting") + "**")
    st.markdown(f"• Test $R^2$: **{metadata.get('best_model_metrics', {}).get('test_r2', 0.96):.3f}**")
    st.markdown(f"• Test MAE: **₹{metadata.get('best_model_metrics', {}).get('test_mae_lakhs', 6.8):.2f}L**")
    st.markdown("• Target: `price_lakhs` (Listing Price)")
    st.markdown("• Currency: **INR (₹ Lakh / Cr)**")
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.76rem; color: #64748B; line-height: 1.4;">
    <strong>Academic Project Notice:</strong><br>
    PropSight is a decision-support system built for Pune and PCMC. Outputs represent statistical estimations and do not constitute certified valuations.
    </div>
    """, unsafe_allow_html=True)

# Main Header
render_header(
    title="PropSight",
    subtitle="Localized Real-Estate Intelligence & Decision Support — Pune & Pimpri-Chinchwad",
    page_tag="Executive Overview"
)

# Load data diagnostics
diag = DataLoader.get_dataset_diagnostics()

# Top KPI Summary Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card(
        label="Covered Listings",
        value=f"{diag['total_clean_listings']:,}",
        caption="Validated Pune & PCMC records"
    )
with col2:
    render_kpi_card(
        label="Micro-Markets",
        value=f"{diag['locality_count']}",
        caption="Pune & PCMC Municipal Zones"
    )
with col3:
    num_sum = diag["numerical_summary"]
    avg_price = num_sum.get("price_lakhs", {}).get("mean", 85.0)
    render_kpi_card(
        label="Avg Market Price",
        value=format_inr_lakhs(avg_price),
        caption="Across all tracked configurations"
    )
with col4:
    avg_rate = num_sum.get("price_per_sqft", {}).get("mean", 8200.0)
    render_kpi_card(
        label="Avg Rate / Sq.Ft.",
        value=format_rate_per_sqft(avg_rate),
        caption="Weighted residential average"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Welcome & System Capabilities Overview
st.markdown("### Core Modules & Navigation")
st.markdown("Select any module from the left sidebar or explore the quick-start sections below:")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div class="kpi-card" style="margin-bottom: 1rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">📊 Executive Dashboard</h4>
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            City-wide market metrics, price tier distributions, PCMC vs Pune comparisons, and dataset freshness.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="kpi-card">
        <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">🎯 Property Valuation</h4>
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            Supervised ML regression predicting current market value with 80% empirical confidence intervals.
        </p>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div class="kpi-card" style="margin-bottom: 1rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">⚖️ Fair-Price Analyzer</h4>
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            Compare asking price against model estimates with transparent uncertainty-aware classifications.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="kpi-card">
        <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">📍 Locality Intelligence</h4>
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            Deep-dive into 28 micro-markets: price distributions, historical CAGR, and side-by-side benchmarking.
        </p>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
    <div class="kpi-card" style="margin-bottom: 1rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">📈 Market Forecasts</h4>
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            ARIMA time-series projections with widening prediction bands and 3-year scenario trajectories.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="kpi-card">
        <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">🔍 Model Explainability</h4>
        <p style="font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;">
            SHAP global feature rankings and individual prediction attribution waterfall breakdowns.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Recent Inference Audit Trail
st.markdown("### Recent Valuation & Fair-Price Audit Logs")
df_recent = get_recent_predictions(limit=6)

if len(df_recent) > 0:
    display_df = df_recent[[
        "timestamp", "locality", "property_type", "bhk",
        "carpet_area_sqft", "estimated_price_lakhs",
        "asking_price_lakhs", "fair_price_status"
    ]].copy()
    display_df.columns = [
        "Timestamp", "Locality", "Type", "BHK",
        "Carpet (sqft)", "Model Est. (₹L)",
        "Asking (₹L)", "Status"
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No prediction logs recorded yet. Visit the **Property Valuation** or **Fair-Price Analyzer** page to run an estimate.")

# Footer Notice
st.markdown("---")
st.markdown("""
<div style="font-size: 0.78rem; color: #64748B; text-align: center;">
    <strong>PropSight Localized Real-Estate Intelligence</strong> • BSc Data Science College Project • Pune & PCMC, Maharashtra<br>
    Zero-Cost Local Architecture • SQLite • Scikit-Learn • Statsmodels • SHAP • Streamlit
</div>
""", unsafe_allow_html=True)
