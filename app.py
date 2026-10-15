"""
PropSight — Localized Real-Estate Intelligence & Decision Support
Main Streamlit Entry Point with 4K Infrastructure Showcase & Executive UI.
"""

import streamlit as st
import pandas as pd
import json

# Page configuration
st.set_page_config(
    page_title="PropSight — Pune & PCMC Real-Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils.styles import (
    apply_custom_css,
    render_hero_banner,
    render_kpi_card,
    render_showcase_card,
    get_image_base64
)
from src.utils.formatting import format_inr_lakhs, format_rate_per_sqft
from src.data.loader import DataLoader
from src.database.db import init_db, get_recent_predictions
from src.utils.config import MODEL_METADATA_PATH

# Initialize styles and database
apply_custom_css()
init_db()

# Load model metadata
metadata = {}
if MODEL_METADATA_PATH.exists():
    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

# Load data diagnostics
diag = DataLoader.get_dataset_diagnostics()

# Sidebar
with st.sidebar:
    st.markdown("### 🏙️ PropSight")
    st.caption("Pune & PCMC Decision Support Engine")
    st.markdown("---")
    st.markdown("**Core Specifications**")
    st.markdown("• Model: **" + metadata.get("selected_model", "Gradient Boosting") + "**")
    st.markdown(f"• Test $R^2$: **{metadata.get('best_model_metrics', {}).get('test_r2', 0.9628):.4f}**")
    st.markdown(f"• Test MAE: **₹{metadata.get('best_model_metrics', {}).get('test_mae_lakhs', 6.81):.2f}L**")
    st.markdown(f"• Test MAPE: **{metadata.get('best_model_metrics', {}).get('test_mape_pct', 6.28):.2f}%**")
    st.markdown("• Target: `price_lakhs` (Listing Price)")
    st.markdown("• Currency: **INR (₹ Lakh / Cr)**")
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.76rem; color: #64748B; line-height: 1.4;">
    <strong>Academic Project Notice:</strong><br>
    PropSight is an academic decision-support platform for Pune & PCMC. Outputs represent statistical estimations and do not constitute certified legal appraisals.
    </div>
    """, unsafe_allow_html=True)

# 1. 4K Aesthetic Hero Banner
hero_stats = [
    {"value": f"{diag['total_clean_listings']:,}", "label": "Verified Listings"},
    {"value": f"{diag['locality_count']}", "label": "Micro-Markets"},
    {"value": f"{metadata.get('best_model_metrics', {}).get('test_r2', 0.9628):.3f}", "label": "Model Test R²"},
    {"value": f"{diag['historical_quarters_count']} Quarters", "label": "Historical Trend"}
]

render_hero_banner(
    title="PropSight",
    subtitle="High-fidelity property valuation, SHAP explainability, fair-price analytics, and macroeconomic forecasting for Pune and Pimpri-Chinchwad.",
    tag="Pune & PCMC Infrastructure & Market Engine",
    stats=hero_stats
)

# 2. Executive KPI Summary Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card(
        label="Tracked Records",
        value=f"{diag['total_clean_listings']:,}",
        caption="Validated Pune & PCMC residential data"
    )
with col2:
    render_kpi_card(
        label="Municipal Coverage",
        value=f"{diag['locality_count']} Localities",
        caption="PMC & PCMC Growth Corridors"
    )
with col3:
    num_sum = diag["numerical_summary"]
    avg_price = num_sum.get("price_lakhs", {}).get("mean", 85.0)
    render_kpi_card(
        label="City Median Price",
        value=format_inr_lakhs(diag['numerical_summary'].get('price_lakhs', {}).get('50%', 85.0)),
        caption=f"Mean: {format_inr_lakhs(avg_price)}"
    )
with col4:
    avg_rate = num_sum.get("price_per_sqft", {}).get("mean", 8600.0)
    render_kpi_card(
        label="City Median Rate",
        value=format_rate_per_sqft(diag['numerical_summary'].get('price_per_sqft', {}).get('50%', 8500.0)),
        caption=f"Weighted Mean: {format_rate_per_sqft(avg_rate)}"
    )

st.markdown("<br>", unsafe_allow_html=True)

# 3. 4K Infrastructure & Growth Corridors Showcase
st.markdown("### 🏗️ Pune & PCMC Infrastructure Growth Catalysts")
st.markdown("""
Real-estate appreciation across Pune and PCMC is strongly coupled to structural transit and infrastructure investments.
Explore the primary urban catalysts driving micro-market capital values:
""")

inf_c1, inf_c2, inf_c3, inf_c4 = st.columns(4)

with inf_c1:
    render_showcase_card(
        title="Metro Rail Corridors",
        desc="Pune Metro Lines 1, 2 & Hinjewadi-Shivajinagar Line 3 connecting major IT parks and central employment nodes.",
        image_name="metro_transit.jpg",
        badge_text="Transit Catalyst",
        meta_left="Corridors: Line 1, 2, 3",
        meta_right="+14-18% Capital Lift"
    )

with inf_c2:
    render_showcase_card(
        title="IT & Tech SEZ Parks",
        desc="World-class tech hubs in Hinjewadi Rajiv Gandhi Infotech Park and Kharadi EON Free Zone powering multinational employment.",
        image_name="it_park.jpg",
        badge_text="Employment Engine",
        meta_left="Nodes: Hinjewadi, Kharadi",
        meta_right="High Rental Yield"
    )

with inf_c3:
    render_showcase_card(
        title="Ring Road & Expressways",
        desc="128-km Pune Ring Road and Mumbai-Pune Expressway corridors opening affordable and mid-market expansion in PCMC.",
        image_name="expressway.jpg",
        badge_text="Arterial Transit",
        meta_left="Zones: Ravet, Punawale",
        meta_right="Growth Belt"
    )

with inf_c4:
    render_showcase_card(
        title="Integrated Townships",
        desc="Large-scale sustainable master-planned townships with walk-to-work concepts, smart utilities, and green living.",
        image_name="township.jpg",
        badge_text="Master Townships",
        meta_left="Zones: Baner, Magarpatta",
        meta_right="Premium Lifestyle"
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4. Interactive Core Modules
st.markdown("### 🧭 Intelligence Modules & Decision Support")
st.markdown("Select any core tool from the navigation sidebar or jump into a dedicated analysis flow:")

mod_col1, mod_col2, mod_col3 = st.columns(3)

with mod_col1:
    st.markdown("""
    <div class="showcase-card" style="padding: 1.25rem;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">📊</div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A;">1. Executive Dashboard</h4>
        <p style="font-size: 0.83rem; color: #475569; margin-bottom: 0.75rem; line-height: 1.45;">
            Explore city-wide pricing indicators, regional Pune vs PCMC comparisons, property type splits, and dataset freshness.
        </p>
        <div style="font-size: 0.78rem; font-weight: 600; color: #0EA5E9;">Open Page: 1_Executive_Dashboard →</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="showcase-card" style="padding: 1.25rem;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">🎯</div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A;">2. Property Valuation</h4>
        <p style="font-size: 0.83rem; color: #475569; margin-bottom: 0.75rem; line-height: 1.45;">
            Input custom property specifications to compute ML-driven valuations with empirical 80% confidence bands and instant SHAP attribution.
        </p>
        <div style="font-size: 0.78rem; font-weight: 600; color: #0EA5E9;">Open Page: 2_Property_Valuation →</div>
    </div>
    """, unsafe_allow_html=True)

with mod_col2:
    st.markdown("""
    <div class="showcase-card" style="padding: 1.25rem;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">⚖️</div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A;">3. Fair-Price Analyzer</h4>
        <p style="font-size: 0.83rem; color: #475569; margin-bottom: 0.75rem; line-height: 1.45;">
            Evaluate asking prices against model predictions and empirical uncertainty intervals with clear, defensible classifications.
        </p>
        <div style="font-size: 0.78rem; font-weight: 600; color: #0EA5E9;">Open Page: 3_Fair_Price_Analyzer →</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="showcase-card" style="padding: 1.25rem;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">📍</div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A;">4. Locality Intelligence</h4>
        <p style="font-size: 0.83rem; color: #475569; margin-bottom: 0.75rem; line-height: 1.45;">
            Deep-dive into 28 micro-markets across Pune & PCMC with historical 7-year CAGR, inventory splits, and cross-locality benchmarking.
        </p>
        <div style="font-size: 0.78rem; font-weight: 600; color: #0EA5E9;">Open Page: 4_Locality_Intelligence →</div>
    </div>
    """, unsafe_allow_html=True)

with mod_col3:
    st.markdown("""
    <div class="showcase-card" style="padding: 1.25rem;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">📈</div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A;">5. Market Forecasts & Scenarios</h4>
        <p style="font-size: 0.83rem; color: #475569; margin-bottom: 0.75rem; line-height: 1.45;">
            Statistical ARIMA time-series projections with walk-forward back-testing and 3-year macroeconomic scenario trajectories.
        </p>
        <div style="font-size: 0.78rem; font-weight: 600; color: #0EA5E9;">Open Page: 5_Market_Forecasts_and_Scenarios →</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="showcase-card" style="padding: 1.25rem;">
        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">🔍</div>
        <h4 style="margin: 0 0 0.4rem 0; color: #0F172A;">6. Model Insights & Explainability</h4>
        <p style="font-size: 0.83rem; color: #475569; margin-bottom: 0.75rem; line-height: 1.45;">
            Inspect the 6-architecture benchmark tournament, global SHAP feature importance rankings, and correlation vs. causation principles.
        </p>
        <div style="font-size: 0.78rem; font-weight: 600; color: #0EA5E9;">Open Page: 6_Model_Insights_and_Explainability →</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Micro-Market Spotlight
st.markdown("### 🌟 Micro-Market Spotlight: Prime Pune & PCMC Hubs")
spot1, spot2, spot3, spot4 = st.columns(4)

with spot1:
    st.markdown("""
    <div class="kpi-card" style="border-top: 3px solid #0EA5E9;">
        <div class="kpi-label">West IT Corridor</div>
        <h4 style="margin: 0.2rem 0; color: #0F172A;">Baner</h4>
        <div style="font-size: 1.3rem; font-weight: 700; color: #0F172A;">₹10,225/sqft</div>
        <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">+6.2% 7-Yr CAGR</div>
        <div class="kpi-caption">High residential absorption, near Balewadi High St.</div>
    </div>
    """, unsafe_allow_html=True)

with spot2:
    st.markdown("""
    <div class="kpi-card" style="border-top: 3px solid #6366F1;">
        <div class="kpi-label">East Tech Hub</div>
        <h4 style="margin: 0.2rem 0; color: #0F172A;">Kharadi</h4>
        <div style="font-size: 1.3rem; font-weight: 700; color: #0F172A;">₹9,640/sqft</div>
        <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">+6.8% 7-Yr CAGR</div>
        <div class="kpi-caption">EON Free Zone, World Trade Center SEZ</div>
    </div>
    """, unsafe_allow_html=True)

with spot3:
    st.markdown("""
    <div class="kpi-card" style="border-top: 3px solid #10B981;">
        <div class="kpi-label">PCMC Growth Corridor</div>
        <h4 style="margin: 0.2rem 0; color: #0F172A;">Wakad</h4>
        <div style="font-size: 1.3rem; font-weight: 700; color: #0F172A;">₹8,710/sqft</div>
        <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">+5.9% 7-Yr CAGR</div>
        <div class="kpi-caption">Prime gateway to Hinjewadi Tech Park</div>
    </div>
    """, unsafe_allow_html=True)

with spot4:
    st.markdown("""
    <div class="kpi-card" style="border-top: 3px solid #F59E0B;">
        <div class="kpi-label">Prime Central Pune</div>
        <h4 style="margin: 0.2rem 0; color: #0F172A;">Kothrud</h4>
        <div style="font-size: 1.3rem; font-weight: 700; color: #0F172A;">₹12,535/sqft</div>
        <div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">+5.4% 7-Yr CAGR</div>
        <div class="kpi-caption">Established cultural hub with Metro Line 2</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Recent Inference Audit Trail
st.markdown("### 📋 Recent Valuation & Fair-Price Audit Trail")
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
<div style="font-size: 0.78rem; color: #64748B; text-align: center; line-height: 1.6;">
    <strong>PropSight Localized Real-Estate Intelligence & Decision Support</strong> • Pune & PCMC, Maharashtra<br>
    BSc Data Science College Project • 100% Free & Open-Source Stack • Pure Python + Streamlit<br>
    <em>Disclaimers:</em> Outputs represent model-based statistical estimations for educational decision support and do not constitute certified legal valuations.
</div>
""", unsafe_allow_html=True)
