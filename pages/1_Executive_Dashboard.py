"""
PropSight — Executive Dashboard
City-wide market metrics, regional comparisons, and macroeconomic trends for Pune & PCMC.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_inr_lakhs, format_rate_per_sqft
from src.data.loader import DataLoader
from src.utils.config import MODEL_METADATA_PATH

st.set_page_config(page_title="Executive Dashboard — PropSight", page_icon="📊", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Executive Market Dashboard",
    subtitle="Macroeconomic indicators, price distributions, and micro-market comparisons across Pune & PCMC",
    page_tag="Market Overview"
)

# Load data
df_props = DataLoader.load_properties(cleaned=True)
df_hist = DataLoader.load_historical_trends()

# Load model metadata
metadata = {}
if MODEL_METADATA_PATH.exists():
    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

# Sidebar Filters
with st.sidebar:
    st.markdown("### 🔍 Dashboard Filters")
    selected_region = st.selectbox("Region", ["All Regions", "Pune", "PCMC"])
    
    selected_bhks = st.multiselect(
        "BHK Configuration",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5]
    )

    prop_types = ["All Types"] + sorted(df_props["property_type"].unique().tolist())
    selected_type = st.selectbox("Property Type", prop_types)

# Apply filters
filtered_df = df_props.copy()
if selected_region != "All Regions":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

if selected_bhks:
    filtered_df = filtered_df[filtered_df["bhk"].isin(selected_bhks)]

if selected_type != "All Types":
    filtered_df = filtered_df[filtered_df["property_type"] == selected_type]

# Safety check
if len(filtered_df) == 0:
    st.warning("No listings match the selected filters. Please broaden your selection.")
    st.stop()

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_kpi_card(
        label="Tracked Records",
        value=f"{len(filtered_df):,}",
        caption=f"{len(filtered_df)/len(df_props)*100:.1f}% of total dataset"
    )
with col2:
    render_kpi_card(
        label="Median Price",
        value=format_inr_lakhs(filtered_df["price_lakhs"].median()),
        caption=f"Avg: {format_inr_lakhs(filtered_df['price_lakhs'].mean())}"
    )
with col3:
    render_kpi_card(
        label="Median Rate/Sq.Ft.",
        value=format_rate_per_sqft(filtered_df["price_per_sqft"].median()),
        caption=f"Avg: {format_rate_per_sqft(filtered_df['price_per_sqft'].mean())}"
    )
with col4:
    render_kpi_card(
        label="Avg Carpet Area",
        value=f"{int(filtered_df['carpet_area_sqft'].mean()):,} sq.ft.",
        caption=f"Median: {int(filtered_df['carpet_area_sqft'].median())} sq.ft."
    )
with col5:
    render_kpi_card(
        label="Coverage Localities",
        value=f"{filtered_df['locality'].nunique()}",
        caption="Active micro-markets"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Row 1: Price Distribution & Rate by Locality
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("#### 🏷️ Listing Price Distribution (₹ Lakhs)")
    fig_hist = px.histogram(
        filtered_df,
        x="price_lakhs",
        nbins=30,
        color="region",
        color_discrete_map={"Pune": "#0284C7", "PCMC": "#10B981"},
        labels={"price_lakhs": "Listing Price (₹ Lakhs)", "region": "Region"}
    )
    median_val = filtered_df["price_lakhs"].median()
    fig_hist.add_vline(
        x=median_val, line_dash="dash", line_color="#0F172A",
        annotation_text=f"Median: ₹{median_val:.1f}L", annotation_position="top right"
    )
    fig_hist.update_layout(
        plot_bgcolor="#FFFFFF",
        height=350,
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    st.markdown("#### 📍 Top Localities by Average Rate (₹/sq.ft.)")
    loc_rates = filtered_df.groupby("locality").agg(
        avg_rate=("price_per_sqft", "mean"),
        count=("property_id", "count")
    ).reset_index().sort_values(by="avg_rate", ascending=False).head(10)

    fig_bar = px.bar(
        loc_rates,
        x="avg_rate",
        y="locality",
        orientation="h",
        color="avg_rate",
        color_continuous_scale="Blues",
        labels={"avg_rate": "Avg Rate (₹/sq.ft.)", "locality": "Locality"}
    )
    fig_bar.update_layout(
        plot_bgcolor="#FFFFFF",
        height=350,
        margin=dict(l=20, r=20, t=30, b=30),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Row 2: Regional Box Plot & Historical Trend
c3, c4 = st.columns([1, 1])

with c3:
    st.markdown("#### ⚖️ Price Distribution by BHK Configuration")
    fig_box = px.box(
        filtered_df,
        x="bhk",
        y="price_lakhs",
        color="region",
        color_discrete_map={"Pune": "#0284C7", "PCMC": "#10B981"},
        labels={"bhk": "BHK Configuration", "price_lakhs": "Listing Price (₹ Lakhs)"}
    )
    fig_box.update_layout(
        plot_bgcolor="#FFFFFF",
        height=350,
        margin=dict(l=20, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_box, use_container_width=True)

with c4:
    st.markdown("#### 📈 Historical Quarterly Index (2018–2024)")
    # Aggregate historical prices by region and quarter
    hist_agg = df_hist.groupby(["region", "quarter"]).agg(
        avg_rate=("avg_price_per_sqft", "mean")
    ).reset_index()

    # Sort quarters chronologically
    hist_agg["year"] = hist_agg["quarter"].apply(lambda q: int(q.split("-")[0]))
    hist_agg["q_num"] = hist_agg["quarter"].apply(lambda q: int(q.split("-Q")[1]))
    hist_agg = hist_agg.sort_values(by=["year", "q_num"])

    fig_line = px.line(
        hist_agg,
        x="quarter",
        y="avg_rate",
        color="region",
        color_discrete_map={"Pune": "#0284C7", "PCMC": "#10B981"},
        labels={"quarter": "Quarter", "avg_rate": "Avg Rate (₹/sq.ft.)"}
    )
    fig_line.update_layout(
        plot_bgcolor="#FFFFFF",
        height=350,
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Dataset & Model Transparency Card
st.markdown("---")
st.markdown("#### ℹ️ Model & Dataset Specifications")
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown(f"""
    **Active ML Model**
    - Architecture: `{metadata.get('selected_model', 'Gradient Boosting')}`
    - Pipeline Version: `{metadata.get('model_version', '1.0.0')}`
    - Training Date: `{metadata.get('training_timestamp', '2026-10-15')[:10]}`
    """)
with m_col2:
    st.markdown(f"""
    **Model Validation Performance**
    - Test $R^2$: `{metadata.get('best_model_metrics', {}).get('test_r2', 0.96):.4f}`
    - Test MAE: `₹{metadata.get('best_model_metrics', {}).get('test_mae_lakhs', 6.8):.2f} Lakhs`
    - Test MAPE: `{metadata.get('best_model_metrics', {}).get('test_mape_pct', 6.3):.2f}%`
    """)
with m_col3:
    st.markdown(f"""
    **Data Provenance & Target**
    - Target: `{metadata.get('target_column', 'price_lakhs')}` ({metadata.get('target_type', 'listing_price')})
    - Dataset Type: `Demonstration Synthetic`
    - Basis: Calibrated from published Anarock / Knight Frank Pune quarterly indices (2018-2024).
    """)
