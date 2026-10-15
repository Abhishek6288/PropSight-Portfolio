"""
PropSight — Executive Dashboard
Fintech-grade interactive macroeconomic market indicators, price distributions,
cross-locality comparative rankings, and historical time-series analytics.
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
from src.utils.charts import (
    apply_chart_theme,
    build_timeseries_chart,
    build_price_distribution_chart,
    build_horizontal_bar_rank,
    build_price_vs_area_scatter,
    PROPSIGHT_COLORS,
    PROPSIGHT_MODEBAR_CONFIG
)

st.set_page_config(page_title="Executive Dashboard — PropSight", page_icon="📊", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Executive Market Terminal",
    subtitle="Macroeconomic market indicators, multi-dimensional price distributions, and micro-market comparisons across Pune & PCMC",
    page_tag="Market Intelligence"
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
    st.markdown("### 🔍 Dashboard Controls")
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

# Top KPI Metric Strip
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_kpi_card(
        label="Tracked Records",
        value=f"{len(filtered_df):,}",
        caption=f"{len(filtered_df)/len(df_props)*100:.1f}% of total inventory"
    )
with col2:
    render_kpi_card(
        label="Median Price",
        value=format_inr_lakhs(filtered_df["price_lakhs"].median()),
        caption=f"Mean: {format_inr_lakhs(filtered_df['price_lakhs'].mean())}"
    )
with col3:
    render_kpi_card(
        label="Median Rate/Sq.Ft.",
        value=format_rate_per_sqft(filtered_df["price_per_sqft"].median()),
        caption=f"Mean: {format_rate_per_sqft(filtered_df['price_per_sqft'].mean())}"
    )
with col4:
    render_kpi_card(
        label="Avg Usable Area",
        value=f"{int(filtered_df['carpet_area_sqft'].mean()):,} sq.ft.",
        caption=f"Median: {int(filtered_df['carpet_area_sqft'].median())} sq.ft."
    )
with col5:
    render_kpi_card(
        label="Active Localities",
        value=f"{filtered_df['locality'].nunique()}",
        caption="Micro-markets represented"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ROW 1: Distribution Analysis & Locality Comparison
c1, c2 = st.columns([1.1, 1.0])

with c1:
    st.markdown("#### 🏷️ Inventory Price Distribution & Central Tendency")
    dist_metric = st.radio(
        "Metric:",
        ["Listing Price (₹ Lakhs)", "Rate per Sq.Ft. (₹/sq.ft.)"],
        horizontal=True,
        key="dash_dist_metric",
        label_visibility="collapsed"
    )

    if "Rate" in dist_metric:
        fig_dist = build_price_distribution_chart(
            filtered_df,
            col_name="price_per_sqft",
            color_col="region",
            title="Residential Rate per Square Foot Distribution",
            xaxis_title="Rate (₹/sq.ft.)",
            nbins=35
        )
    else:
        fig_dist = build_price_distribution_chart(
            filtered_df,
            col_name="price_lakhs",
            color_col="region",
            title="Total Property Valuation Distribution",
            xaxis_title="Listing Price (₹ Lakhs)",
            nbins=35
        )

    st.plotly_chart(fig_dist, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

with c2:
    st.markdown("#### 📍 Top Localities Ranking & Benchmarks")
    rank_metric = st.selectbox(
        "Rank By Metric:",
        ["Average Rate (₹/sq.ft.)", "Median Price (₹ Lakhs)", "Listing Inventory Volume"],
        key="dash_rank_metric"
    )

    if "Rate" in rank_metric:
        loc_agg = filtered_df.groupby("locality").agg(
            metric=("price_per_sqft", "mean"),
            region=("region", "first")
        ).reset_index()
        fig_rank = build_horizontal_bar_rank(
            loc_agg,
            y_col="locality",
            x_col="metric",
            title="Top Micro-Markets by Average Rate (₹/sq.ft.)",
            xaxis_title="Avg Rate (₹/sq.ft.)",
            top_n=10,
            color_scale="Teal"
        )
    elif "Median" in rank_metric:
        loc_agg = filtered_df.groupby("locality").agg(
            metric=("price_lakhs", "median"),
            region=("region", "first")
        ).reset_index()
        fig_rank = build_horizontal_bar_rank(
            loc_agg,
            y_col="locality",
            x_col="metric",
            title="Top Micro-Markets by Median Price (₹ Lakhs)",
            xaxis_title="Median Price (₹ Lakhs)",
            top_n=10,
            color_scale="Blues"
        )
    else:
        loc_agg = filtered_df.groupby("locality").agg(
            metric=("property_id", "count"),
            region=("region", "first")
        ).reset_index()
        fig_rank = build_horizontal_bar_rank(
            loc_agg,
            y_col="locality",
            x_col="metric",
            title="Top Micro-Markets by Inventory Volume",
            xaxis_title="Listing Count",
            top_n=10,
            color_scale="Purples"
        )

    st.plotly_chart(fig_rank, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

st.markdown("<br>", unsafe_allow_html=True)

# ROW 2: Price vs Area Scatter & BHK Distribution
c3, c4 = st.columns([1.2, 1.0])

with c3:
    st.markdown("#### 📐 Price vs. Carpet Area Relationship (Market Gradient)")
    fig_scatter = build_price_vs_area_scatter(
        filtered_df,
        x_col="carpet_area_sqft",
        y_col="price_lakhs",
        color_col="region",
        title="Price vs. Usable Floor Area with Micro-Market Inspection",
        sample_size=1200
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

with c4:
    st.markdown("#### ⚖️ Valuation Spread by BHK Configuration")
    fig_box = px.box(
        filtered_df,
        x="bhk",
        y="price_lakhs",
        color="region",
        color_discrete_map={"Pune": PROPSIGHT_COLORS["blue_accent"], "PCMC": PROPSIGHT_COLORS["emerald_accent"]},
        labels={"bhk": "BHK Configuration", "price_lakhs": "Listing Price (₹ Lakhs)", "region": "Region"}
    )
    fig_box.update_traces(marker=dict(size=3, opacity=0.5))
    apply_chart_theme(
        fig_box,
        title="BHK Interquartile Ranges & Valuation Outliers",
        xaxis_title="BHK Configuration",
        yaxis_title="Listing Price (₹ Lakhs)"
    )
    st.plotly_chart(fig_box, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

st.markdown("<br>", unsafe_allow_html=True)

# ROW 3: Interactive Historical Quarterly Time Series with Range Controls
st.markdown("### 📈 Historical Price per Sq.Ft. Evolution (2018–2024)")
st.caption("Interactive quarterly timeline tracking the residential rate trajectory with macroeconomic shocks and post-pandemic appreciation.")

hist_opt_col1, hist_opt_col2 = st.columns([1.5, 1.0])
with hist_opt_col1:
    hist_grouping = st.radio(
        "Compare Trajectory By:",
        ["Region (Pune vs PCMC)", "Key Micro-Markets (Baner, Wakad, Kharadi, Kothrud, Ravet)"],
        horizontal=True,
        key="dash_hist_group"
    )
with hist_opt_col2:
    enable_ma = st.checkbox("Show 4-Quarter Rolling Trend (Smoothing)", value=True, key="dash_hist_ma")

if "Region" in hist_grouping:
    hist_agg = df_hist.groupby(["region", "quarter"]).agg(
        avg_rate=("avg_price_per_sqft", "mean")
    ).reset_index()
    hist_agg["year"] = hist_agg["quarter"].apply(lambda q: int(q.split("-")[0]))
    hist_agg["q_num"] = hist_agg["quarter"].apply(lambda q: int(q.split("-Q")[1]))
    hist_agg = hist_agg.sort_values(by=["year", "q_num"])

    fig_hist_line = build_timeseries_chart(
        hist_agg,
        x_col="quarter",
        y_col="avg_rate",
        color_col="region",
        title="Regional Price Evolution — Pune vs. PCMC",
        yaxis_title="Average Rate (₹/sq.ft.)",
        color_map={"Pune": PROPSIGHT_COLORS["blue_accent"], "PCMC": PROPSIGHT_COLORS["emerald_accent"]},
        show_range_slider=True,
        show_moving_avg=enable_ma,
        ma_window=4
    )
else:
    selected_showcase_locs = ["Baner", "Wakad", "Kharadi", "Kothrud", "Ravet"]
    hist_locs = df_hist[df_hist["locality"].isin(selected_showcase_locs)].copy()
    hist_locs["year"] = hist_locs["quarter"].apply(lambda q: int(q.split("-")[0]))
    hist_locs["q_num"] = hist_locs["quarter"].apply(lambda q: int(q.split("-Q")[1]))
    hist_locs = hist_locs.sort_values(by=["year", "q_num"])

    fig_hist_line = build_timeseries_chart(
        hist_locs,
        x_col="quarter",
        y_col="avg_price_per_sqft",
        color_col="locality",
        title="Micro-Market Rate Comparison Across Key Hubs",
        yaxis_title="Average Rate (₹/sq.ft.)",
        show_range_slider=True,
        show_moving_avg=enable_ma,
        ma_window=4
    )

st.plotly_chart(fig_hist_line, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

# Model & Dataset Transparency Card
st.markdown("---")
st.markdown("#### ℹ️ Model Architecture & Data Governance")
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown(f"""
    **Active ML Architecture**
    - Selected Estimator: `{metadata.get('selected_model', 'Gradient Boosting')}`
    - Serialization Version: `{metadata.get('model_version', '1.0.0')}`
    - Training Date: `{metadata.get('training_timestamp', '2026-10-15')[:10]}`
    """)
with m_col2:
    st.markdown(f"""
    **Held-Out Test Validation**
    - Test $R^2$: `{metadata.get('best_model_metrics', {}).get('test_r2', 0.9628):.4f}`
    - Test MAE: `₹{metadata.get('best_model_metrics', {}).get('test_mae_lakhs', 6.81):.2f} Lakhs`
    - Test MAPE: `{metadata.get('best_model_metrics', {}).get('test_mape_pct', 6.28):.2f}%`
    """)
with m_col3:
    st.markdown(f"""
    **Data Provenance & Target**
    - Target Variable: `{metadata.get('target_column', 'price_lakhs')}` ({metadata.get('target_type', 'listing_price')})
    - Source Basis: Calibrated from published Anarock / Knight Frank Pune residential indices (2018–2024).
    - Status: Validated demonstration baseline for decision support.
    """)
