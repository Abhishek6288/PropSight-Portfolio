"""
PropSight — Locality Intelligence
Deep micro-market analytics, historical price trajectories, cluster scatters,
and side-by-side benchmarking for Pune & PCMC.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_inr_lakhs, format_rate_per_sqft, format_pct
from src.analytics.locality import LocalityAnalytics
from src.utils.charts import (
    apply_chart_theme,
    PROPSIGHT_COLORS,
    PROPSIGHT_MODEBAR_CONFIG
)

st.set_page_config(page_title="Locality Intelligence — PropSight", page_icon="📍", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Locality Intelligence & Analytics",
    subtitle="Micro-market profiles, historical price trends, sample size reliability, and cross-locality benchmarking",
    page_tag="Locality Analytics"
)

@st.cache_resource
def get_locality_analytics():
    return LocalityAnalytics()

analytics = get_locality_analytics()
localities = analytics.get_available_localities()

# Locality Selection
selected_loc = st.selectbox(
    "Select Primary Micro-Market for Analysis:",
    options=localities,
    index=localities.index("Baner") if "Baner" in localities else 0
)

# Fetch KPIs
kpis = analytics.get_locality_kpis(selected_loc)

# Reliability Warning if sample size is limited
if not kpis["is_reliable"]:
    st.markdown(f"""
    <div class="callout-box callout-warning">
        ⚠️ <strong>Sample Size Caution:</strong> {kpis['warning']}
    </div>
    """, unsafe_allow_html=True)

# Top KPI Summary Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card(
        label="Region & Tier",
        value=kpis["region"],
        caption=f"{kpis['tier']} Micro-market"
    )
with k2:
    render_kpi_card(
        label="Sample Listings",
        value=f"{kpis['sample_size']}",
        caption="Reliable" if kpis["is_reliable"] else "Limited Data"
    )
with k3:
    render_kpi_card(
        label="Median Price",
        value=format_inr_lakhs(kpis["median_price_lakhs"]),
        caption=f"Avg: {format_inr_lakhs(kpis['avg_price_lakhs'])}"
    )
with k4:
    render_kpi_card(
        label="Median Rate/Sq.Ft.",
        value=format_rate_per_sqft(kpis["median_rate_per_sqft"]),
        caption=f"Avg: {format_rate_per_sqft(kpis['avg_rate_per_sqft'])}"
    )
with k5:
    render_kpi_card(
        label="7-Year CAGR",
        value=format_pct(kpis["historical_cagr_pct"]),
        caption="2018-Q1 to 2024-Q4"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Charts Row 1: Historical Trend & Price Distribution
c1, c2 = st.columns([1.3, 1.0])

with c1:
    fig_hist = analytics.plot_historical_series(selected_loc, show_moving_avg=True)
    st.plotly_chart(fig_hist, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

with c2:
    fig_dist = analytics.plot_price_distribution(selected_loc)
    st.plotly_chart(fig_dist, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

# Charts Row 2: Price vs Area Scatter & BHK Donut
c3, c4 = st.columns([1.3, 1.0])

with c3:
    st.markdown(f"#### 📐 Property Landscape in {selected_loc}: Price vs. Carpet Area")
    fig_loc_scatter = analytics.plot_locality_scatter(selected_loc)
    st.plotly_chart(fig_loc_scatter, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

with c4:
    st.markdown("#### 🛏️ BHK Inventory Composition")
    bhk_data = kpis["bhk_distribution"]
    total_bhk_units = sum(bhk_data.values())
    df_bhk = pd.DataFrame({
        "BHK": [f"{k} BHK" for k in bhk_data.keys()],
        "Count": list(bhk_data.values())
    })
    
    donut_colors = [
        PROPSIGHT_COLORS["primary_teal"],
        PROPSIGHT_COLORS["blue_accent"],
        PROPSIGHT_COLORS["sky_accent"],
        PROPSIGHT_COLORS["indigo_accent"],
        PROPSIGHT_COLORS["amber_accent"]
    ]

    fig_bhk = px.pie(
        df_bhk,
        names="BHK",
        values="Count",
        hole=0.55,
        color_discrete_sequence=donut_colors
    )
    fig_bhk.update_traces(
        textposition="outside",
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Listings: %{value} units<br>Share: %{percent}<extra></extra>"
    )
    fig_bhk.add_annotation(
        text=f"<b>{total_bhk_units}</b><br><span style='font-size:10px; color:#64748B;'>Units</span>",
        x=0.5, y=0.5,
        font=dict(size=14, color=PROPSIGHT_COLORS["navy_primary"]),
        showarrow=False
    )
    apply_chart_theme(
        fig_bhk,
        title=f"Configuration Breakdown — {selected_loc}",
        height=350,
        show_legend=False
    )
    st.plotly_chart(fig_bhk, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

st.markdown("<br>", unsafe_allow_html=True)

# Row 3: Infrastructure Attributes
st.markdown("#### 🏢 Infrastructure & Locality Connectivity Profile")
meta_df = analytics.df_meta[analytics.df_meta["locality"] == selected_loc]
if len(meta_df) > 0:
    row = meta_df.iloc[0]
    metro_status = "✅ Operational / Under Construction" if row.get("metro_connectivity", False) else "❌ No Immediate Access"
    it_status = "✅ Major Tech Park / SEZ Hub" if row.get("it_hub_proximity", False) else "ℹ️ Primarily Residential / Mixed"
    pincode = row.get("pincode", "411001")
    lat = row.get("latitude", 18.5204)
    lon = row.get("longitude", 73.8567)

    st.markdown(f"""
    <div class="kpi-card" style="padding: 1.25rem;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; font-size: 0.88rem;">
            <div><strong>Municipal Corporation:</strong><br><span style="color: #0EA5E9; font-weight: 600;">{kpis['region']}</span></div>
            <div><strong>Market Tier:</strong><br><span style="color: #0F172A; font-weight: 600;">{kpis['tier']}</span></div>
            <div><strong>Postal Code:</strong><br><span style="color: #0F172A; font-weight: 600;">PIN {pincode}</span></div>
            <div><strong>Coordinates:</strong><br><span style="color: #64748B;">{lat:.4f}° N, {lon:.4f}° E</span></div>
            <div><strong>Metro Transit:</strong><br><span>{metro_status}</span></div>
            <div><strong>IT Corridor Proximity:</strong><br><span>{it_status}</span></div>
            <div><strong>Observed Price Range:</strong><br><span style="color: #0F172A; font-weight: 600;">₹{kpis['price_range_lakhs'][0]:.1f}L – ₹{kpis['price_range_lakhs'][1]:.1f}L</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Cross-Locality Comparative Benchmarking
st.markdown("### ⚖️ Side-by-Side Micro-Market Benchmarking")
default_compare = [selected_loc]
for loc in ["Wakad", "Kharadi", "Kothrud", "Ravet"]:
    if loc in localities and loc not in default_compare and len(default_compare) < 5:
        default_compare.append(loc)

compare_selection = st.multiselect(
    "Select Localities to Compare (2 to 5):",
    options=localities,
    default=default_compare
)

if len(compare_selection) >= 2:
    comp_df = analytics.compare_localities(compare_selection)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Comparative Interactive Chart
    comp_chart_col1, comp_chart_col2 = st.columns([1.5, 1.0])
    with comp_chart_col1:
        comp_metric = st.selectbox(
            "Comparative Metric:",
            ["Average Rate (₹/sq.ft.)", "Median Price (₹ Lakhs)", "Listings Count"],
            key="comp_chart_metric"
        )
        if "Rate" in comp_metric:
            y_col_target = "Avg Rate (₹/sq.ft.)"
            chart_title = "Average Price per Square Foot Benchmark"
        elif "Median" in comp_metric:
            y_col_target = "Median Price (₹ Lakhs)"
            chart_title = "Median Capital Value Benchmark"
        else:
            y_col_target = "Listings Count"
            chart_title = "Sample Inventory Count"

        fig_comp_bar = px.bar(
            comp_df,
            x="Locality",
            y=y_col_target,
            color="Region",
            color_discrete_map={"Pune": PROPSIGHT_COLORS["blue_accent"], "PCMC": PROPSIGHT_COLORS["emerald_accent"]},
            title=chart_title
        )
        fig_comp_bar.update_traces(
            texttemplate="%{y:,.0f}",
            textposition="outside",
            cliponaxis=False
        )
        apply_chart_theme(
            fig_comp_bar,
            title=chart_title,
            xaxis_title="Locality",
            yaxis_title=comp_metric,
            height=340
        )
        st.plotly_chart(fig_comp_bar, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

    with comp_chart_col2:
        # 7-Year CAGR comparison bar
        comp_df["CAGR_Num"] = comp_df["7-Year Historical CAGR"].str.rstrip("%").astype(float)
        fig_cagr = px.bar(
            comp_df,
            x="Locality",
            y="CAGR_Num",
            color="CAGR_Num",
            color_continuous_scale="Teal",
            title="7-Year Historical CAGR (%)"
        )
        fig_cagr.update_traces(texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False)
        apply_chart_theme(fig_cagr, title="7-Year Historical CAGR (%)", xaxis_title="Locality", yaxis_title="CAGR (%)", height=340)
        fig_cagr.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cagr, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

else:
    st.info("Select at least 2 localities above to enable comparative benchmarking.")
