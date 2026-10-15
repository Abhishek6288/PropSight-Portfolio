"""
PropSight — Locality Intelligence
Deep micro-market analytics, historical price trajectories, and side-by-side benchmarking.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_inr_lakhs, format_rate_per_sqft, format_pct
from src.analytics.locality import LocalityAnalytics

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
c1, c2 = st.columns([1.2, 1.0])

with c1:
    fig_hist = analytics.plot_historical_series(selected_loc)
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    fig_dist = analytics.plot_price_distribution(selected_loc)
    st.plotly_chart(fig_dist, use_container_width=True)

# Charts Row 2: BHK Breakdown & Infrastructure Connectivity
c3, c4 = st.columns([1.0, 1.2])

with c3:
    st.markdown("#### 🛏️ BHK Inventory Composition")
    bhk_data = kpis["bhk_distribution"]
    df_bhk = pd.DataFrame({
        "BHK": [f"{k} BHK" for k in bhk_data.keys()],
        "Count": list(bhk_data.values())
    })
    fig_bhk = px.pie(
        df_bhk,
        names="BHK",
        values="Count",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig_bhk.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_bhk, use_container_width=True)

with c4:
    st.markdown("#### 🏢 Infrastructure & Locality Attributes")
    meta_df = analytics.df_meta[analytics.df_meta["locality"] == selected_loc]
    if len(meta_df) > 0:
        row = meta_df.iloc[0]
        metro_status = "✅ Operational / Under Construction" if row.get("metro_connectivity", False) else "❌ No Immediate Access"
        it_status = "✅ Major Tech Park / SEZ Hub" if row.get("it_hub_proximity", False) else "ℹ️ Primarily Residential / Mixed"
        pincode = row.get("pincode", "411001")
        lat = row.get("latitude", 18.5204)
        lon = row.get("longitude", 73.8567)

        st.markdown(f"""
        <div class="kpi-card">
            <table style="width: 100%; font-size: 0.88rem; line-height: 2;">
                <tr><td><strong>Municipal Region:</strong></td><td>{kpis['region']}</td></tr>
                <tr><td><strong>Market Category:</strong></td><td>{kpis['tier']}</td></tr>
                <tr><td><strong>PIN Code:</strong></td><td>{pincode}</td></tr>
                <tr><td><strong>Coordinates:</strong></td><td>{lat:.4f}° N, {lon:.4f}° E</td></tr>
                <tr><td><strong>Metro Connectivity:</strong></td><td>{metro_status}</td></tr>
                <tr><td><strong>IT Hub Proximity:</strong></td><td>{it_status}</td></tr>
                <tr><td><strong>Price Band:</strong></td><td>₹{kpis['price_range_lakhs'][0]:.1f}L – ₹{kpis['price_range_lakhs'][1]:.1f}L</td></tr>
            </table>
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
    "Select Localities to Compare:",
    options=localities,
    default=default_compare
)

if len(compare_selection) >= 2:
    comp_df = analytics.compare_localities(compare_selection)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Comparative Bar Chart
    fig_comp_bar = px.bar(
        comp_df,
        x="Locality",
        y="Avg Rate (₹/sq.ft.)",
        color="Region",
        color_discrete_map={"Pune": "#0284C7", "PCMC": "#10B981"},
        title="Comparative Rate per Square Foot (₹/sq.ft.)"
    )
    fig_comp_bar.update_layout(
        plot_bgcolor="#FFFFFF",
        height=340,
        margin=dict(l=20, r=20, t=40, b=30)
    )
    st.plotly_chart(fig_comp_bar, use_container_width=True)
else:
    st.info("Select at least 2 localities above to enable comparative benchmarking.")
