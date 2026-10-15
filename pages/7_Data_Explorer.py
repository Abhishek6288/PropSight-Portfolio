"""
PropSight — Data Explorer & Data Quality
Transparent dataset inspection, schema definitions, missingness diagnostics,
outlier auditing, and data provenance disclosures.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.data.loader import DataLoader
from src.data.schema import PROPERTY_SCHEMA_TYPES

st.set_page_config(page_title="Data Explorer — PropSight", page_icon="📁", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Data Explorer & Data Quality Audit",
    subtitle="Schema inspection, missingness checks, outlier detection diagnostics, and data provenance",
    page_tag="Data Governance"
)

# Load data and diagnostics
df_props = DataLoader.load_properties(cleaned=True)
df_raw = DataLoader.load_properties(cleaned=False)
diag = DataLoader.get_dataset_diagnostics()

# Governance & Transparency Banner
st.markdown(f"""
<div class="callout-box" style="border-left-color: #0EA5E9;">
    <strong>Data Provenance & Ethical Disclosure (PRD §7 & §17):</strong><br>
    • <strong>Dataset Classification:</strong> <span class="badge badge-synthetic">Demonstration Synthetic</span> (Academic Demonstration Dataset)<br>
    • <strong>Calibration Source:</strong> {diag['data_source']}<br>
    • <strong>Target Variable:</strong> <code>{diag['target_column']}</code> (Type: <code>{diag['target_type']}</code>)<br>
    • <strong>Academic Disclaimer:</strong> This dataset has been calibrated to mirror the statistical distributions, micro-market pricing tiers, and quarterly cyclical trends of Pune and PCMC. It is intended strictly for BSc Data Science evaluation and decision-support modeling.
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card(
        label="Raw Listings",
        value=f"{diag['total_raw_listings']:,}",
        caption="Before deduplication & cleaning"
    )
with k2:
    render_kpi_card(
        label="Cleaned Listings",
        value=f"{diag['total_clean_listings']:,}",
        caption="Validated for ML inference"
    )
with k3:
    render_kpi_card(
        label="Tracked Localities",
        value=f"{diag['locality_count']}",
        caption="Pune & PCMC Micro-Markets"
    )
with k4:
    render_kpi_card(
        label="Historical Series",
        value=f"{diag['historical_quarters_count']} Quarters",
        caption=diag["historical_date_range"]
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs: Data Browser, Summary Stats, Quality & Outliers, Data Dictionary
tab_browse, tab_stats, tab_quality, tab_dict = st.tabs([
    "🔍 Browse Records",
    "📊 Statistical Summaries",
    "🛡️ Quality & Outlier Diagnostics",
    "📖 Data Dictionary & Provenance"
])

# TAB 1: Browse Records
with tab_browse:
    st.markdown("### 📋 Interactive Property Record Explorer")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_loc = st.selectbox("Filter Locality:", ["All"] + sorted(df_props["locality"].unique().tolist()))
    with col_f2:
        f_bhk = st.multiselect("Filter BHK:", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    browse_df = df_props.copy()
    if f_loc != "All":
        browse_df = browse_df[browse_df["locality"] == f_loc]
    if f_bhk:
        browse_df = browse_df[browse_df["bhk"].isin(f_bhk)]

    st.dataframe(browse_df.head(100), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(100, len(browse_df))} of {len(browse_df)} matching records.")

# TAB 2: Statistical Summaries
with tab_stats:
    st.markdown("### 📈 Numerical Feature Distributions")
    num_cols = ["carpet_area_sqft", "price_lakhs", "price_per_sqft", "bhk", "bathrooms", "floor", "property_age_years"]
    desc_df = df_props[num_cols].describe().T[[
        "count", "mean", "std", "min", "25%", "50%", "75%", "max"
    ]].round(2)
    st.dataframe(desc_df, use_container_width=True)

    st.markdown("#### Locality Inventory Distribution")
    loc_counts = df_props["locality"].value_counts().reset_index()
    loc_counts.columns = ["Locality", "Listing Count"]
    fig_loc = px.bar(
        loc_counts,
        x="Locality",
        y="Listing Count",
        color="Listing Count",
        color_continuous_scale="Blues",
        title="Listing Distribution across Pune & PCMC Micro-Markets"
    )
    fig_loc.update_layout(plot_bgcolor="#FFFFFF", height=340, xaxis=dict(tickangle=-45))
    st.plotly_chart(fig_loc, use_container_width=True)

# TAB 3: Quality & Outlier Diagnostics
with tab_quality:
    st.markdown("### 🛡️ Data Cleaning and Integrity Audit")
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("#### Missing Values Audit")
        missing_df = pd.DataFrame({
            "Column": list(diag["missing_values"].keys()),
            "Missing Count": list(diag["missing_values"].values()),
            "Missing %": [f"{(v/diag['total_raw_listings'])*100:.2f}%" for v in diag["missing_values"].values()]
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    with col_q2:
        st.markdown("#### Outlier & Bound Checks")
        st.markdown("""
        The pipeline executes multi-stage statistical sanity checks:
        - **Carpet Area Sanity:** Minimum 250 sq.ft., maximum 10,000 sq.ft.
        - **Price Bounds:** Minimum ₹10 Lakhs, maximum ₹2,500 Lakhs (₹25 Cr).
        - **Area Hierarchy:** Built-up area verified to equal or exceed carpet area.
        - **Locality IQR Filtering:** Outliers exceeding $2.5 \\times \\text{IQR}$ on locality-level price per square foot are removed.
        """)

    # Target Distribution Boxplot
    st.markdown("#### Locality-Level Price/Sq.Ft. Distribution (Boxplot)")
    fig_box = px.box(
        df_props,
        x="locality",
        y="price_per_sqft",
        color="region",
        color_discrete_map={"Pune": "#0284C7", "PCMC": "#10B981"},
        labels={"locality": "Locality", "price_per_sqft": "Price/Sq.Ft. (₹)"}
    )
    fig_box.update_layout(plot_bgcolor="#FFFFFF", height=380, xaxis=dict(tickangle=-45))
    st.plotly_chart(fig_box, use_container_width=True)

# TAB 4: Data Dictionary
with tab_dict:
    st.markdown("### 📖 Entity Schema & Field Definitions")
    dict_records = [
        {"Field": "property_id", "Type": "String", "Description": "Unique identifier for property record (e.g. PUN-100045)"},
        {"Field": "locality", "Type": "Categorical", "Description": "Recognized micro-market in Pune or PCMC"},
        {"Field": "region", "Type": "Categorical", "Description": "Municipal Corporation jurisdiction: Pune or PCMC"},
        {"Field": "property_type", "Type": "Categorical", "Description": "Apartment, Villa / Independent House, Penthouse, Builder Floor"},
        {"Field": "bhk", "Type": "Integer", "Description": "Bedroom-Hall-Kitchen configuration (1 to 5)"},
        {"Field": "carpet_area_sqft", "Type": "Float", "Description": "Net usable floor area inside walls (RERA standard)"},
        {"Field": "built_up_area_sqft", "Type": "Float", "Description": "Gross area including external walls and balcony"},
        {"Field": "floor", "Type": "Integer", "Description": "Vertical floor position in building (0 = Ground)"},
        {"Field": "total_floors", "Type": "Integer", "Description": "Total building height in stories"},
        {"Field": "property_age_years", "Type": "Float", "Description": "Building age in years (0 = Under Construction)"},
        {"Field": "bathrooms", "Type": "Integer", "Description": "Number of bathroom facilities"},
        {"Field": "furnishing", "Type": "Categorical", "Description": "Unfurnished, Semi-Furnished, or Fully Furnished"},
        {"Field": "possession_status", "Type": "Categorical", "Description": "Ready to Move vs Under Construction"},
        {"Field": "price_lakhs", "Type": "Float", "Description": "TARGET: Listing price in INR Lakhs (1 Lakh = ₹100,000)"},
        {"Field": "price_per_sqft", "Type": "Float", "Description": "Derived: (price_lakhs * 100,000) / carpet_area_sqft"},
        {"Field": "target_type", "Type": "String", "Description": "listing_price (asking/listed rate rather than deed-registered)"},
        {"Field": "is_synthetic", "Type": "Boolean", "Description": "True for calibrated demonstration data"}
    ]
    st.dataframe(pd.DataFrame(dict_records), use_container_width=True, hide_index=True)
