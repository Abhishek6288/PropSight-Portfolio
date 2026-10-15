"""
PropSight — Fair-Price Analyzer
Compares seller asking prices against model estimates with uncertainty-aware thresholding.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_inr_lakhs, format_pct, format_rate_per_sqft
from src.utils.config import (
    SUPPORTED_LOCALITIES,
    PROPERTY_TYPES,
    FURNISHING_STATUSES,
    POSSESSION_STATUSES,
    AMENITIES_LIST
)
from src.valuation.fair_price import FairPriceAnalyzer, FairPriceResult
from src.data.loader import DataLoader
from src.utils.charts import (
    build_valuation_gauge_bullet,
    apply_chart_theme,
    PROPSIGHT_COLORS,
    PROPSIGHT_MODEBAR_CONFIG
)

st.set_page_config(page_title="Fair-Price Analyzer — PropSight", page_icon="⚖️", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Fair-Price Analyzer",
    subtitle="Assess seller asking prices against model predictions and empirical uncertainty intervals",
    page_tag="Decision Support"
)

# Load engine with caching
@st.cache_resource
def get_analyzer():
    return FairPriceAnalyzer()

analyzer = get_analyzer()

# Retrieve pre-filled payload from session state if available
default_payload = st.session_state.get("last_valuation_payload", {})

col_form, col_output = st.columns([1.1, 1.4])

with col_form:
    st.markdown("### 🏷️ Asking Price & Property Details")

    with st.form("fair_price_form"):
        # Asking Price (Mandatory)
        asking_input = st.number_input(
            "Seller's Asking Price (₹ Lakhs)",
            min_value=10.0,
            max_value=2500.0,
            value=float(default_payload.get("price_lakhs", 95.0)),
            step=1.0,
            help="Enter the total listed asking price in Lakhs (1 Lakh = ₹100,000)"
        )

        st.markdown("---")
        st.caption("Property Attributes for Model Valuation:")

        f_loc = st.selectbox(
            "Locality",
            options=sorted(SUPPORTED_LOCALITIES),
            index=sorted(SUPPORTED_LOCALITIES).index(default_payload.get("locality", "Baner")) if default_payload.get("locality") in SUPPORTED_LOCALITIES else 0
        )
        f_type = st.selectbox("Property Type", options=PROPERTY_TYPES, index=0)

        bhk_col, carpet_col = st.columns(2)
        with bhk_col:
            f_bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5], index=int(default_payload.get("bhk", 2)) - 1)
        with carpet_col:
            f_carpet = st.number_input("Carpet Area (sq.ft.)", min_value=300.0, max_value=8000.0, value=float(default_payload.get("carpet_area_sqft", 850.0)), step=25.0)

        fl_col, tot_col = st.columns(2)
        with fl_col:
            f_floor = st.number_input("Floor Level", min_value=0, max_value=50, value=int(default_payload.get("floor", 4)), step=1)
        with tot_col:
            f_total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=int(default_payload.get("total_floors", 14)), step=1)

        bath_col, age_col = st.columns(2)
        with bath_col:
            f_bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5, 6], index=min(f_bhk - 1, 5))
        with age_col:
            f_age = st.number_input("Property Age (Years)", min_value=0.0, max_value=40.0, value=float(default_payload.get("property_age_years", 2.0)), step=0.5)

        furn_col, poss_col = st.columns(2)
        with furn_col:
            f_furnishing = st.selectbox("Furnishing", options=FURNISHING_STATUSES, index=1)
        with poss_col:
            f_possession = st.selectbox("Possession", options=POSSESSION_STATUSES, index=0)

        submit_btn = st.form_submit_button("⚖️ Analyze Asking Price", use_container_width=True)

with col_output:
    st.markdown("### 📊 Fair-Price Assessment")

    if submit_btn:
        if f_floor > f_total_floors:
            st.error(f"Validation Error: Floor ({f_floor}) cannot exceed total building floors ({f_total_floors}).")
        else:
            property_payload = {
                "locality": f_loc,
                "property_type": f_type,
                "bhk": f_bhk,
                "carpet_area_sqft": f_carpet,
                "floor": f_floor,
                "total_floors": f_total_floors,
                "property_age_years": f_age,
                "bathrooms": f_bathrooms,
                "balconies": max(1, f_bhk - 1),
                "parking_spaces": 1 if f_bhk <= 2 else 2,
                "furnishing": f_furnishing,
                "possession_status": f_possession,
                "amenities_count": 5
            }

            with st.spinner("Analyzing asking price against model expectation..."):
                res: FairPriceResult = analyzer.analyze(
                    property_data=property_payload,
                    asking_price_lakhs=asking_input,
                    log_to_db=True
                )

            # Classification Badge Callout
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight: 600;">Pricing Status</span>
                    <span class="badge {res.badge_style}" style="font-size: 0.9rem; padding: 0.3rem 0.75rem;">{res.classification}</span>
                </div>
                <div style="font-size: 0.95rem; color: #1E293B; line-height: 1.5;">
                    {res.classification_rationale}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key Comparison KPIs
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card(
                    label="Asking Price",
                    value=format_inr_lakhs(res.asking_price_lakhs),
                    caption=f"₹{res.asking_price_lakhs * 100000 / f_carpet:,.0f}/sq.ft."
                )
            with k2:
                render_kpi_card(
                    label="Model Estimate",
                    value=format_inr_lakhs(res.estimated_price_lakhs),
                    caption=f"₹{res.estimated_price_lakhs * 100000 / f_carpet:,.0f}/sq.ft."
                )
            with k3:
                render_kpi_card(
                    label="Deviation",
                    value=format_pct(res.percentage_diff_pct),
                    caption=f"₹{res.absolute_diff_lakhs:+.2f} Lakhs"
                )
            with k4:
                render_kpi_card(
                    label="80% Expected Range",
                    value=f"₹{res.uncertainty_lower_lakhs:.1f}L - ₹{res.uncertainty_upper_lakhs:.1f}L",
                    caption="Empirical tolerance band"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Fetch locality benchmark median
            df_props = DataLoader.load_properties(cleaned=True)
            loc_subset = df_props[df_props["locality"] == f_loc]
            loc_median = float(loc_subset["price_lakhs"].median()) if len(loc_subset) > 0 else None
            loc_avg_rate = float(loc_subset["price_per_sqft"].mean()) if len(loc_subset) > 0 else 8500.0
            loc_median_rate = float(loc_subset["price_per_sqft"].median()) if len(loc_subset) > 0 else 8200.0

            # Chart 1: Visual Comparison Bullet Gauge
            st.markdown("#### 🎯 Asking Price vs. Model Valuation & 80% Confidence Interval")
            fig_bullet = build_valuation_gauge_bullet(
                estimated_lakhs=res.estimated_price_lakhs,
                lower_lakhs=res.uncertainty_lower_lakhs,
                upper_lakhs=res.uncertainty_upper_lakhs,
                asking_lakhs=res.asking_price_lakhs,
                locality_median_lakhs=loc_median
            )
            st.plotly_chart(fig_bullet, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

            # Chart 2: Rate per Sq.Ft. Micro-Market Benchmark
            st.markdown("#### 📊 Rate per Square Foot: Micro-Market Benchmark")
            asking_rate = (res.asking_price_lakhs * 100000.0) / f_carpet
            est_rate = (res.estimated_price_lakhs * 100000.0) / f_carpet

            benchmark_df = pd.DataFrame([
                {"Category": "Seller Asking Rate", "Rate (₹/sq.ft.)": round(asking_rate, 1), "Color": PROPSIGHT_COLORS["rose_accent"] if res.classification == "Potentially Overpriced" else PROPSIGHT_COLORS["emerald_accent"]},
                {"Category": "Model Estimated Rate", "Rate (₹/sq.ft.)": round(est_rate, 1), "Color": PROPSIGHT_COLORS["primary_teal"]},
                {"Category": f"{f_loc} Median Rate", "Rate (₹/sq.ft.)": round(loc_median_rate, 1), "Color": PROPSIGHT_COLORS["blue_accent"]},
                {"Category": f"{f_loc} Average Rate", "Rate (₹/sq.ft.)": round(loc_avg_rate, 1), "Color": PROPSIGHT_COLORS["slate_dark"]}
            ])

            fig_rate_bench = px.bar(
                benchmark_df,
                x="Rate (₹/sq.ft.)",
                y="Category",
                orientation="h",
                color="Category",
                color_discrete_map={row["Category"]: row["Color"] for _, row in benchmark_df.iterrows()}
            )
            fig_rate_bench.update_traces(
                texttemplate="₹%{x:,.0f}/sq.ft.",
                textposition="outside",
                textfont=dict(size=11, color=PROPSIGHT_COLORS["slate_dark"]),
                cliponaxis=False
            )
            apply_chart_theme(
                fig_rate_bench,
                title=f"Rate/Sq.Ft. Positioning vs. {f_loc} Residential Market",
                xaxis_title="Effective Rate (₹/sq.ft.)",
                yaxis_title="",
                height=250,
                show_legend=False
            )
            st.plotly_chart(fig_rate_bench, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

            # Methodological Disclosure & Disclaimer Box
            st.markdown(f"""
            <div class="callout-box">
                <strong>Methodology & Threshold Disclosure:</strong><br>
                • <strong>Under/Over-Pricing Threshold:</strong> Defined by empirical 80% test residual intervals [₹{res.uncertainty_lower_lakhs:.1f}L to ₹{res.uncertainty_upper_lakhs:.1f}L].<br>
                • <strong>Target Definition:</strong> <code>{res.target_type}</code> (asking/listed rate rather than deed-registered transaction value).<br>
                • <strong>Responsible Decision Support:</strong> {res.disclaimer}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Enter an asking price and property details on the left, then click **'Analyze Asking Price'**.")

        st.markdown("""
        <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 1.5rem; color: #475569;">
            <h4 style="margin-top: 0; color: #0F172A;">How Fair-Price Analysis Works:</h4>
            <ul style="font-size: 0.88rem; line-height: 1.6; margin-bottom: 0;">
                <li><strong>Potentially Undervalued:</strong> The asking price sits below the lower bound of the model's 80% empirical prediction interval.</li>
                <li><strong>Within Expected Range:</strong> The asking price is comfortably bracketed by the model's confidence band.</li>
                <li><strong>Potentially Overpriced:</strong> The asking price sits above the upper bound of the model's 80% empirical prediction interval.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
