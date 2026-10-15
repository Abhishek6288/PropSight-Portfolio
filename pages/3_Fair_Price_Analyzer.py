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

            # Visual Comparison Gauge / Range Chart
            st.markdown("#### 📐 Asking Price vs Model Uncertainty Band")
            fig_range = go.Figure()

            # Uncertainty range horizontal bar
            fig_range.add_trace(go.Bar(
                name="Expected Range (80% CI)",
                y=["Property Value"],
                x=[res.uncertainty_upper_lakhs - res.uncertainty_lower_lakhs],
                base=[res.uncertainty_lower_lakhs],
                orientation="h",
                marker=dict(color="#BAE6FD", line=dict(color="#0284C7", width=1.5)),
                text=f"Expected Band: ₹{res.uncertainty_lower_lakhs:.1f}L – ₹{res.uncertainty_upper_lakhs:.1f}L",
                textposition="inside",
                hoverinfo="text"
            ))

            # Model Point Estimate marker
            fig_range.add_trace(go.Scatter(
                x=[res.estimated_price_lakhs],
                y=["Property Value"],
                mode="markers",
                name="Model Point Estimate",
                marker=dict(size=14, color="#0F172A", symbol="diamond"),
                text=f"Model Est: ₹{res.estimated_price_lakhs:.2f}L",
                hoverinfo="text"
            ))

            # Asking Price marker
            ask_color = "#10B981" if res.classification == "Potentially Undervalued" else ("#EF4444" if res.classification == "Potentially Overpriced" else "#0284C7")
            fig_range.add_trace(go.Scatter(
                x=[res.asking_price_lakhs],
                y=["Property Value"],
                mode="markers",
                name="Seller Asking Price",
                marker=dict(size=16, color=ask_color, symbol="circle"),
                text=f"Asking Price: ₹{res.asking_price_lakhs:.2f}L",
                hoverinfo="text"
            ))

            fig_range.update_layout(
                xaxis_title="Property Value (₹ Lakhs)",
                plot_bgcolor="#FFFFFF",
                height=220,
                margin=dict(l=20, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
            )
            st.plotly_chart(fig_range, use_container_width=True)

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
