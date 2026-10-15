"""
PropSight — Property Valuation
Interactive valuation engine predicting approximate current market value
with empirical uncertainty intervals and instant SHAP explainability.
"""

import streamlit as st
import pandas as pd

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_inr_lakhs, format_rate_per_sqft, format_sqft
from src.utils.config import (
    SUPPORTED_LOCALITIES,
    PROPERTY_TYPES,
    FURNISHING_STATUSES,
    POSSESSION_STATUSES,
    AMENITIES_LIST
)
from src.valuation.engine import PropertyValuationEngine, ValuationResult
from src.explainability.explainer import PropertyValuationExplainer

st.set_page_config(page_title="Property Valuation — PropSight", page_icon="🎯", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Property Valuation Engine",
    subtitle="Supervised ML current-value estimation with empirical confidence intervals and SHAP attribution",
    page_tag="Valuation Module"
)

# Load engines with session caching
@st.cache_resource
def get_valuation_engine():
    return PropertyValuationEngine()

@st.cache_resource
def get_explainer():
    return PropertyValuationExplainer()

engine = get_valuation_engine()
explainer = get_explainer()

# Layout: Left column for form inputs, Right column for output
col_form, col_output = st.columns([1.1, 1.4])

with col_form:
    st.markdown("### 📋 Enter Property Specifications")
    
    with st.form("valuation_form"):
        # Locality & Type
        f_loc = st.selectbox("Locality", options=sorted(SUPPORTED_LOCALITIES), index=0)
        f_type = st.selectbox("Property Type", options=PROPERTY_TYPES, index=0)
        
        # BHK & Area
        bhk_col, carpet_col = st.columns(2)
        with bhk_col:
            f_bhk = st.selectbox("BHK Configuration", options=[1, 2, 3, 4, 5], index=1)
        with carpet_col:
            f_carpet = st.number_input("Carpet Area (sq.ft.)", min_value=300.0, max_value=8000.0, value=850.0, step=25.0)

        # Floor details
        fl_col, tot_col = st.columns(2)
        with fl_col:
            f_floor = st.number_input("Floor Level", min_value=0, max_value=50, value=4, step=1)
        with tot_col:
            f_total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=14, step=1)

        # Bathrooms & Age
        bath_col, age_col = st.columns(2)
        with bath_col:
            f_bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5, 6], index=min(f_bhk - 1, 5))
        with age_col:
            f_age = st.number_input("Property Age (Years)", min_value=0.0, max_value=40.0, value=2.0, step=0.5)

        # Furnishing & Possession
        furn_col, poss_col = st.columns(2)
        with furn_col:
            f_furnishing = st.selectbox("Furnishing Status", options=FURNISHING_STATUSES, index=1)
        with poss_col:
            f_possession = st.selectbox("Possession Status", options=POSSESSION_STATUSES, index=0)

        # Parking & Balconies
        park_col, balc_col = st.columns(2)
        with park_col:
            f_parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3], index=1)
        with balc_col:
            f_balconies = st.selectbox("Balconies", options=[0, 1, 2, 3, 4], index=1)

        # Amenities
        f_amenities = st.multiselect(
            "Available Amenities",
            options=AMENITIES_LIST,
            default=["24x7 Security", "Power Backup", "Covered Parking", "Gymnasium", "Clubhouse"]
        )

        submit_btn = st.form_submit_button("⚡ Estimate Property Value", use_container_width=True)

with col_output:
    st.markdown("### 📊 Valuation Assessment")

    if submit_btn:
        # Input validation
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
                "balconies": f_balconies,
                "parking_spaces": f_parking,
                "furnishing": f_furnishing,
                "possession_status": f_possession,
                "amenities_count": len(f_amenities)
            }

            with st.spinner("Executing valuation pipeline and computing uncertainty intervals..."):
                val_result: ValuationResult = engine.estimate_value(property_payload, log_to_db=True)
                exp_result = explainer.explain_instance(property_payload)

            # Store in session state for cross-page reference
            st.session_state["last_valuation_payload"] = property_payload
            st.session_state["last_valuation_result"] = val_result

            # Display Primary Result KPIs
            k1, k2, k3 = st.columns(3)
            with k1:
                render_kpi_card(
                    label="Estimated Value",
                    value=format_inr_lakhs(val_result.estimated_price_lakhs),
                    caption=f"Logged Transaction #{val_result.prediction_id}"
                )
            with k2:
                render_kpi_card(
                    label="Rate / Sq.Ft.",
                    value=format_rate_per_sqft(val_result.estimated_price_per_sqft),
                    caption=f"On {format_sqft(f_carpet)} carpet"
                )
            with k3:
                lower_str = format_inr_lakhs(val_result.uncertainty_lower_lakhs)
                upper_str = format_inr_lakhs(val_result.uncertainty_upper_lakhs)
                render_kpi_card(
                    label="80% Expected Range",
                    value=f"{lower_str} - {upper_str}",
                    caption="Empirical residual interval"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # SHAP Waterfall Attribution Chart
            st.markdown("#### 🔬 Key Prediction Drivers (SHAP Waterfall)")
            fig_waterfall = explainer.plot_waterfall(exp_result)
            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Top Factors Breakdown
            st.markdown("##### 📌 Top Feature Attributions")
            contrib_df = pd.DataFrame(exp_result["top_contributions"])[[
                "feature_label", "attribution_lakhs", "impact"
            ]]
            contrib_df.columns = ["Feature", "Attribution (₹ Lakhs)", "Effect"]
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)

            # Model & Responsible Use Notice
            st.markdown("""
            <div class="callout-box">
                <strong>Model Specifications & Responsible Use:</strong><br>
                • Estimator: <code>Gradient Boosting Regressor</code> (Trained on Pune/PCMC residential dataset)<br>
                • Target Definition: <code>listing_price</code> (Asking/listing price in INR Lakhs)<br>
                • <em>Notice:</em> This estimate is generated for academic decision-support purposes and does not replace a certified professional property appraisal.
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Configure your property parameters on the left and click **'Estimate Property Value'** to view model assessment.")

        st.markdown("""
        <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 1.5rem; text-align: center; color: #64748B;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏢</div>
            <strong>How PropSight Computes Valuation:</strong><br>
            1. Validates physical plausibility and boundaries.<br>
            2. Applies training-fitted StandardScaler and One-Hot Encoders.<br>
            3. Derives relative height, density, and loading ratios.<br>
            4. Generates point estimate via gradient boosted regression trees.<br>
            5. Computes empirical 80% confidence interval based on held-out test residuals.<br>
            6. Produces SHAP feature attributions explaining the price difference from market baseline.
        </div>
        """, unsafe_allow_html=True)
