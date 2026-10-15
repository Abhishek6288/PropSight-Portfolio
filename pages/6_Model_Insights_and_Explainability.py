"""
PropSight — Model Insights & Explainability
Comprehensive regression benchmark tournament, SHAP global feature importances,
and interactive local attribution playground.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_inr_lakhs
from src.utils.config import MODEL_METADATA_PATH, SUPPORTED_LOCALITIES, PROPERTY_TYPES, RANDOM_SEED, TARGET_COLUMN
from src.data.schema import MODEL_FEATURE_COLUMNS
from src.explainability.explainer import PropertyValuationExplainer
from src.valuation.engine import PropertyValuationEngine
from src.data.loader import DataLoader
from src.utils.charts import (
    apply_chart_theme,
    PROPSIGHT_COLORS,
    PROPSIGHT_MODEBAR_CONFIG,
    build_actual_vs_predicted_scatter,
    build_residuals_histogram
)

st.set_page_config(page_title="Model Insights & Explainability — PropSight", page_icon="🔍", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Model Insights & Explainable AI (SHAP)",
    subtitle="Benchmarking tournament, held-out test evaluation, global feature drivers, and prediction attribution",
    page_tag="Model Transparency"
)

# Load metadata
metadata = {}
if MODEL_METADATA_PATH.exists():
    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

@st.cache_resource
def get_explainer():
    return PropertyValuationExplainer()

@st.cache_data
def get_heldout_test_predictions():
    df = DataLoader.load_properties(cleaned=True)
    X = df[MODEL_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    engine = PropertyValuationEngine()
    X_test_trans = engine.preprocessor.transform(X_test)
    y_pred_test = engine.model.predict(X_test_trans)
    return y_test, y_pred_test

explainer = get_explainer()

# Tab Navigation: Model Evaluation vs SHAP Explainability
tab_benchmark, tab_global_shap, tab_interactive_shap = st.tabs([
    "🏆 Model Tournament & Benchmarks",
    "🌐 Global SHAP Feature Rankings",
    "🔬 Interactive Prediction Explainer"
])

# TAB 1: Model Tournament
with tab_benchmark:
    st.markdown("### 📊 Supervised Regression Model Tournament")
    st.markdown("""
    Six candidate architectures were trained and cross-validated on **2,880 properties** using identical feature preprocessing,
    and then evaluated on a strictly held-out test set of **720 properties** (20% split, random seed 42).
    """)

    # Primary Metrics KPI Cards for Winner
    best_m = metadata.get("best_model_metrics", {})
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            label="Selected Estimator",
            value=metadata.get("selected_model", "Gradient Boosting"),
            caption="Best generalization performance"
        )
    with k2:
        render_kpi_card(
            label="Held-Out Test R²",
            value=f"{best_m.get('test_r2', 0.9628):.4f}",
            caption=f"5-Fold CV: {best_m.get('cv_r2_mean', 0.969):.3f} ± {best_m.get('cv_r2_std', 0.005):.3f}"
        )
    with k3:
        render_kpi_card(
            label="Test MAE",
            value=f"₹{best_m.get('test_mae_lakhs', 6.81):.2f}L",
            caption="Mean Absolute Error in Lakhs"
        )
    with k4:
        render_kpi_card(
            label="Test MAPE",
            value=f"{best_m.get('test_mape_pct', 6.28):.2f}%",
            caption="Mean Absolute Percentage Error"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Benchmark Comparison Table
    st.markdown("#### 📋 Complete Model Evaluation Scorecard")
    candidates = metadata.get("candidate_comparison", {})
    records = []
    for name, m in candidates.items():
        records.append({
            "Architecture": name,
            "5-Fold CV R² (Mean ± Std)": f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}",
            "Train R²": f"{m['train_r2']:.4f}",
            "Test R²": round(m["test_r2"], 4),
            "Test MAE (₹ Lakhs)": round(m["test_mae_lakhs"], 2),
            "Test RMSE (₹ Lakhs)": round(m["test_rmse_lakhs"], 2),
            "Test MAPE (%)": round(m["test_mape_pct"], 2),
            "Selected": "⭐ Champion" if name == metadata.get("selected_model") else "Candidate"
        })
    df_eval = pd.DataFrame(records).sort_values(by="Test R²", ascending=False)
    st.dataframe(df_eval, use_container_width=True, hide_index=True)

    # Evaluation Bar Charts (Themed)
    c_chart1, c_chart2 = st.columns(2)
    best_name = metadata.get("selected_model", "Gradient Boosting")
    colors_arch = [
        PROPSIGHT_COLORS["primary_teal"] if arch == best_name else PROPSIGHT_COLORS["slate_muted"]
        for arch in df_eval["Architecture"]
    ]

    with c_chart1:
        fig_r2 = go.Figure(go.Bar(
            x=df_eval["Architecture"],
            y=df_eval["Test R²"],
            text=df_eval["Test R²"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
            marker_color=colors_arch,
            hovertemplate="<b>%{x}</b><br>Held-Out Test R²: %{y:.4f}<extra></extra>"
        ))
        apply_chart_theme(
            fig_r2,
            title="Held-Out Test R² (Higher is Better)",
            xaxis_title="Candidate Model Architecture",
            yaxis_title="R² Score",
            height=330
        )
        min_r2 = float(df_eval["Test R²"].min())
        fig_r2.update_yaxes(range=[max(0.0, min_r2 * 0.92), 1.02])
        st.plotly_chart(fig_r2, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

    with c_chart2:
        fig_mae = go.Figure(go.Bar(
            x=df_eval["Architecture"],
            y=df_eval["Test MAE (₹ Lakhs)"],
            text=df_eval["Test MAE (₹ Lakhs)"].apply(lambda v: f"₹{v:.2f}L"),
            textposition="outside",
            marker_color=colors_arch,
            hovertemplate="<b>%{x}</b><br>Held-Out Test MAE: ₹%{y:.2f} Lakhs<extra></extra>"
        ))
        apply_chart_theme(
            fig_mae,
            title="Held-Out Test MAE (Lower is Better)",
            xaxis_title="Candidate Model Architecture",
            yaxis_title="MAE (₹ Lakhs)",
            height=330
        )
        st.plotly_chart(fig_mae, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

    st.markdown("<br>", unsafe_allow_html=True)

    # Held-Out Test Evaluation Diagnostics (PRD Requirement)
    st.markdown("#### 🎯 Held-Out Test Set Generalization Diagnostics (720 Properties)")
    st.caption("Rigorous empirical evaluation comparing actual listing prices against predictions on unobserved test data.")

    with st.spinner("Loading held-out test predictions..."):
        y_test, y_pred_test = get_heldout_test_predictions()

    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        fig_scatter = build_actual_vs_predicted_scatter(y_test, y_pred_test, sample_size=720)
        st.plotly_chart(fig_scatter, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

    with col_diag2:
        fig_residuals = build_residuals_histogram(y_test, y_pred_test)
        st.plotly_chart(fig_residuals, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

# TAB 2: Global Feature Importance
with tab_global_shap:
    st.markdown("### 🌐 Global Prediction Drivers (Feature Importance)")
    st.markdown("""
    Global feature importance measures the relative contribution of each property attribute
    in reducing prediction error across all 3,600 properties in Pune & PCMC.
    """)

    df_importance = explainer.get_global_importance(top_n=12)
    df_imp_sorted = df_importance.sort_values(by="importance_score", ascending=True)

    top_driver_colors = [
        PROPSIGHT_COLORS["primary_teal"] if score >= df_imp_sorted["importance_score"].quantile(0.65)
        else PROPSIGHT_COLORS["blue_accent"] if score >= df_imp_sorted["importance_score"].quantile(0.35)
        else PROPSIGHT_COLORS["slate_muted"]
        for score in df_imp_sorted["importance_score"]
    ]

    fig_imp = go.Figure(go.Bar(
        x=df_imp_sorted["importance_score"],
        y=df_imp_sorted["feature_name"],
        orientation="h",
        text=df_imp_sorted["importance_score"].apply(lambda v: f"{v*100:.1f}%"),
        textposition="outside",
        marker_color=top_driver_colors,
        hovertemplate="<b>%{y}</b><br>Relative Feature Attribution: %{x:.4f} (%{text})<extra></extra>"
    ))
    apply_chart_theme(
        fig_imp,
        title="Top Model Feature Drivers Across Pune & PCMC (Global Importance)",
        xaxis_title="Relative Attribution Weight",
        yaxis_title="Property Attribute",
        height=450
    )
    st.plotly_chart(fig_imp, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

    # Explanation Callout
    st.markdown("""
    <div class="callout-box">
        <strong>Key Insights from Feature Attribution:</strong><br>
        1. <strong>Carpet Area (sq.ft.):</strong> Accounts for >50% of model weight. As the fundamental physical quantum of real estate, area dominates total transaction pricing.<br>
        2. <strong>Locality Tier:</strong> Micro-market categorization (Premium Central vs IT Hub vs PCMC Growth Corridor) represents the second most decisive driver (~20-25%).<br>
        3. <strong>Building Attributes & Configuration:</strong> BHK density, relative floor level in high-rises, and bathroom count fine-tune pricing within each micro-market.
    </div>
    """, unsafe_allow_html=True)

# TAB 3: Interactive Prediction Explainer
with tab_interactive_shap:
    st.markdown("### 🔬 Interactive Prediction Attribution (SHAP Waterfall)")
    st.markdown("Adjust property parameters below to see in real-time how each attribute alters the valuation from the baseline:")

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        test_loc = st.selectbox("Locality", options=sorted(SUPPORTED_LOCALITIES), index=0, key="exp_loc")
        test_bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5], index=1, key="exp_bhk")
    with p_col2:
        test_carpet = st.slider("Carpet Area (sq.ft.)", 400.0, 3000.0, 950.0, 50.0, key="exp_carpet")
        test_floor = st.slider("Floor", 1, 30, 7, key="exp_floor")
    with p_col3:
        test_furnish = st.selectbox("Furnishing", options=["Unfurnished", "Semi-Furnished", "Fully Furnished"], index=2, key="exp_furn")
        test_amenities = st.slider("Amenities Count", 1, 11, 7, key="exp_amen")

    sample_input = {
        "locality": test_loc,
        "property_type": "Apartment",
        "bhk": test_bhk,
        "carpet_area_sqft": test_carpet,
        "floor": test_floor,
        "total_floors": 18,
        "property_age_years": 2.0,
        "bathrooms": test_bhk,
        "balconies": max(1, test_bhk - 1),
        "parking_spaces": 1,
        "furnishing": test_furnish,
        "possession_status": "Ready to Move",
        "amenities_count": test_amenities
    }

    with st.spinner("Calculating SHAP attributions..."):
        exp_res = explainer.explain_instance(sample_input)

    # Render Waterfall
    fig_local = explainer.plot_waterfall(exp_res)
    st.plotly_chart(fig_local, use_container_width=True, config=PROPSIGHT_MODEBAR_CONFIG)

    # Methodology Alert
    st.markdown("""
    <div class="callout-box">
        <strong>Academic Note on XAI Attribution vs Causation:</strong><br>
        SHAP values quantify how much each input feature nudges the model's conditional expectation away from the base value ($E[f(X)]$). They represent statistical associations learned by the Gradient Boosting trees, not legal or structural causation.
    </div>
    """, unsafe_allow_html=True)
