"""
PropSight — About, Methodology & Limitations
Complete academic project documentation, mathematical formulations,
system architecture, and responsible use disclaimers.
"""

import streamlit as st
import json

from src.utils.styles import apply_custom_css, render_header
from src.utils.config import MODEL_METADATA_PATH

st.set_page_config(page_title="About & Methodology — PropSight", page_icon="📖", layout="wide")
apply_custom_css()

# Header
render_header(
    title="About, Methodology & Academic Governance",
    subtitle="Theoretical foundations, mathematical formulations, technology architecture, and responsible use principles",
    page_tag="Methodology & Governance"
)

# Load metadata
metadata = {}
if MODEL_METADATA_PATH.exists():
    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

# Navigation tabs
t_method, t_limits, t_stack, t_about = st.tabs([
    "📐 Methodology & Formulations",
    "⚠️ Limitations & Governance",
    "💻 Technology Architecture",
    "🎓 Academic Capstone Context"
])

# TAB 1: Methodology & Formulations
with t_method:
    st.markdown("### 1. Valuation Modeling (Supervised Regression)")
    st.markdown(r"""
    The property valuation task is framed as a supervised regression problem where we learn a function:
    $$\hat{y} = f(X)$$
    where:
    - $\hat{y}$ is the predicted property listing price in **INR (₹) Lakhs**.
    - $X \in \mathbb{R}^{d}$ is the feature vector composed of physical dimensions (carpet area, built-up area, floor position, age, BHK, bathrooms, amenities) and localized categorical encodings (micro-market, region, furnishing, possession).

    #### Feature Transformations & Leakage Prevention
    - **One-Hot Encoding with Unknown Handling:** Categorical variables are transformed using `OneHotEncoder(handle_unknown='ignore')` to ensure unforeseen categories at inference time do not break the pipeline.
    - **Engineered Non-Leaking Ratios:**
      - Vertical Height Index: $\text{Floor Ratio} = \frac{\text{Floor}}{\text{Total Floors}}$
      - Spatial Density: $\text{Area per BHK} = \frac{\text{Carpet Area}}{\text{BHK}}$
      - Bathroom Density: $\text{Bath per BHK} = \frac{\text{Bathrooms}}{\text{BHK}}$
      - Construction Loading: $\text{Built to Carpet} = \frac{\text{Built-up Area}}{\text{Carpet Area}}$
    - **Strict Target Leakage Control:** Rate per square foot ($\frac{\text{Price}}{\text{Area}}$) is strictly excluded from input features because it is mathematically derived from the target variable.
    """)

    st.markdown("---")
    st.markdown("### 2. Uncertainty Quantification (Prediction Intervals)")
    st.markdown(r"""
    Rather than presenting a deterministic point forecast, PropSight constructs an **empirical 80% prediction interval** derived from the residual distribution on the held-out test set ($N = 720$):
    $$e_i = y_i - \hat{y}_i \quad \forall i \in \text{Test Set}$$
    $$\text{Lower Bound} = \hat{y} + \text{Quantile}_{0.10}(e)$$
    $$\text{Upper Bound} = \hat{y} + \text{Quantile}_{0.90}(e)$$
    For the active Gradient Boosting model, the empirical bounds are:
    $$\text{Quantile}_{0.10}(e) = -6.29 \text{ Lakhs}, \quad \text{Quantile}_{0.90}(e) = +6.22 \text{ Lakhs}$$
    """)

    st.markdown("---")
    st.markdown("### 3. Fair-Price Analysis Framework")
    st.markdown(r"""
    The asking price $P_{\text{ask}}$ is categorized relative to the empirical prediction interval:
    $$\text{Classification} = \begin{cases}
    \textbf{Potentially Undervalued} & \text{if } P_{\text{ask}} < \text{Lower Bound} \\
    \textbf{Within Expected Range} & \text{if } \text{Lower Bound} \le P_{\text{ask}} \le \text{Upper Bound} \\
    \textbf{Potentially Overpriced} & \text{if } P_{\text{ask}} > \text{Upper Bound}
    \end{cases}$$
    Relative percentage deviation is computed with a safeguarded non-zero denominator:
    $$\Delta\% = \frac{P_{\text{ask}} - \hat{y}}{\hat{y}} \times 100\%$$
    """)

    st.markdown("---")
    st.markdown("### 4. Time-Series Forecasting & Back-Testing")
    st.markdown(r"""
    Locality-level forecasting models quarterly average rate per square foot ($r_t$) using an Auto-Regressive Integrated Moving Average $\text{ARIMA}(p, d, q)$ specification:
    $$\phi(B)(1 - B)^d r_t = \theta(B)\epsilon_t$$
    - **Chronological Walk-Forward Back-Testing:** Evaluated on the held-out 4 quarters of 2024 ($N_{\text{test}} = 4$) without future data contamination.
    - **Expanding Prediction Intervals:** Confidence intervals widen asymptotically with the forecast step $h$:
      $$\sigma_h = \sigma_\epsilon \sqrt{1 + \sum_{j=1}^{h-1} \psi_j^2}$$
      guaranteeing that epistemic uncertainty increases for distant quarters.
    """)

# TAB 2: Limitations & Governance
with t_limits:
    st.markdown("### ⚠️ Key Limitations & Boundary Disclaimers")
    st.markdown("""
    To preserve data-science integrity, the following limitations are explicitly documented:

    1. **Demonstration Dataset Notice:**
       - In accordance with PRD §7, in the absence of free, permitted, real-time live MLS feeds for Pune/PCMC, PropSight employs a **demonstration dataset** calibrated to published quarterly market aggregates (Anarock and Knight Frank Research).
       - While statistical relationships, price per square foot tiers, and depreciation curves mirror observed market realities, model estimates must **never be represented as actual legal market transactions**.

    2. **Asking vs. Transaction Price (`target_type`):**
       - The target variable is defined as `listing_price`. Listing prices generally carry a **3% to 8% negotiation premium** above final deed-registered transaction prices in the Pune/PCMC sub-registrar offices.

    3. **Unobserved Micro-Attributes:**
       - The model cannot observe unrecorded property characteristics such as:
         - Exact scenic view (e.g. hill-facing vs highway-facing).
         - Quality of interior architectural woodwork and fittings.
         - Legal title clarity, encumbrance certificate status, or builder RERA dispute history.
         - Vastu compliance, which heavily influences demand in specific Pune micro-markets.

    4. **Forecasting Horizon Restrictions:**
       - Projections beyond 2028 are strictly classified as future/speculative research scope. Time-series extrapolation over 5-10 years without macroeconomic structural equations compounds forecast errors uncontrollably.
    """)

# TAB 3: Technology Architecture
with t_stack:
    st.markdown("### 💻 Technology Stack & Zero-Cost Compliance")
    st.markdown("""
    PropSight was engineered to run locally on standard CPU hardware with **₹0 budget**, adhering to open-source first principles:
    """)

    tech_table = [
        {"Component": "Frontend & UI", "Technology": "Streamlit (Pure Python)", "Licence": "Apache 2.0", "Cost": "₹0"},
        {"Component": "Data Wrangling", "Technology": "Pandas, NumPy", "Licence": "BSD 3-Clause", "Cost": "₹0"},
        {"Component": "Machine Learning", "Technology": "Scikit-Learn, XGBoost", "Licence": "BSD / Apache 2.0", "Cost": "₹0"},
        {"Component": "Forecasting", "Technology": "Statsmodels (ARIMA)", "Licence": "BSD 3-Clause", "Cost": "₹0"},
        {"Component": "Explainable AI (XAI)", "Technology": "SHAP (TreeExplainer)", "Licence": "MIT", "Cost": "₹0"},
        {"Component": "Interactive Visualizations", "Technology": "Plotly Graph Objects & Express", "Licence": "MIT", "Cost": "₹0"},
        {"Component": "Database & Audit Logging", "Technology": "SQLite3 (Serverless)", "Licence": "Public Domain", "Cost": "₹0"},
        {"Component": "Automated Testing", "Technology": "Pytest", "Licence": "MIT", "Cost": "₹0"}
    ]
    st.dataframe(pd.DataFrame(tech_table), use_container_width=True, hide_index=True)

# TAB 4: Academic Capstone Context
with t_about:
    st.markdown("### 🎓 Academic Project Context")
    st.markdown("""
    **Project Title:** PropSight — Localized Real-Estate Intelligence & Decision-Support System  
    **Target Geography:** Pune Municipal Corporation (PMC) & Pimpri-Chinchwad Municipal Corporation (PCMC), Maharashtra, India  
    **Academic Level:** BSc Data Science Capstone Project  
    **Primary Source of Truth:** PropSight Product Requirements Document (PRD)

    #### Core Principles
    - **Academic Defensibility:** All preprocessing, cross-validation, and back-testing procedures are strictly reproducible and leak-free.
    - **Responsible AI:** Transparent prediction intervals communicate uncertainty rather than implying artificial infallibility.
    - **Model Explainability:** SHAP feature attribution demystifies black-box tree ensembles for property buyers and analysts.
    """)
