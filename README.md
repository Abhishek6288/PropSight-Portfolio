# PropSight: Localized Real-Estate Intelligence & Decision Support

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.47-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E.svg)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/SHAP-0.53-green.svg)](https://shap.readthedocs.io/)
[![Database](https://img.shields.io/badge/SQLite-Idempotent-003B57.svg)](https://sqlite.org/)
[![Licence](https://img.shields.io/badge/Budget-₹0%20(Open%20Source)-success.svg)](#non-negotiable-constraints)

> **BSc Data Science Capstone Project**  
> **Geographic Scope:** Pune Municipal Corporation (PMC) & Pimpri-Chinchwad Municipal Corporation (PCMC), Maharashtra, India.

---

## 1. Executive Summary & Mission

**PropSight** is an academically defensible, locally runnable real-estate intelligence and decision-support application built for Pune and Pimpri-Chinchwad (PCMC). It bridges empirical data-science methodologies with real-world property decisions by delivering:

1. **Supervised ML Valuation:** Accurately estimates current property value with empirical 80% prediction intervals rather than deceptive single-point numbers.
2. **Explainable AI (SHAP):** Unpacks individual predictions via interactive waterfall attribution charts, revealing how each feature shifts the value from the market baseline.
3. **Uncertainty-Aware Fair-Price Analyzer:** Classifies seller asking prices into *Potentially Undervalued*, *Within Expected Range*, and *Potentially Overpriced* with transparent tolerance bounds.
4. **Locality Intelligence:** Analyzes 28 micro-markets with historical CAGR, price distributions, and side-by-side comparative benchmarking.
5. **Econometric Time-Series Forecasting:** Evaluates quarterly historical price trends (2018–2024), performs walk-forward back-testing (ARIMA vs. baselines), and projects future rates with expanding confidence bands and macroeconomic scenarios.
6. **Transparent Data Governance:** Complete schema auditing, outlier diagnostics, and SQLite transaction logging.

---

## 2. Non-Negotiable Constraints & Architecture

- **Frontend:** Streamlit only.
- **Language & Runtime:** Python 3.12 on local CPU architecture.
- **Cost & Dependencies:** ₹0 software, infrastructure, and API budget. No paid cloud services, GPU requirements, or external LLM API keys.
- **Persistence:** SQLite serverless database with parameterized queries and idempotent initialization.
- **Target Variable Definition:** `price_lakhs` (Total price in INR Lakhs; $1\text{ Lakh} = \text{₹}100,000$). The target is explicitly defined as `listing_price`.
- **Target Leakage Prevention:** Rate per square foot ($\frac{\text{Price}}{\text{Area}}$) is strictly excluded from all training and inference features.

---

## 3. Project Directory Structure

```text
puffpetals/
├── app.py                             # Main Streamlit application entry point & overview
├── conftest.py                        # Pytest path configuration
├── requirements.txt                   # Free & open-source Python dependencies
├── .gitignore                         # Version control exclusions
├── README.md                          # Comprehensive project documentation
├── pages/
│   ├── 1_Executive_Dashboard.py       # City-wide KPIs, price distributions, and regional trends
│   ├── 2_Property_Valuation.py        # Supervised ML inference with empirical intervals & SHAP
│   ├── 3_Fair_Price_Analyzer.py       # Asking price comparison & uncertainty classification
│   ├── 4_Locality_Intelligence.py     # 28 micro-market profiles & side-by-side benchmarking
│   ├── 5_Market_Forecasts_and_Scenarios.py # ARIMA forecasting & 3-year scenario fan charts
│   ├── 6_Model_Insights_and_Explainability.py # Model tournament scorecard & global SHAP rankings
│   ├── 7_Data_Explorer.py             # Schema inspector, missingness checks & data dictionary
│   └── 8_About_and_Methodology.py     # Academic formulations, principles & limitations
├── src/
│   ├── data/
│   │   ├── schema.py                  # Schema definitions & physical boundary constraints
│   │   ├── synthetic_generator.py     # Calibrated demonstration data generator (2018-2024)
│   │   ├── cleaner.py                 # Outlier detection (IQR), deduplication & audit reporting
│   │   ├── loader.py                  # Unified data loader & dataset diagnostic utilities
│   │   └── feature_engineering.py     # Leak-free domain transformers & ColumnTransformer
│   ├── models/
│   │   └── trainer.py                 # Multi-model evaluation tournament & pipeline serializer
│   ├── valuation/
│   │   ├── engine.py                  # Property valuation engine & prediction interval estimator
│   │   └── fair_price.py              # Fair-price comparison engine & classification rationale
│   ├── explainability/
│   │   └── explainer.py               # SHAP TreeExplainer & Plotly waterfall charts
│   ├── analytics/
│   │   └── locality.py                # Micro-market KPI aggregation & comparative analytics
│   ├── forecasting/
│   │   └── engine.py                  # Chronological back-testing, ARIMA modeling & scenarios
│   ├── database/
│   │   └── db.py                      # SQLite idempotent initialization & audit logging
│   └── utils/
│       ├── config.py                  # System paths, constants & random seeds
│       ├── formatting.py              # Indian currency (₹ L/Cr), sq.ft., and % formatters
│       └── styles.py                  # Executive UI styling and CSS component renderers
├── data/
│   ├── raw/                           # Raw demonstration property & historical trend CSVs
│   ├── processed/                     # Cleaned, validated CSVs
│   └── db/                            # SQLite database (propsight.db)
├── models/
│   └── artifacts/                     # Serialized pipeline, metadata.json & SHAP background
├── notebooks/
│   └── eda_and_experiments.py         # Standalone EDA and model evaluation runner
└── tests/
    └── test_propsight.py              # Automated Pytest suite (16 test cases)
```

---

## 4. Machine Learning Methodology & Benchmark Results

Property valuation is treated as a supervised regression task. Six architectures were trained using identical 5-fold cross-validation on training data ($N = 2,880$) and tested on a held-out test split ($N = 720$, 20%):

### Model Tournament Scorecard

| Architecture | 5-Fold CV $R^2$ (Mean ± Std) | Train $R^2$ | Held-Out Test $R^2$ | Test MAE (₹ Lakhs) | Test RMSE (₹ Lakhs) | Test MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9696 ± 0.0045** | **0.9946** | **0.9628** | **₹6.81L** | **₹12.15L** | **6.28%** | ⭐ **Selected Champion** |
| **XGBoost Regressor** | 0.9681 ± 0.0037 | 0.9938 | 0.9600 | ₹6.66L | ₹12.59L | 6.09% | Strong Contender |
| **Random Forest** | 0.9523 ± 0.0036 | 0.9917 | 0.9352 | ₹8.55L | ₹16.03L | 7.77% | Ensemble Baseline |
| **Ridge Regression** | 0.9345 ± 0.0055 | 0.9367 | 0.9201 | ₹11.02L | ₹17.80L | 11.74% | Regularized Linear |
| **Linear Regression** | 0.9342 ± 0.0050 | 0.9371 | 0.9199 | ₹11.09L | ₹17.82L | 11.94% | Linear Baseline |
| **Lasso Regression** | 0.9327 ± 0.0056 | 0.9346 | 0.9186 | ₹11.13L | ₹17.97L | 11.63% | Regularized Linear |

### Uncertainty Estimation
Rather than point forecasts, PropSight calculates an empirical 80% prediction interval based on test residuals ($e = y_{\text{test}} - \hat{y}_{\text{test}}$):
- **Lower Bound:** $\hat{y} + \text{Quantile}_{0.10}(e) = \hat{y} - 6.29\text{ Lakhs}$
- **Upper Bound:** $\hat{y} + \text{Quantile}_{0.90}(e) = \hat{y} + 6.22\text{ Lakhs}$

---

## 5. Time-Series Forecasting & Back-Testing

Locality-level forecasting models quarterly average price per sq.ft. from 2018-Q1 to 2024-Q4 (28 quarters):
- **Chronological Split:** Back-tested on 2024 held-out quarters without random splitting.
- **Model Comparison:** 4-Quarter Moving Average (MAPE: 7.14%), Simple Exponential Smoothing (MAPE: 3.57%), and $\text{ARIMA}(1, 1, 1)$ (MAPE: **0.42%**, MAE: **₹39.2/sq.ft.**).
- **Epistemic Uncertainty:** Prediction intervals widen monotonically with forecast horizon.
- **Scope Boundary:** Projections beyond 2028 are strictly classified as future/speculative research scope and are rejected with explanatory notes.

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.5 on Windows / Linux / macOS)
- Git

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <repo-url>
cd puffpetals
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 2: Install Zero-Cost Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
```bash
python -m pytest tests/ -v
```
*Expected: 16 passed tests covering data loading, cleaning, inference, fair price, forecasting, DB logging, and SHAP.*

### Step 4: Launch PropSight Streamlit Application
```bash
streamlit run app.py
```
*The web browser will open automatically at `http://localhost:8501`.*

---

## 7. Data Provenance & Limitations

### Data Provenance
In adherence to PRD §7, in the absence of free, permitted live MLS feeds for Pune/PCMC, PropSight operates on a **calibrated demonstration dataset** ($N = 3,600$ listings across 28 micro-markets; 784 historical quarterly index observations). Distributions are calibrated from published market reports by Anarock Research and Knight Frank India (2018–2024).

### Explicit Limitations & Disclaimers
1. **Decision Support Only:** PropSight is not a certified legal appraisal, registered valuer's report, or investment guarantee.
2. **Target Type:** Target is `listing_price`, which includes an estimated 3% to 8% seller negotiation premium over sub-registrar deed-registered transaction prices.
3. **Unobserved Micro-Attributes:** View quality, interior architectural finishes, legal encumbrances, and Vastu compliance are unobserved in the tabular schema.
4. **Synthetic Labeling:** All demonstration records are explicitly labeled with `is_synthetic = True`.

---

## 8. Automated Test Summary

```text
tests/test_propsight.py::test_data_loader_properties PASSED
tests/test_propsight.py::test_data_loader_historical PASSED
tests/test_propsight.py::test_data_diagnostics PASSED
tests/test_propsight.py::test_cleaner_physical_bounds PASSED
tests/test_propsight.py::test_cleaner_deduplication PASSED
tests/test_propsight.py::test_preprocessor_pipeline PASSED
tests/test_propsight.py::test_preprocessor_unknown_category PASSED
tests/test_propsight.py::test_valuation_inference PASSED
tests/test_propsight.py::test_valuation_invalid_inputs PASSED
tests/test_propsight.py::test_fair_price_classifications PASSED
tests/test_propsight.py::test_fair_price_zero_negative_asking PASSED
tests/test_propsight.py::test_forecast_eligibility PASSED
tests/test_propsight.py::test_forecast_generation PASSED
tests/test_propsight.py::test_forecast_backtesting PASSED
tests/test_propsight.py::test_database_logging PASSED
tests/test_propsight.py::test_shap_explanation PASSED

16 passed in 3.59s
```
