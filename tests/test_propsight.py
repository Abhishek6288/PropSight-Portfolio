"""
Automated Test Suite for PropSight.
Covers data loading, schema validation, cleaning, feature engineering,
model inference, fair-price logic, forecasting eligibility, database logging,
and SHAP explainability.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import (
    SUPPORTED_LOCALITIES,
    PROPERTY_TYPES,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    DB_PATH
)
from src.data.schema import PropertyConstraints, MODEL_FEATURE_COLUMNS
from src.data.loader import DataLoader
from src.data.cleaner import PropertyDataCleaner
from src.data.feature_engineering import build_preprocessor
from src.valuation.engine import PropertyValuationEngine, ValuationResult
from src.valuation.fair_price import FairPriceAnalyzer, FairPriceResult
from src.forecasting.engine import LocalityForecastEngine
from src.explainability.explainer import PropertyValuationExplainer
from src.database.db import init_db, log_prediction, get_recent_predictions, get_connection

@pytest.fixture(scope="session")
def setup_environment():
    """Ensure database and artifacts exist before test run."""
    init_db()

# --- 1. Data Loading & Schema Tests ---
def test_data_loader_properties():
    df = DataLoader.load_properties(cleaned=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    for col in ["locality", "bhk", "carpet_area_sqft", "price_lakhs"]:
        assert col in df.columns

def test_data_loader_historical():
    df_hist = DataLoader.load_historical_trends()
    assert isinstance(df_hist, pd.DataFrame)
    assert len(df_hist) > 100
    assert "quarter" in df_hist.columns
    assert "avg_price_per_sqft" in df_hist.columns

def test_data_diagnostics():
    diag = DataLoader.get_dataset_diagnostics()
    assert diag["total_clean_listings"] > 0
    assert diag["locality_count"] >= 20
    assert diag["target_column"] == "price_lakhs"
    assert diag["is_synthetic"] is True

# --- 2. Data Cleaning & Validation Tests ---
def test_cleaner_physical_bounds():
    cleaner = PropertyDataCleaner()
    # Create test dataframe with an invalid row
    bad_df = pd.DataFrame([{
        "locality": "Baner", "property_type": "Apartment", "bhk": 2,
        "carpet_area_sqft": 50.0, # Below MIN_CARPET_AREA (250)
        "built_up_area_sqft": 60.0, "floor": 2, "total_floors": 10,
        "property_age_years": 1.0, "bathrooms": 2, "balconies": 1,
        "parking_spaces": 1, "furnishing": "Semi-Furnished",
        "possession_status": "Ready to Move", "amenities_count": 5,
        "price_lakhs": 50.0
    }])
    cleaned_df, report = cleaner.clean(bad_df)
    assert len(cleaned_df) == 0 # Should be rejected
    assert report["steps"]["physical_bounds"]["rows_removed"] == 1

def test_cleaner_deduplication():
    cleaner = PropertyDataCleaner()
    row = {
        "locality": "Wakad", "property_type": "Apartment", "bhk": 2,
        "carpet_area_sqft": 800.0, "built_up_area_sqft": 1000.0,
        "floor": 3, "total_floors": 12, "property_age_years": 2.0,
        "bathrooms": 2, "balconies": 1, "parking_spaces": 1,
        "furnishing": "Semi-Furnished", "possession_status": "Ready to Move",
        "amenities_count": 5, "price_lakhs": 75.0
    }
    dup_df = pd.DataFrame([row, row]) # Two identical rows
    cleaned_df, report = cleaner.clean(dup_df)
    assert len(cleaned_df) == 1
    assert report["steps"]["deduplication"]["rows_removed"] == 1

# --- 3. Feature Preprocessing Tests ---
def test_preprocessor_pipeline():
    pipe, num_f, cat_f = build_preprocessor()
    sample = pd.DataFrame([{
        "locality": "Kothrud", "region": "Pune", "property_type": "Apartment",
        "bhk": 3, "carpet_area_sqft": 1400.0, "built_up_area_sqft": 1750.0,
        "floor": 4, "total_floors": 12, "property_age_years": 3.0,
        "bathrooms": 3, "balconies": 2, "parking_spaces": 2,
        "furnishing": "Fully Furnished", "possession_status": "Ready to Move",
        "amenities_count": 8
    }])
    # Fit and transform
    X_trans = pipe.fit_transform(sample)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 1
    assert X_trans.shape[1] > 10

def test_preprocessor_unknown_category():
    pipe, _, _ = build_preprocessor()
    sample_train = pd.DataFrame([{
        "locality": "Baner", "region": "Pune", "property_type": "Apartment",
        "bhk": 2, "carpet_area_sqft": 850.0, "built_up_area_sqft": 1100.0,
        "floor": 2, "total_floors": 10, "property_age_years": 1.0,
        "bathrooms": 2, "balconies": 1, "parking_spaces": 1,
        "furnishing": "Semi-Furnished", "possession_status": "Ready to Move",
        "amenities_count": 5
    }])
    pipe.fit(sample_train)

    # Test with unseen locality and unseen property type
    sample_unknown = pd.DataFrame([{
        "locality": "Unseen_Location_XYZ", "region": "UnknownRegion", "property_type": "Castle",
        "bhk": 2, "carpet_area_sqft": 850.0, "built_up_area_sqft": 1100.0,
        "floor": 2, "total_floors": 10, "property_age_years": 1.0,
        "bathrooms": 2, "balconies": 1, "parking_spaces": 1,
        "furnishing": "Unfurnished", "possession_status": "Ready to Move",
        "amenities_count": 5
    }])
    # Should not raise an error due to handle_unknown='ignore'
    X_unknown = pipe.transform(sample_unknown)
    assert X_unknown.shape[0] == 1

# --- 4. Property Valuation Engine Tests ---
def test_valuation_inference():
    engine = PropertyValuationEngine()
    prop_input = {
        "locality": "Baner", "property_type": "Apartment", "bhk": 2,
        "carpet_area_sqft": 850.0, "floor": 5, "total_floors": 14,
        "property_age_years": 2.0, "bathrooms": 2, "furnishing": "Semi-Furnished",
        "possession_status": "Ready to Move", "amenities_count": 6
    }
    res = engine.estimate_value(prop_input, log_to_db=False)
    assert isinstance(res, ValuationResult)
    assert res.estimated_price_lakhs > 10.0
    assert res.estimated_price_per_sqft > 3000.0
    assert res.uncertainty_lower_lakhs < res.estimated_price_lakhs
    assert res.uncertainty_upper_lakhs > res.estimated_price_lakhs

def test_valuation_invalid_inputs():
    engine = PropertyValuationEngine()
    # Floor higher than total floors
    bad_input = {
        "locality": "Baner", "property_type": "Apartment", "bhk": 2,
        "carpet_area_sqft": 850.0, "floor": 20, "total_floors": 10,
        "property_age_years": 2.0, "bathrooms": 2
    }
    with pytest.raises(ValueError, match="Total floors"):
        engine.estimate_value(bad_input, log_to_db=False)

    # Carpet area out of bounds
    bad_carpet = {
        "locality": "Baner", "property_type": "Apartment", "bhk": 2,
        "carpet_area_sqft": 50.0, "floor": 2, "total_floors": 10,
        "property_age_years": 2.0, "bathrooms": 2
    }
    with pytest.raises(ValueError, match="Carpet area"):
        engine.estimate_value(bad_carpet, log_to_db=False)

# --- 5. Fair-Price Analyzer Tests ---
def test_fair_price_classifications():
    analyzer = FairPriceAnalyzer()
    prop = {
        "locality": "Baner", "property_type": "Apartment", "bhk": 2,
        "carpet_area_sqft": 850.0, "floor": 5, "total_floors": 14,
        "property_age_years": 2.0, "bathrooms": 2, "furnishing": "Semi-Furnished",
        "possession_status": "Ready to Move", "amenities_count": 6
    }
    val = analyzer.engine.estimate_value(prop, log_to_db=False)
    est = val.estimated_price_lakhs

    # 1. Undervalued case: asking is significantly lower than lower bound
    res_under = analyzer.analyze(prop, asking_price_lakhs=val.uncertainty_lower_lakhs - 15.0, log_to_db=False)
    assert res_under.classification == "Potentially Undervalued"
    assert res_under.percentage_diff_pct < 0

    # 2. Fair case: asking equals model estimate
    res_fair = analyzer.analyze(prop, asking_price_lakhs=est, log_to_db=False)
    assert res_fair.classification == "Within Expected Range"
    assert abs(res_fair.percentage_diff_pct) < 1.0

    # 3. Overpriced case: asking is higher than upper bound
    res_over = analyzer.analyze(prop, asking_price_lakhs=val.uncertainty_upper_lakhs + 20.0, log_to_db=False)
    assert res_over.classification == "Potentially Overpriced"
    assert res_over.percentage_diff_pct > 0

def test_fair_price_zero_negative_asking():
    analyzer = FairPriceAnalyzer()
    prop = {"locality": "Baner", "carpet_area_sqft": 800.0, "bhk": 2, "floor": 2, "total_floors": 10, "bathrooms": 2}
    with pytest.raises(ValueError):
        analyzer.analyze(prop, asking_price_lakhs=0.0)
    with pytest.raises(ValueError):
        analyzer.analyze(prop, asking_price_lakhs=-50.0)

# --- 6. Time-Series Forecasting & Eligibility Tests ---
def test_forecast_eligibility():
    engine = LocalityForecastEngine()
    eligible, _ = engine.check_eligibility("Baner", target_year=2026)
    assert eligible is True

    # Reject horizon > 2028 per PRD
    rejected, reason = engine.check_eligibility("Baner", target_year=2035)
    assert rejected is False
    assert "exceeds empirical forecasting limits" in reason

def test_forecast_generation():
    engine = LocalityForecastEngine()
    fc = engine.generate_forecast("Baner", steps_ahead=4)
    assert len(fc["forecast_quarters"]) == 4
    assert len(fc["forecast_table"]) == 4
    # Verify uncertainty interval widens monotonically
    spreads = fc["forecast_table"]["Uncertainty Spread"].values
    assert spreads[-1] > spreads[0]

def test_forecast_backtesting():
    engine = LocalityForecastEngine()
    bt = engine.backtest_models("Baner", test_quarters=4)
    assert "ARIMA(1,1,1)" in bt["models"]
    assert "Moving Average (4Q)" in bt["models"]
    assert bt["best_backtest_mape_pct"] < 10.0

# --- 7. SQLite Persistence & Audit Logging Tests ---
def test_database_logging():
    init_db()
    test_input = {
        "locality": "Kharadi", "property_type": "Apartment", "bhk": 3,
        "carpet_area_sqft": 1300.0, "floor": 6, "total_floors": 16,
        "property_age_years": 2.0, "bathrooms": 3, "furnishing": "Semi-Furnished",
        "possession_status": "Ready to Move", "amenities_count": 7
    }
    inserted_id = log_prediction(
        input_data=test_input,
        estimated_lakhs=125.5,
        lower_lakhs=119.0,
        upper_lakhs=132.0,
        model_version="1.0.0",
        model_name="Gradient Boosting",
        target_type="listing_price",
        asking_price=130.0,
        fair_price_status="Within Expected Range",
        notes="Automated Pytest Log"
    )
    assert inserted_id > 0
    recent = get_recent_predictions(5)
    assert len(recent) > 0
    assert recent.iloc[0]["locality"] == "Kharadi"

# --- 8. SHAP Explainability Tests ---
def test_shap_explanation():
    explainer = PropertyValuationExplainer()
    prop_sample = {
        "locality": "Kothrud", "property_type": "Apartment", "bhk": 3,
        "carpet_area_sqft": 1400.0, "floor": 4, "total_floors": 12,
        "property_age_years": 3.0, "bathrooms": 3, "furnishing": "Fully Furnished",
        "possession_status": "Ready to Move", "amenities_count": 8
    }
    exp = explainer.explain_instance(prop_sample)
    assert "base_value_lakhs" in exp
    assert "top_contributions" in exp
    assert len(exp["top_contributions"]) > 0
    # Verify Plotly waterfall figure creation
    fig = explainer.plot_waterfall(exp)
    assert fig is not None
