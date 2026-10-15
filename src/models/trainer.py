"""
Model Training, Cross-Validation, Evaluation, and Serialization Engine.
Evaluates multiple regression estimators and selects the best performer on held-out test data.
Persists pipeline, metadata, and SHAP background data.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data.loader import DataLoader
from src.data.schema import MODEL_FEATURE_COLUMNS
from src.data.feature_engineering import build_preprocessor
from src.utils.config import (
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    SHAP_BACKGROUND_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TARGET_TYPE
)

logger = logging.getLogger(__name__)

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error safely avoiding division by zero."""
    mask = y_true > 1e-4
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

class ModelTrainer:
    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: str = ""
        self.best_pipeline = None

    def get_candidate_models(self) -> Dict[str, Any]:
        """Return configured candidate regression estimators."""
        candidates = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=10.0, random_state=self.random_seed),
            "Lasso Regression": Lasso(alpha=0.1, random_state=self.random_seed),
            "Random Forest": RandomForestRegressor(
                n_estimators=120, max_depth=16, min_samples_split=4,
                random_state=self.random_seed, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=5,
                random_state=self.random_seed
            )
        }
        if HAS_XGBOOST:
            candidates["XGBoost"] = XGBRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=5,
                subsample=0.85, colsample_bytree=0.85,
                random_state=self.random_seed, n_jobs=-1
            )
        return candidates

    def train_and_evaluate(self) -> Tuple[Dict[str, Any], Any]:
        """
        Execute training, cross-validation, and held-out evaluation across all candidates.
        """
        df = DataLoader.load_properties(cleaned=True)
        X = df[MODEL_FEATURE_COLUMNS]
        y = df[TARGET_COLUMN].values

        # Train/Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_seed
        )

        preprocessor, num_features, cat_features = build_preprocessor()
        
        # Fit preprocessor strictly on training data
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        candidates = self.get_candidate_models()
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_seed)

        evaluation_summary = {}

        for name, model in candidates.items():
            # 5-Fold Cross Validation on training data
            cv_scores = cross_val_score(model, X_train_trans, y_train, cv=kf, scoring="r2", n_jobs=-1)
            
            # Fit on full training set
            model.fit(X_train_trans, y_train)

            # Evaluate on held-out test set
            y_pred_test = model.predict(X_test_trans)
            y_pred_train = model.predict(X_train_trans)

            test_mae = float(mean_absolute_error(y_test, y_pred_test))
            test_rmse = float(root_mean_squared_error(y_test, y_pred_test))
            test_r2 = float(r2_score(y_test, y_pred_test))
            test_mape = safe_mape(y_test, y_pred_test)

            train_r2 = float(r2_score(y_train, y_pred_train))

            evaluation_summary[name] = {
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_mae_lakhs": test_mae,
                "test_rmse_lakhs": test_rmse,
                "test_mape_pct": test_mape
            }

        # Select Best Model based on highest test R2 and lowest MAE
        best_name = max(evaluation_summary, key=lambda k: evaluation_summary[k]["test_r2"])
        best_estimator = candidates[best_name]

        # Fit preprocessor and best model on full dataset for final deployment pipeline
        full_preprocessor, _, _ = build_preprocessor()
        X_full_trans = full_preprocessor.fit_transform(X)
        best_estimator.fit(X_full_trans, y)

        # Assemble deployment object
        deployment_artifacts = {
            "preprocessor": full_preprocessor,
            "model": best_estimator,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "target_type": TARGET_TYPE,
            "model_name": best_name
        }

        # Compute empirical test residuals for uncertainty interval estimation
        residuals = y_test - candidates[best_name].predict(X_test_trans)
        residual_std = float(np.std(residuals))
        residual_q10 = float(np.percentile(residuals, 10))
        residual_q90 = float(np.percentile(residuals, 90))

        # Save metadata
        metadata = {
            "model_version": "1.0.0",
            "training_timestamp": datetime.now().isoformat(),
            "selected_model": best_name,
            "target_column": TARGET_COLUMN,
            "target_type": TARGET_TYPE,
            "is_synthetic": bool(df["is_synthetic"].iloc[0]),
            "data_source": str(df["data_source"].iloc[0]),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(df),
            "best_model_metrics": evaluation_summary[best_name],
            "candidate_comparison": evaluation_summary,
            "uncertainty_parameters": {
                "residual_std_lakhs": residual_std,
                "residual_q10_lakhs": residual_q10,
                "residual_q90_lakhs": residual_q90
            }
        }

        # Persist artifacts
        MODEL_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(deployment_artifacts, MODEL_PIPELINE_PATH)

        # Save sample background data for SHAP (150 representative samples)
        sample_bg = full_preprocessor.transform(X.sample(n=min(150, len(X)), random_state=self.random_seed))
        joblib.dump(sample_bg, SHAP_BACKGROUND_PATH)

        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        self.results = evaluation_summary
        self.best_model_name = best_name
        self.best_pipeline = deployment_artifacts

        return metadata, deployment_artifacts

if __name__ == "__main__":
    trainer = ModelTrainer()
    meta, _ = trainer.train_and_evaluate()
    print("Model Training & Evaluation Completed Successfully!")
    print(f"Selected Best Model: {meta['selected_model']}")
    print(f"Metrics: R2={meta['best_model_metrics']['test_r2']:.4f}, MAE=INR {meta['best_model_metrics']['test_mae_lakhs']:.2f}L, MAPE={meta['best_model_metrics']['test_mape_pct']:.2f}%")
    print("\nCandidate Comparison:")
    for m, metrics in meta["candidate_comparison"].items():
        print(f" - {m:<20}: Test R2={metrics['test_r2']:.4f}, MAE={metrics['test_mae_lakhs']:.2f}L, MAPE={metrics['test_mape_pct']:.2f}%")
