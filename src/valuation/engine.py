"""
Property Valuation Engine for PropSight.
Executes serialized preprocessing and inference pipeline with uncertainty interval estimation.
"""

import json
import joblib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.data.schema import PropertyConstraints, MODEL_FEATURE_COLUMNS
from src.data.synthetic_generator import LOCALITY_PROFILES
from src.utils.config import MODEL_PIPELINE_PATH, MODEL_METADATA_PATH
from src.database.db import log_prediction

@dataclass
class ValuationResult:
    estimated_price_lakhs: float
    estimated_price_per_sqft: float
    uncertainty_lower_lakhs: float
    uncertainty_upper_lakhs: float
    confidence_level_pct: int
    model_version: str
    model_name: str
    target_type: str
    is_synthetic: bool
    data_source: str
    prediction_id: Optional[int] = None
    warnings: Optional[str] = None

class PropertyValuationEngine:
    def __init__(self):
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained pipeline and metadata from disk."""
        if not MODEL_PIPELINE_PATH.exists() or not MODEL_METADATA_PATH.exists():
            from src.models.trainer import ModelTrainer
            trainer = ModelTrainer()
            trainer.train_and_evaluate()

        self.artifacts = joblib.load(MODEL_PIPELINE_PATH)
        self.preprocessor = self.artifacts["preprocessor"]
        self.model = self.artifacts["model"]

        with open(MODEL_METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        self.uncertainty = self.metadata.get("uncertainty_parameters", {
            "residual_std_lakhs": 6.0,
            "residual_q10_lakhs": -6.5,
            "residual_q90_lakhs": 6.5
        })

    def validate_inputs(self, property_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate input parameters against physical plausibility bounds."""
        carpet = float(property_data.get("carpet_area_sqft", 0.0))
        bhk = int(property_data.get("bhk", 0))
        bathrooms = int(property_data.get("bathrooms", 0))
        floor = int(property_data.get("floor", 0))
        total_floors = int(property_data.get("total_floors", 0))

        if carpet < PropertyConstraints.MIN_CARPET_AREA or carpet > PropertyConstraints.MAX_CARPET_AREA:
            return False, f"Carpet area must be between {PropertyConstraints.MIN_CARPET_AREA} and {PropertyConstraints.MAX_CARPET_AREA} sq.ft."
        
        if bhk < PropertyConstraints.MIN_BHK or bhk > PropertyConstraints.MAX_BHK:
            return False, f"BHK must be between {PropertyConstraints.MIN_BHK} and {PropertyConstraints.MAX_BHK}."

        if bathrooms < PropertyConstraints.MIN_BATHROOMS or bathrooms > PropertyConstraints.MAX_BATHROOMS:
            return False, f"Bathrooms must be between {PropertyConstraints.MIN_BATHROOMS} and {PropertyConstraints.MAX_BATHROOMS}."

        if floor < 0 or floor > 60:
            return False, "Floor must be between 0 (Ground) and 60."

        if total_floors < floor:
            return False, f"Total floors ({total_floors}) cannot be less than property floor ({floor})."

        return True, None

    def estimate_value(
        self,
        property_data: Dict[str, Any],
        log_to_db: bool = True,
        notes: Optional[str] = None
    ) -> ValuationResult:
        """
        Compute property valuation estimate, prediction intervals, and log transaction.
        """
        is_valid, error_msg = self.validate_inputs(property_data)
        if not is_valid:
            raise ValueError(f"Input validation error: {error_msg}")

        # Ensure all model columns are populated
        prop_record = {}
        for col in MODEL_FEATURE_COLUMNS:
            if col in property_data:
                prop_record[col] = property_data[col]
            elif col == "region":
                loc = property_data.get("locality", "Baner")
                prop_record[col] = LOCALITY_PROFILES.get(loc, {}).get("region", "Pune")
            elif col == "built_up_area_sqft":
                prop_record[col] = float(property_data.get("carpet_area_sqft", 800.0)) * 1.25
            elif col == "balconies":
                prop_record[col] = max(1, int(property_data.get("bhk", 2)) - 1)
            elif col == "parking_spaces":
                prop_record[col] = 1 if int(property_data.get("bhk", 2)) <= 2 else 2
            elif col == "amenities_count":
                prop_record[col] = 5
            elif col == "property_age_years":
                prop_record[col] = 3.0
            else:
                prop_record[col] = 0

        df_input = pd.DataFrame([prop_record])

        # Preprocessing & Inference
        X_trans = self.preprocessor.transform(df_input)
        raw_pred = float(self.model.predict(X_trans)[0])

        # Minimum physical floor in Pune/PCMC
        estimated_lakhs = max(float(PropertyConstraints.MIN_PRICE_LAKHS), round(raw_pred, 2))

        # Price per sq.ft.
        carpet = float(prop_record["carpet_area_sqft"])
        price_per_sqft = round((estimated_lakhs * 100000.0) / carpet, 1) if carpet > 0 else 0.0

        # Empirical 80% uncertainty bounds from test residuals
        q10 = self.uncertainty["residual_q10_lakhs"]
        q90 = self.uncertainty["residual_q90_lakhs"]
        lower_bound = max(float(PropertyConstraints.MIN_PRICE_LAKHS), round(estimated_lakhs + q10, 2))
        upper_bound = round(estimated_lakhs + q90, 2)

        # Warnings / Disclaimers
        warnings = None
        if self.metadata.get("is_synthetic", False):
            warnings = "Estimate generated by demonstration baseline model. For academic decision-support only."

        # DB Logging
        pred_id = None
        if log_to_db:
            pred_id = log_prediction(
                input_data=prop_record,
                estimated_lakhs=estimated_lakhs,
                lower_lakhs=lower_bound,
                upper_lakhs=upper_bound,
                model_version=self.metadata.get("model_version", "1.0.0"),
                model_name=self.metadata.get("selected_model", "Gradient Boosting"),
                target_type=self.metadata.get("target_type", "listing_price"),
                notes=notes
            )

        return ValuationResult(
            estimated_price_lakhs=estimated_lakhs,
            estimated_price_per_sqft=price_per_sqft,
            uncertainty_lower_lakhs=lower_bound,
            uncertainty_upper_lakhs=upper_bound,
            confidence_level_pct=80,
            model_version=self.metadata.get("model_version", "1.0.0"),
            model_name=self.metadata.get("selected_model", "Gradient Boosting"),
            target_type=self.metadata.get("target_type", "listing_price"),
            is_synthetic=self.metadata.get("is_synthetic", True),
            data_source=self.metadata.get("data_source", "Demonstration Model"),
            prediction_id=pred_id,
            warnings=warnings
        )

if __name__ == "__main__":
    engine = PropertyValuationEngine()
    test_input = {
        "locality": "Baner",
        "property_type": "Apartment",
        "bhk": 2,
        "carpet_area_sqft": 850.0,
        "floor": 5,
        "total_floors": 14,
        "property_age_years": 2.0,
        "bathrooms": 2,
        "furnishing": "Semi-Furnished",
        "possession_status": "Ready to Move",
        "amenities_count": 6
    }
    res = engine.estimate_value(test_input)
    print("Inference Successful:")
    print(f"Estimate: INR {res.estimated_price_lakhs} Lakhs (Rate: INR {res.estimated_price_per_sqft}/sqft)")
    print(f"80% Uncertainty Range: INR {res.uncertainty_lower_lakhs}L - {res.uncertainty_upper_lakhs}L")
    print(f"Model: {res.model_name} (Logged ID: {res.prediction_id})")
