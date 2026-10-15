"""
Feature Engineering and Preprocessing Pipeline for PropSight.
Enforces strict target-leakage prevention and robust unknown-category handling.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.data.schema import MODEL_FEATURE_COLUMNS
from src.data.synthetic_generator import LOCALITY_PROFILES

class RealEstateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer non-leaking domain ratios:
    - floor_ratio: relative vertical position
    - area_per_bhk: spaciousness metric
    - bath_per_bhk: luxury/convenience ratio
    - built_to_carpet_ratio: loading efficiency
    - locality_tier: mapped from locality profiles
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle both DataFrame and dict/ndarray
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=MODEL_FEATURE_COLUMNS)

        # Region fallback if missing
        if "region" not in df.columns or df["region"].isnull().any():
            df["region"] = df["locality"].apply(
                lambda loc: LOCALITY_PROFILES.get(loc, {}).get("region", "Pune")
            )

        # Ratios
        total_f = df["total_floors"].replace(0, 1)
        df["floor_ratio"] = df["floor"] / total_f
        df["area_per_bhk"] = df["carpet_area_sqft"] / df["bhk"].replace(0, 1)
        df["bath_per_bhk"] = df["bathrooms"] / df["bhk"].replace(0, 1)
        
        carpet = df["carpet_area_sqft"].replace(0, 1)
        built = df["built_up_area_sqft"].fillna(df["carpet_area_sqft"] * 1.25)
        df["built_to_carpet_ratio"] = built / carpet

        # Locality Tier
        df["locality_tier"] = df["locality"].apply(
            lambda loc: LOCALITY_PROFILES.get(loc, {}).get("tier", "Mid-Market")
        )

        return df

def build_preprocessor() -> Tuple[Pipeline, List[str], List[str]]:
    """
    Build a complete scikit-learn ColumnTransformer pipeline for preprocessing.
    Categorical: OneHotEncoder with handle_unknown='ignore'
    Numerical: StandardScaler
    """
    categorical_features = [
        "locality",
        "region",
        "property_type",
        "furnishing",
        "possession_status",
        "locality_tier"
    ]

    numerical_features = [
        "bhk",
        "carpet_area_sqft",
        "built_up_area_sqft",
        "floor",
        "total_floors",
        "property_age_years",
        "bathrooms",
        "balconies",
        "parking_spaces",
        "amenities_count",
        "floor_ratio",
        "area_per_bhk",
        "bath_per_bhk",
        "built_to_carpet_ratio"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline(steps=[
        ("feature_engineer", RealEstateFeatureEngineer()),
        ("col_transform", preprocessor)
    ])

    return full_pipeline, numerical_features, categorical_features

if __name__ == "__main__":
    from src.data.loader import DataLoader
    df = DataLoader.load_properties(cleaned=True)
    pipe, num_f, cat_f = build_preprocessor()
    X = df[MODEL_FEATURE_COLUMNS]
    X_trans = pipe.fit_transform(X)
    print(f"Feature engineering pipeline verified! Output shape: {X_trans.shape}")
