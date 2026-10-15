"""
Data Loader and Repository Service for PropSight.
Provides unified access to properties, historical trends, and locality metadata
with caching and data quality summary utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.utils.config import (
    RAW_PROPERTIES_PATH,
    PROCESSED_PROPERTIES_PATH,
    RAW_HISTORICAL_PATH,
    LOCALITY_METADATA_PATH,
    TARGET_COLUMN,
    TARGET_TYPE
)
from src.data.synthetic_generator import ensure_raw_datasets_exist
from src.data.cleaner import clean_and_persist_dataset

class DataLoader:
    """Unified data access and diagnostics provider."""

    @staticmethod
    def load_properties(cleaned: bool = True) -> pd.DataFrame:
        """Load property records (cleaned by default)."""
        if cleaned:
            if not PROCESSED_PROPERTIES_PATH.exists():
                ensure_raw_datasets_exist()
                df_clean, _ = clean_and_persist_dataset()
                return df_clean
            return pd.read_csv(PROCESSED_PROPERTIES_PATH)
        else:
            if not RAW_PROPERTIES_PATH.exists():
                ensure_raw_datasets_exist()
            return pd.read_csv(RAW_PROPERTIES_PATH)

    @staticmethod
    def load_historical_trends() -> pd.DataFrame:
        """Load quarterly locality price indices (2018-Q1 to 2024-Q4)."""
        if not RAW_HISTORICAL_PATH.exists():
            ensure_raw_datasets_exist()
        return pd.read_csv(RAW_HISTORICAL_PATH)

    @staticmethod
    def load_locality_metadata() -> pd.DataFrame:
        """Load locality geographic coordinates and infrastructure metadata."""
        if not LOCALITY_METADATA_PATH.exists():
            ensure_raw_datasets_exist()
        return pd.read_csv(LOCALITY_METADATA_PATH)

    @staticmethod
    def get_dataset_diagnostics() -> Dict[str, Any]:
        """Compute comprehensive data quality and coverage statistics."""
        df_raw = DataLoader.load_properties(cleaned=False)
        df_clean = DataLoader.load_properties(cleaned=True)
        df_hist = DataLoader.load_historical_trends()
        df_meta = DataLoader.load_locality_metadata()

        # Missing values summary
        missing_summary = df_raw.isnull().sum().to_dict()

        # Locality distribution
        locality_counts = df_clean["locality"].value_counts().to_dict()

        # Numerical statistics
        num_cols = ["carpet_area_sqft", "price_lakhs", "price_per_sqft", "bhk", "bathrooms", "floor", "property_age_years"]
        num_summary = df_clean[num_cols].describe().to_dict()

        return {
            "total_raw_listings": len(df_raw),
            "total_clean_listings": len(df_clean),
            "locality_count": df_clean["locality"].nunique(),
            "region_breakdown": df_clean["region"].value_counts().to_dict(),
            "target_column": TARGET_COLUMN,
            "target_type": TARGET_TYPE,
            "is_synthetic": bool(df_clean["is_synthetic"].iloc[0]) if "is_synthetic" in df_clean.columns else False,
            "data_source": str(df_clean["data_source"].iloc[0]) if "data_source" in df_clean.columns else "Unknown",
            "missing_values": missing_summary,
            "locality_distribution": locality_counts,
            "numerical_summary": num_summary,
            "historical_quarters_count": df_hist["quarter"].nunique(),
            "historical_date_range": f"{df_hist['quarter'].min()} to {df_hist['quarter'].max()}"
        }

if __name__ == "__main__":
    diag = DataLoader.get_dataset_diagnostics()
    print("Diagnostics Summary:")
    print(f"Clean Records: {diag['total_clean_listings']}, Localities: {diag['locality_count']}")
    print(f"Target: {diag['target_column']} ({diag['target_type']})")
    print(f"Historical Quarters: {diag['historical_quarters_count']} ({diag['historical_date_range']})")
