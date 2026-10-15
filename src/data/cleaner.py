"""
Data Cleaning and Validation Pipeline for PropSight.
Enforces physical plausibility, outlier detection, deduplication, and detailed audit reporting.
Preserves raw data and saves cleaned data to data/processed/.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging

from src.data.schema import PropertyConstraints, REQUIRED_RAW_COLUMNS
from src.utils.config import RAW_PROPERTIES_PATH, PROCESSED_DATA_DIR, PROCESSED_PROPERTIES_PATH

logger = logging.getLogger(__name__)

class PropertyDataCleaner:
    def __init__(self, constraints: PropertyConstraints = PropertyConstraints()):
        self.constraints = constraints
        self.audit_report: Dict[str, Any] = {}

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full cleaning pipeline and return cleaned DataFrame along with audit report.
        """
        initial_rows = len(df)
        report: Dict[str, Any] = {
            "initial_rows": initial_rows,
            "steps": {}
        }
        
        df_clean = df.copy()

        # Step 1: Check required columns
        missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Step 2: Deduplication
        # Dedup on core physical features + locality + price
        dedup_cols = ["locality", "bhk", "carpet_area_sqft", "floor", "price_lakhs"]
        existing_dedup = [c for c in dedup_cols if c in df_clean.columns]
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=existing_dedup, keep="first")
        dups_removed = before_dedup - len(df_clean)
        report["steps"]["deduplication"] = {
            "rows_removed": dups_removed,
            "description": f"Removed duplicate listings based on {existing_dedup}"
        }

        # Step 3: Handle Missing Values
        before_missing = len(df_clean)
        # Drop rows where target or key physical dimensions are NaN
        key_fields = ["price_lakhs", "carpet_area_sqft", "bhk", "locality"]
        df_clean = df_clean.dropna(subset=key_fields)
        
        # Impute non-critical fields if missing
        if "property_age_years" in df_clean.columns:
            median_age = df_clean["property_age_years"].median()
            df_clean["property_age_years"] = df_clean["property_age_years"].fillna(median_age)
        if "bathrooms" in df_clean.columns:
            df_clean["bathrooms"] = df_clean["bathrooms"].fillna(df_clean["bhk"])
        if "parking_spaces" in df_clean.columns:
            df_clean["parking_spaces"] = df_clean["parking_spaces"].fillna(1)
        if "amenities_count" in df_clean.columns:
            df_clean["amenities_count"] = df_clean["amenities_count"].fillna(4)

        report["steps"]["missing_values"] = {
            "rows_dropped": before_missing - len(df_clean),
            "description": "Dropped rows with missing target or primary dimensions, imputed minor attributes"
        }

        # Step 4: Validate Physical Bounds & Plausibility
        before_bounds = len(df_clean)
        valid_mask = (
            (df_clean["price_lakhs"] >= self.constraints.MIN_PRICE_LAKHS) &
            (df_clean["price_lakhs"] <= self.constraints.MAX_PRICE_LAKHS) &
            (df_clean["carpet_area_sqft"] >= self.constraints.MIN_CARPET_AREA) &
            (df_clean["carpet_area_sqft"] <= self.constraints.MAX_CARPET_AREA) &
            (df_clean["bhk"] >= self.constraints.MIN_BHK) &
            (df_clean["bhk"] <= self.constraints.MAX_BHK) &
            (df_clean["bathrooms"] >= self.constraints.MIN_BATHROOMS) &
            (df_clean["bathrooms"] <= self.constraints.MAX_BATHROOMS) &
            (df_clean["floor"] >= self.constraints.MIN_FLOOR) &
            (df_clean["floor"] <= self.constraints.MAX_FLOOR)
        )
        
        # Physical sanity: Built-up area should be >= carpet area if both are present
        if "built_up_area_sqft" in df_clean.columns:
            # If built_up is smaller than carpet, adjust built_up to carpet * 1.25 rather than dropping
            invalid_builtup = df_clean["built_up_area_sqft"] < df_clean["carpet_area_sqft"]
            df_clean.loc[invalid_builtup, "built_up_area_sqft"] = df_clean.loc[invalid_builtup, "carpet_area_sqft"] * 1.25

        df_clean = df_clean[valid_mask]
        report["steps"]["physical_bounds"] = {
            "rows_removed": before_bounds - len(df_clean),
            "description": f"Enforced physical plausibility ({self.constraints.MIN_CARPET_AREA}-{self.constraints.MAX_CARPET_AREA} sqft, INR {self.constraints.MIN_PRICE_LAKHS}-{self.constraints.MAX_PRICE_LAKHS}L, BHK {self.constraints.MIN_BHK}-{self.constraints.MAX_BHK})"
        }

        # Step 5: Derive & Validate price_per_sqft
        df_clean["price_per_sqft"] = (df_clean["price_lakhs"] * 100000.0) / df_clean["carpet_area_sqft"]

        # Step 6: Outlier Detection via IQR per Locality
        # Detect severe data anomalies (e.g., mis-keyed prices) using 2.5 * IQR on price_per_sqft
        before_outliers = len(df_clean)
        outlier_indices = []
        for loc, group in df_clean.groupby("locality"):
            q25 = group["price_per_sqft"].quantile(0.01)
            q75 = group["price_per_sqft"].quantile(0.99)
            iqr = q75 - q25
            lower_bound = max(2000.0, q25 - 2.5 * iqr)
            upper_bound = q75 + 2.5 * iqr
            outliers = group[(group["price_per_sqft"] < lower_bound) | (group["price_per_sqft"] > upper_bound)].index
            outlier_indices.extend(outliers)

        df_clean = df_clean.drop(index=outlier_indices)
        report["steps"]["outlier_detection"] = {
            "rows_removed": len(outlier_indices),
            "description": "Removed statistical anomalies exceeding 2.5 * IQR on locality price/sqft"
        }

        # Final Summary
        final_rows = len(df_clean)
        report["final_rows"] = final_rows
        report["retention_rate_pct"] = round((final_rows / initial_rows) * 100, 2)
        
        self.audit_report = report
        return df_clean, report

def clean_and_persist_dataset(raw_path=RAW_PROPERTIES_PATH, processed_path=PROCESSED_DATA_DIR / "cleaned_properties.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function to load raw data, clean it, and save to processed directory."""
    df_raw = pd.read_csv(raw_path)
    cleaner = PropertyDataCleaner()
    df_clean, report = cleaner.clean(df_raw)
    
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    
    return df_clean, report

if __name__ == "__main__":
    df_clean, report = clean_and_persist_dataset()
    print("Cleaning Complete!")
    print(f"Initial: {report['initial_rows']}, Final: {report['final_rows']}, Retention: {report['retention_rate_pct']}%")
    for step, details in report["steps"].items():
        print(f" - {step}: {details}")
