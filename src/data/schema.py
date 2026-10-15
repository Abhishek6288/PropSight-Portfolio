"""
Data Schema and Validation Definitions for PropSight.
Enforces realistic physical bounds and field types for Pune/PCMC property data.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class PropertyConstraints:
    MIN_CARPET_AREA: float = 250.0      # sq.ft.
    MAX_CARPET_AREA: float = 10000.0    # sq.ft.
    MIN_BUILT_UP_AREA: float = 300.0    # sq.ft.
    MAX_BUILT_UP_AREA: float = 12000.0  # sq.ft.
    MIN_BHK: int = 1
    MAX_BHK: int = 6
    MIN_BATHROOMS: int = 1
    MAX_BATHROOMS: int = 8
    MIN_FLOOR: int = 0
    MAX_FLOOR: int = 60
    MIN_AGE: float = 0.0
    MAX_AGE: float = 50.0
    MIN_PRICE_LAKHS: float = 10.0       # ₹10 Lakhs minimum realistic listing in Pune/PCMC
    MAX_PRICE_LAKHS: float = 2500.0     # ₹25 Crores maximum realistic listing

PROPERTY_SCHEMA_TYPES: Dict[str, str] = {
    "property_id": "string",
    "locality": "category",
    "region": "category",
    "property_type": "category",
    "bhk": "int64",
    "carpet_area_sqft": "float64",
    "built_up_area_sqft": "float64",
    "floor": "int64",
    "total_floors": "int64",
    "property_age_years": "float64",
    "bathrooms": "int64",
    "balconies": "int64",
    "parking_spaces": "int64",
    "furnishing": "category",
    "possession_status": "category",
    "amenities_count": "int64",
    "price_lakhs": "float64",
    "price_per_sqft": "float64",
    "listing_date": "datetime64[ns]",
    "target_type": "string",
    "is_synthetic": "bool",
    "data_source": "string"
}

REQUIRED_RAW_COLUMNS = [
    "locality",
    "property_type",
    "bhk",
    "carpet_area_sqft",
    "built_up_area_sqft",
    "floor",
    "total_floors",
    "property_age_years",
    "bathrooms",
    "balconies",
    "parking_spaces",
    "furnishing",
    "possession_status",
    "amenities_count",
    "price_lakhs"
]

MODEL_FEATURE_COLUMNS = [
    "locality",
    "region",
    "property_type",
    "bhk",
    "carpet_area_sqft",
    "built_up_area_sqft",
    "floor",
    "total_floors",
    "property_age_years",
    "bathrooms",
    "balconies",
    "parking_spaces",
    "furnishing",
    "possession_status",
    "amenities_count"
]
