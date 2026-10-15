"""
Configuration and Project Paths for PropSight.
Ensures robust project-root resolution and reproducible random seeds.
"""

from pathlib import Path
import os

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Core directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
DATABASE_DIR = PROJECT_ROOT / "data" / "db"
DB_PATH = DATABASE_DIR / "propsight.db"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure essential directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
RAW_PROPERTIES_PATH = RAW_DATA_DIR / "pune_pcmc_properties.csv"
RAW_HISTORICAL_PATH = RAW_DATA_DIR / "pune_pcmc_historical_trends.csv"
LOCALITY_METADATA_PATH = RAW_DATA_DIR / "pune_pcmc_localities.csv"

PROCESSED_PROPERTIES_PATH = PROCESSED_DATA_DIR / "cleaned_properties.csv"
MODEL_PIPELINE_PATH = ARTIFACTS_DIR / "valuation_pipeline.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
SHAP_BACKGROUND_PATH = ARTIFACTS_DIR / "shap_background.joblib"

# Global Random State for Reproducibility
RANDOM_SEED = 42

# Valuation Model Target Definition
TARGET_COLUMN = "price_lakhs"
TARGET_TYPE = "listing_price"  # clearly distinguished from registered_transaction

# Supported Localities in Pune and PCMC
SUPPORTED_LOCALITIES = [
    # Pune West & IT Corridor
    "Baner", "Balewadi", "Aundh", "Hinjewadi", "Wakad", "Bavdhan", "Pashan",
    # Pune Central & Premium
    "Kothrud", "Shivajinagar", "Koregaon Park", "Kalyani Nagar", "Erandwane",
    # Pune East & Nagar Rd / Solapur Rd
    "Kharadi", "Viman Nagar", "Hadapsar", "Magarpatta", "Wagholi", "Dhanori",
    # PCMC (Pimpri-Chinchwad)
    "Pimple Saudagar", "Pimple Nilakh", "Ravet", "Punawale", "Tathawade",
    "Moshi", "Nigdi", "Akurdi", "Chikhali", "Bhosari"
]

PROPERTY_TYPES = [
    "Apartment",
    "Independent House / Villa",
    "Penthouse",
    "Builder Floor"
]

FURNISHING_STATUSES = [
    "Unfurnished",
    "Semi-Furnished",
    "Fully Furnished"
]

POSSESSION_STATUSES = [
    "Ready to Move",
    "Under Construction"
]

AMENITIES_LIST = [
    "Clubhouse",
    "Swimming Pool",
    "Gymnasium",
    "24x7 Security",
    "Power Backup",
    "High Speed Elevators",
    "Covered Parking",
    "Children Play Area",
    "Intercom Facility",
    "Jogging Track",
    "Landscaped Garden"
]
