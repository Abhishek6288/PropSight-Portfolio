"""
SQLite Database Layer for PropSight.
Provides idempotent schema creation, parameterized queries, and audit logging for predictions.
"""

import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.utils.config import DB_PATH
from src.data.loader import DataLoader

def get_connection() -> sqlite3.Connection:
    """Obtain a SQLite connection with row-factory support."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(force_reinit: bool = False):
    """
    Idempotent database initialization.
    Creates tables for properties, localities, historical_prices, and prediction_logs.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if force_reinit:
        cursor.execute("DROP TABLE IF EXISTS prediction_logs")
        cursor.execute("DROP TABLE IF EXISTS historical_prices")
        cursor.execute("DROP TABLE IF EXISTS localities")
        cursor.execute("DROP TABLE IF EXISTS properties")

    # Table: localities
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS localities (
        locality TEXT PRIMARY KEY,
        region TEXT NOT NULL,
        tier TEXT NOT NULL,
        base_rate_sqft REAL,
        latitude REAL,
        longitude REAL,
        pincode INTEGER,
        metro_connectivity INTEGER,
        it_hub_proximity INTEGER,
        is_synthetic INTEGER DEFAULT 1,
        data_source TEXT
    );
    """)

    # Table: properties
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS properties (
        property_id TEXT PRIMARY KEY,
        locality TEXT NOT NULL,
        region TEXT NOT NULL,
        property_type TEXT NOT NULL,
        bhk INTEGER NOT NULL,
        carpet_area_sqft REAL NOT NULL,
        built_up_area_sqft REAL,
        floor INTEGER,
        total_floors INTEGER,
        property_age_years REAL,
        bathrooms INTEGER,
        balconies INTEGER,
        parking_spaces INTEGER,
        furnishing TEXT,
        possession_status TEXT,
        amenities_count INTEGER,
        price_lakhs REAL NOT NULL,
        price_per_sqft REAL,
        listing_date TEXT,
        target_type TEXT,
        is_synthetic INTEGER DEFAULT 1,
        data_source TEXT,
        FOREIGN KEY (locality) REFERENCES localities(locality)
    );
    """)

    # Table: historical_prices
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        locality TEXT NOT NULL,
        region TEXT NOT NULL,
        quarter TEXT NOT NULL,
        year INTEGER NOT NULL,
        quarter_num INTEGER NOT NULL,
        avg_price_per_sqft REAL NOT NULL,
        median_price_per_sqft REAL,
        quarterly_growth_pct REAL,
        sample_transaction_volume INTEGER,
        is_synthetic INTEGER DEFAULT 1,
        data_source TEXT,
        UNIQUE(locality, quarter)
    );
    """)

    # Table: prediction_logs (Audit trail for valuation requests)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        locality TEXT NOT NULL,
        property_type TEXT NOT NULL,
        bhk INTEGER NOT NULL,
        carpet_area_sqft REAL NOT NULL,
        floor INTEGER,
        total_floors INTEGER,
        property_age_years REAL,
        bathrooms INTEGER,
        furnishing TEXT,
        possession_status TEXT,
        amenities_count INTEGER,
        estimated_price_lakhs REAL NOT NULL,
        estimated_price_per_sqft REAL,
        uncertainty_lower_lakhs REAL,
        uncertainty_upper_lakhs REAL,
        model_version TEXT,
        model_name TEXT,
        target_type TEXT,
        asking_price_lakhs REAL,
        fair_price_status TEXT,
        notes TEXT
    );
    """)

    # Indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_properties_locality ON properties(locality);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_properties_bhk ON properties(bhk);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_locality ON historical_prices(locality);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_quarter ON historical_prices(quarter);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON prediction_logs(timestamp);")

    conn.commit()

    # Pre-populate properties and historical if empty
    cursor.execute("SELECT COUNT(*) FROM properties")
    prop_count = cursor.fetchone()[0]
    if prop_count == 0:
        _populate_seed_data(conn)

    conn.close()

def _populate_seed_data(conn: sqlite3.Connection):
    """Seed SQLite database with existing cleaned and historical datasets."""
    df_props = DataLoader.load_properties(cleaned=True)
    df_hist = DataLoader.load_historical_trends()
    df_meta = DataLoader.load_locality_metadata()

    df_meta.to_sql("localities", conn, if_exists="replace", index=False)
    df_props.to_sql("properties", conn, if_exists="replace", index=False)
    df_hist.to_sql("historical_prices", conn, if_exists="replace", index=False)
    conn.commit()

def log_prediction(
    input_data: Dict[str, Any],
    estimated_lakhs: float,
    lower_lakhs: float,
    upper_lakhs: float,
    model_version: str,
    model_name: str,
    target_type: str,
    asking_price: Optional[float] = None,
    fair_price_status: Optional[str] = None,
    notes: Optional[str] = None
) -> int:
    """Log an inference transaction into prediction_logs."""
    conn = get_connection()
    cursor = conn.cursor()

    carpet = float(input_data.get("carpet_area_sqft", 0.0))
    rate_per_sqft = (estimated_lakhs * 100000.0 / carpet) if carpet > 0 else None

    cursor.execute("""
    INSERT INTO prediction_logs (
        timestamp, locality, property_type, bhk, carpet_area_sqft,
        floor, total_floors, property_age_years, bathrooms, furnishing,
        possession_status, amenities_count, estimated_price_lakhs,
        estimated_price_per_sqft, uncertainty_lower_lakhs, uncertainty_upper_lakhs,
        model_version, model_name, target_type, asking_price_lakhs,
        fair_price_status, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        input_data.get("locality"),
        input_data.get("property_type"),
        int(input_data.get("bhk", 2)),
        carpet,
        int(input_data.get("floor", 1)),
        int(input_data.get("total_floors", 10)),
        float(input_data.get("property_age_years", 0.0)),
        int(input_data.get("bathrooms", 2)),
        input_data.get("furnishing"),
        input_data.get("possession_status"),
        int(input_data.get("amenities_count", 4)),
        round(estimated_lakhs, 2),
        round(rate_per_sqft, 1) if rate_per_sqft else None,
        round(lower_lakhs, 2),
        round(upper_lakhs, 2),
        model_version,
        model_name,
        target_type,
        round(asking_price, 2) if asking_price is not None else None,
        fair_price_status,
        notes
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_recent_predictions(limit: int = 25) -> pd.DataFrame:
    """Retrieve recent prediction audit logs."""
    conn = get_connection()
    query = "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
    recent = get_recent_predictions(5)
    print(f"Prediction logs retrieved: {len(recent)}")
