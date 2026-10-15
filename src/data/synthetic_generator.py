"""
Calibrated Demonstration Dataset Generator for Pune and Pimpri-Chinchwad (PCMC).
Generates academically defensible, realistic property listings, locality metadata,
and quarterly time-series historical price indices (2018-Q1 to 2024-Q4).

IMPORTANT:
Every generated record is explicitly labeled with:
- `is_synthetic = True`
- `data_source = 'PropSight Demonstration Dataset (Calibrated to Pune/PCMC Market Reports 2018-2024)'`
- `target_type = 'listing_price'`
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
from pathlib import Path

from src.utils.config import (
    RAW_PROPERTIES_PATH,
    RAW_HISTORICAL_PATH,
    LOCALITY_METADATA_PATH,
    RANDOM_SEED
)

# Detailed micro-market parameters based on published Pune/PCMC residential indices
LOCALITY_PROFILES = {
    # Premium Pune Central
    "Koregaon Park": {"region": "Pune", "base_rate": 14500, "rate_std": 1800, "tier": "Premium", "lat": 18.5362, "lon": 73.8940, "pincode": 411001, "metro": 1, "it_hub": 0},
    "Kalyani Nagar": {"region": "Pune", "base_rate": 12800, "rate_std": 1500, "tier": "Premium", "lat": 18.5463, "lon": 73.9034, "pincode": 411006, "metro": 1, "it_hub": 1},
    "Shivajinagar": {"region": "Pune", "base_rate": 13500, "rate_std": 1600, "tier": "Premium", "lat": 18.5314, "lon": 73.8446, "pincode": 411005, "metro": 1, "it_hub": 0},
    "Kothrud": {"region": "Pune", "base_rate": 11800, "rate_std": 1300, "tier": "Prime Established", "lat": 18.5074, "lon": 73.8077, "pincode": 411038, "metro": 1, "it_hub": 0},
    "Erandwane": {"region": "Pune", "base_rate": 13200, "rate_std": 1400, "tier": "Premium", "lat": 18.5098, "lon": 73.8329, "pincode": 411004, "metro": 1, "it_hub": 0},

    # Pune West & IT Corridor
    "Baner": {"region": "Pune", "base_rate": 9600, "rate_std": 1100, "tier": "Prime West", "lat": 18.5590, "lon": 73.7868, "pincode": 411045, "metro": 1, "it_hub": 1},
    "Balewadi": {"region": "Pune", "base_rate": 9200, "rate_std": 1000, "tier": "Prime West", "lat": 18.5726, "lon": 73.7711, "pincode": 411045, "metro": 1, "it_hub": 1},
    "Aundh": {"region": "Pune", "base_rate": 10500, "rate_std": 1200, "tier": "Prime West", "lat": 18.5580, "lon": 73.8074, "pincode": 411007, "metro": 1, "it_hub": 0},
    "Hinjewadi": {"region": "Pune", "base_rate": 7400, "rate_std": 800, "tier": "IT Hub", "lat": 18.5913, "lon": 73.7389, "pincode": 411057, "metro": 1, "it_hub": 1},
    "Wakad": {"region": "Pune", "base_rate": 8200, "rate_std": 900, "tier": "IT Corridor", "lat": 18.5987, "lon": 73.7628, "pincode": 411057, "metro": 1, "it_hub": 1},
    "Bavdhan": {"region": "Pune", "base_rate": 8400, "rate_std": 850, "tier": "West Mid-Market", "lat": 18.5158, "lon": 73.7707, "pincode": 411021, "metro": 0, "it_hub": 0},
    "Pashan": {"region": "Pune", "base_rate": 8800, "rate_std": 900, "tier": "West Mid-Market", "lat": 18.5414, "lon": 73.7925, "pincode": 411021, "metro": 0, "it_hub": 0},

    # Pune East & Nagar / Solapur Rd
    "Kharadi": {"region": "Pune", "base_rate": 9100, "rate_std": 1050, "tier": "IT Hub", "lat": 18.5514, "lon": 73.9348, "pincode": 411014, "metro": 1, "it_hub": 1},
    "Viman Nagar": {"region": "Pune", "base_rate": 10200, "rate_std": 1150, "tier": "Prime East", "lat": 18.5679, "lon": 73.9143, "pincode": 411014, "metro": 1, "it_hub": 1},
    "Hadapsar": {"region": "Pune", "base_rate": 7600, "rate_std": 800, "tier": "East Mid-Market", "lat": 18.5089, "lon": 73.9259, "pincode": 411028, "metro": 0, "it_hub": 1},
    "Magarpatta": {"region": "Pune", "base_rate": 9500, "rate_std": 950, "tier": "Integrated Township", "lat": 18.5152, "lon": 73.9281, "pincode": 411028, "metro": 0, "it_hub": 1},
    "Wagholi": {"region": "Pune", "base_rate": 5800, "rate_std": 650, "tier": "Growth Corridor", "lat": 18.5808, "lon": 73.9787, "pincode": 412207, "metro": 0, "it_hub": 0},
    "Dhanori": {"region": "Pune", "base_rate": 6200, "rate_std": 700, "tier": "Growth Corridor", "lat": 18.5880, "lon": 73.8893, "pincode": 411015, "metro": 0, "it_hub": 0},

    # PCMC (Pimpri-Chinchwad)
    "Pimple Saudagar": {"region": "PCMC", "base_rate": 8100, "rate_std": 800, "tier": "PCMC Prime", "lat": 18.5987, "lon": 73.7978, "pincode": 411027, "metro": 1, "it_hub": 1},
    "Pimple Nilakh": {"region": "PCMC", "base_rate": 8600, "rate_std": 850, "tier": "PCMC Prime", "lat": 18.5818, "lon": 73.7844, "pincode": 411027, "metro": 0, "it_hub": 1},
    "Ravet": {"region": "PCMC", "base_rate": 6600, "rate_std": 700, "tier": "PCMC Growth", "lat": 18.6438, "lon": 73.7516, "pincode": 412101, "metro": 0, "it_hub": 0},
    "Punawale": {"region": "PCMC", "base_rate": 6300, "rate_std": 650, "tier": "PCMC Growth", "lat": 18.6258, "lon": 73.7431, "pincode": 411033, "metro": 0, "it_hub": 0},
    "Tathawade": {"region": "PCMC", "base_rate": 6800, "rate_std": 700, "tier": "PCMC Growth", "lat": 18.6186, "lon": 73.7554, "pincode": 411033, "metro": 0, "it_hub": 1},
    "Moshi": {"region": "PCMC", "base_rate": 5400, "rate_std": 600, "tier": "PCMC Affordable", "lat": 18.6756, "lon": 73.8439, "pincode": 412105, "metro": 0, "it_hub": 0},
    "Nigdi": {"region": "PCMC", "base_rate": 7200, "rate_std": 750, "tier": "PCMC Established", "lat": 18.6548, "lon": 73.7735, "pincode": 411044, "metro": 1, "it_hub": 0},
    "Akurdi": {"region": "PCMC", "base_rate": 6900, "rate_std": 700, "tier": "PCMC Established", "lat": 18.6492, "lon": 73.7801, "pincode": 411035, "metro": 1, "it_hub": 0},
    "Chikhali": {"region": "PCMC", "base_rate": 5100, "rate_std": 550, "tier": "PCMC Affordable", "lat": 18.6835, "lon": 73.8160, "pincode": 411062, "metro": 0, "it_hub": 0},
    "Bhosari": {"region": "PCMC", "base_rate": 5600, "rate_std": 600, "tier": "PCMC Industrial Corridor", "lat": 18.6260, "lon": 73.8475, "pincode": 411039, "metro": 1, "it_hub": 0}
}

def generate_locality_metadata_df() -> pd.DataFrame:
    """Generate locality metadata reference table."""
    records = []
    for loc, info in LOCALITY_PROFILES.items():
        records.append({
            "locality": loc,
            "region": info["region"],
            "tier": info["tier"],
            "base_rate_sqft": info["base_rate"],
            "latitude": info["lat"],
            "longitude": info["lon"],
            "pincode": info["pincode"],
            "metro_connectivity": bool(info["metro"]),
            "it_hub_proximity": bool(info["it_hub"]),
            "is_synthetic": True,
            "data_source": "PropSight Locality Directory (Pune/PCMC)"
        })
    return pd.DataFrame(records)

def generate_historical_trends_df() -> pd.DataFrame:
    """
    Generate quarterly locality price indices from 2018-Q1 to 2024-Q4 (28 quarters).
    Captures macroeconomic cycle:
    - 2018-2019: Steady expansion (~6-7% p.a.)
    - 2020 Q2-Q3: COVID-19 pandemic dampening (-2% to -4% quarterly)
    - 2021-2024: Robust post-pandemic surge in IT corridors (~8-12% p.a.)
    """
    np.random.seed(RANDOM_SEED)
    quarters = []
    for yr in range(2018, 2025):
        for q in range(1, 5):
            quarters.append(f"{yr}-Q{q}")

    rows = []
    for loc, info in LOCALITY_PROFILES.items():
        current_rate = float(info["base_rate"])
        # Backtrack rate to 2018-Q1
        # Over 28 quarters (~7 years), typical overall growth has been ~40-60%
        overall_discount = 0.65 if info["it_hub"] else 0.70
        rate = current_rate * overall_discount
        
        for i, q in enumerate(quarters):
            year = int(q.split("-")[0])
            quarter_num = int(q.split("-Q")[1])
            
            # Quarterly macro growth factor
            if year in [2018, 2019]:
                growth_rate = np.random.normal(0.015, 0.005)
            elif year == 2020 and quarter_num in [2, 3]:
                # COVID lockdowns
                growth_rate = np.random.normal(-0.025, 0.008)
            elif year == 2020 and quarter_num == 4:
                growth_rate = np.random.normal(0.010, 0.005)
            elif year in [2021, 2022]:
                growth_rate = np.random.normal(0.022 if info["it_hub"] else 0.016, 0.006)
            else: # 2023, 2024
                growth_rate = np.random.normal(0.020 if info["it_hub"] else 0.015, 0.005)
                
            rate = rate * (1.0 + growth_rate)
            transaction_volume = int(np.random.normal(85 if info["tier"] == "Prime West" else 60, 15))
            transaction_volume = max(15, transaction_volume)
            
            rows.append({
                "locality": loc,
                "region": info["region"],
                "quarter": q,
                "year": year,
                "quarter_num": quarter_num,
                "avg_price_per_sqft": round(rate, 1),
                "median_price_per_sqft": round(rate * np.random.uniform(0.97, 1.01), 1),
                "quarterly_growth_pct": round(growth_rate * 100, 2),
                "sample_transaction_volume": transaction_volume,
                "is_synthetic": True,
                "data_source": "PropSight Historical Trend Series (2018-2024)"
            })
            
    return pd.DataFrame(rows)

def generate_properties_df(n_samples: int = 3500) -> pd.DataFrame:
    """
    Generate synthetic property records with realistic relationships between
    BHK, carpet area, floor, age, amenities, locality rates, and total price.
    """
    np.random.seed(RANDOM_SEED)
    
    localities = list(LOCALITY_PROFILES.keys())
    # Sampling weights: higher in active development hubs (Wakad, Hinjewadi, Kharadi, Baner, Ravet)
    loc_weights = []
    for loc in localities:
        if loc in ["Wakad", "Hinjewadi", "Kharadi", "Baner", "Ravet", "Pimple Saudagar", "Kothrud"]:
            loc_weights.append(3.0)
        elif loc in ["Balewadi", "Hadapsar", "Punawale", "Tathawade", "Viman Nagar", "Bavdhan"]:
            loc_weights.append(2.0)
        else:
            loc_weights.append(1.0)
    loc_weights = np.array(loc_weights) / sum(loc_weights)
    
    selected_localities = np.random.choice(localities, size=n_samples, p=loc_weights)
    
    # BHK distribution: 1 BHK: 15%, 2 BHK: 48%, 3 BHK: 28%, 4 BHK: 7%, 5 BHK: 2%
    bhk_options = [1, 2, 3, 4, 5]
    bhk_probs = [0.15, 0.48, 0.28, 0.07, 0.02]
    selected_bhks = np.random.choice(bhk_options, size=n_samples, p=bhk_probs)
    
    property_types = ["Apartment", "Independent House / Villa", "Penthouse", "Builder Floor"]
    type_probs = [0.84, 0.08, 0.04, 0.04]
    selected_types = np.random.choice(property_types, size=n_samples, p=type_probs)
    
    furnishings = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
    furnish_probs = [0.30, 0.52, 0.18]
    selected_furnishings = np.random.choice(furnishings, size=n_samples, p=furnish_probs)
    
    possessions = ["Ready to Move", "Under Construction"]
    possession_probs = [0.72, 0.28]
    selected_possessions = np.random.choice(possessions, size=n_samples, p=possession_probs)
    
    records = []
    start_date = datetime(2021, 1, 1)
    
    for i in range(n_samples):
        loc = selected_localities[i]
        bhk = int(selected_bhks[i])
        ptype = selected_types[i]
        furnishing = selected_furnishings[i]
        possession = selected_possessions[i]
        profile = LOCALITY_PROFILES[loc]
        region = profile["region"]
        
        # Carpet area based on BHK
        if bhk == 1:
            carpet = np.random.normal(520, 50)
        elif bhk == 2:
            carpet = np.random.normal(850, 90)
        elif bhk == 3:
            carpet = np.random.normal(1350, 150)
        elif bhk == 4:
            carpet = np.random.normal(2200, 250)
        else: # 5 BHK
            carpet = np.random.normal(3400, 350)
        
        # Villa or Penthouse bonus area
        if ptype in ["Independent House / Villa", "Penthouse"]:
            carpet *= np.random.uniform(1.2, 1.45)
            
        carpet = max(350.0, min(6500.0, round(float(carpet), 1)))
        
        # Built-up area (typically 1.25 to 1.35x carpet area in Pune)
        loading_factor = np.random.uniform(1.24, 1.35)
        built_up = round(carpet * loading_factor, 1)
        
        # Floors
        if ptype == "Independent House / Villa":
            total_floors = np.random.choice([2, 3, 4], p=[0.6, 0.35, 0.05])
            floor = np.random.choice(range(1, total_floors + 1))
        else:
            total_floors = int(np.random.choice([7, 11, 14, 18, 22, 28, 32], p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.07, 0.03]))
            floor = int(np.random.randint(1, total_floors + 1))
            
        # Bathrooms & Balconies
        bathrooms = int(max(1, min(bhk + np.random.choice([0, 1], p=[0.75, 0.25]), 6)))
        balconies = int(max(1, min(bhk + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]), 4)))
        
        # Parking spaces
        if bhk == 1:
            parking = int(np.random.choice([0, 1], p=[0.3, 0.7]))
        elif bhk == 2:
            parking = int(np.random.choice([1, 2], p=[0.85, 0.15]))
        else:
            parking = int(np.random.choice([1, 2, 3], p=[0.25, 0.65, 0.10]))
            
        # Property Age
        if possession == "Under Construction":
            age = 0.0
        else:
            age = float(round(np.random.exponential(4.5), 1))
            age = min(age, 30.0)
            
        # Amenities count (0 to 11)
        base_amenities = 6 if profile["tier"] in ["Prime West", "Premium", "Integrated Township"] else 4
        amenities_count = int(np.clip(np.random.normal(base_amenities, 2.5), 1, 11))
        
        # Price calculation modeling realistic market factors
        base_loc_rate = profile["base_rate"] + np.random.normal(0, profile["rate_std"] * 0.4)
        
        # Floor rise adjustment (higher floors in high-rises often command a premium in Pune)
        floor_rise_pct = 0.0
        if total_floors > 10:
            floor_rise_pct = (floor / total_floors) * 0.06 # up to +6%
            
        # Age depreciation: ~1.2% per year up to 20 years
        age_discount_pct = -min(age * 0.012, 0.25)
        
        # Furnishing premium
        furnish_premium_pct = 0.0
        if furnishing == "Semi-Furnished":
            furnish_premium_pct = 0.04
        elif furnishing == "Fully Furnished":
            furnish_premium_pct = 0.10
            
        # Amenities premium: ~0.8% per amenity above 4
        amenities_premium_pct = (amenities_count - 4) * 0.008
        
        # Property type multiplier
        type_multiplier = 1.0
        if ptype == "Penthouse":
            type_multiplier = 1.15
        elif ptype == "Independent House / Villa":
            type_multiplier = 1.22
            
        # Effective rate per sq.ft.
        effective_rate = base_loc_rate * (
            1.0 + floor_rise_pct + age_discount_pct + furnish_premium_pct + amenities_premium_pct
        ) * type_multiplier
        
        # Add realistic market noise (unobserved features, views, finish quality)
        effective_rate *= np.random.normal(1.0, 0.04)
        effective_rate = max(3800.0, effective_rate)
        
        # Total Price in Lakhs = (carpet_area * effective_rate) / 100,000
        # In Indian real estate, buyers pay on builtup/saleable or all-inclusive carpet
        total_price_inr = carpet * effective_rate
        price_lakhs = round(total_price_inr / 100000.0, 2)
        price_per_sqft = round((price_lakhs * 100000.0) / carpet, 1)
        
        # Random listing date between 2022-01-01 and 2024-12-31
        random_days = np.random.randint(0, 1095)
        listing_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        records.append({
            "property_id": f"PUN-{100000 + i}",
            "locality": loc,
            "region": region,
            "property_type": ptype,
            "bhk": bhk,
            "carpet_area_sqft": carpet,
            "built_up_area_sqft": built_up,
            "floor": floor,
            "total_floors": total_floors,
            "property_age_years": age,
            "bathrooms": bathrooms,
            "balconies": balconies,
            "parking_spaces": parking,
            "furnishing": furnishing,
            "possession_status": possession,
            "amenities_count": amenities_count,
            "price_lakhs": price_lakhs,
            "price_per_sqft": price_per_sqft,
            "listing_date": listing_date,
            "target_type": "listing_price",
            "is_synthetic": True,
            "data_source": "PropSight Demonstration Dataset (Calibrated to Pune/PCMC Market Reports 2018-2024)"
        })
        
    return pd.DataFrame(records)

def ensure_raw_datasets_exist(force_regenerate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Check if raw datasets exist. If not, generate and save them.
    Returns:
        (properties_df, historical_trends_df, localities_df)
    """
    needs_gen = force_regenerate or not (
        RAW_PROPERTIES_PATH.exists() and
        RAW_HISTORICAL_PATH.exists() and
        LOCALITY_METADATA_PATH.exists()
    )
    
    if needs_gen:
        RAW_PROPERTIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        localities_df = generate_locality_metadata_df()
        localities_df.to_csv(LOCALITY_METADATA_PATH, index=False)
        
        historical_df = generate_historical_trends_df()
        historical_df.to_csv(RAW_HISTORICAL_PATH, index=False)
        
        properties_df = generate_properties_df(n_samples=3600)
        properties_df.to_csv(RAW_PROPERTIES_PATH, index=False)
        
        return properties_df, historical_df, localities_df
    else:
        properties_df = pd.read_csv(RAW_PROPERTIES_PATH)
        historical_df = pd.read_csv(RAW_HISTORICAL_PATH)
        localities_df = pd.read_csv(LOCALITY_METADATA_PATH)
        return properties_df, historical_df, localities_df

if __name__ == "__main__":
    p_df, h_df, l_df = ensure_raw_datasets_exist(force_regenerate=True)
    print(f"Generated {len(p_df)} properties, {len(h_df)} historical records, {len(l_df)} localities.")
