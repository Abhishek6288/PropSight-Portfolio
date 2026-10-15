"""
PropSight — Exploratory Data Analysis (EDA) and Model Experimentation Script.
Runs comprehensive statistical checks, correlation analyses, and model comparisons.
Can be executed as a standalone Python script or imported into Jupyter.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import numpy as np
import json

from src.data.loader import DataLoader
from src.data.cleaner import PropertyDataCleaner
from src.models.trainer import ModelTrainer, MODEL_METADATA_PATH

def run_eda():
    print("=" * 60)
    print("PROPSIGHT: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    df = DataLoader.load_properties(cleaned=True)
    df_hist = DataLoader.load_historical_trends()

    print(f"\n1. DATASET OVERVIEW:")
    print(f" - Total Validated Records: {len(df)}")
    print(f" - Localities Tracked: {df['locality'].nunique()}")
    print(f" - Regional Breakdown: {df['region'].value_counts().to_dict()}")

    print(f"\n2. TARGET DISTRIBUTION (Price in INR Lakhs):")
    p = df["price_lakhs"]
    print(f" - Mean Price: INR {p.mean():.2f} Lakhs")
    print(f" - Median Price: INR {p.median():.2f} Lakhs")
    print(f" - Std Deviation: INR {p.std():.2f} Lakhs")
    print(f" - Min / Max: INR {p.min():.2f}L / INR {p.max():.2f}L")

    print(f"\n3. RATE PER SQUARE FOOT DISTRIBUTION:")
    r = df["price_per_sqft"]
    print(f" - Mean Rate: INR {r.mean():.1f}/sq.ft.")
    print(f" - Median Rate: INR {r.median():.1f}/sq.ft.")
    print(f" - IQR (25% - 75%): INR {r.quantile(0.25):.1f} - {r.quantile(0.75):.1f}/sq.ft.")

    print(f"\n4. TOP 5 PREMIUM LOCALITIES (by Avg Rate):")
    top_rates = df.groupby("locality")["price_per_sqft"].mean().sort_values(ascending=False).head(5)
    for loc, val in top_rates.items():
        print(f" - {loc:<20}: INR {val:.1f}/sq.ft.")

    print(f"\n5. TOP 5 AFFORDABLE LOCALITIES (by Avg Rate):")
    low_rates = df.groupby("locality")["price_per_sqft"].mean().sort_values().head(5)
    for loc, val in low_rates.items():
        print(f" - {loc:<20}: INR {val:.1f}/sq.ft.")

    print(f"\n6. HISTORICAL TIME SERIES OVERVIEW:")
    print(f" - Quarters Tracked: {df_hist['quarter'].nunique()} ({df_hist['quarter'].min()} to {df_hist['quarter'].max()})")
    print(f" - Historical Average Quarterly Growth: {df_hist['quarterly_growth_pct'].mean():.2f}%")

def run_model_experiment_summary():
    print("\n" + "=" * 60)
    print("PROPSIGHT: MODEL BENCHMARK TOURNAMENT SUMMARY")
    print("=" * 60)

    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            meta = json.load(f)
        
        print(f"\nSelected Best Architecture: {meta['selected_model']}")
        print(f"Test Evaluation Metrics:")
        for k, v in meta["best_model_metrics"].items():
            print(f" - {k:<20}: {v:.4f}" if isinstance(v, float) else f" - {k:<20}: {v}")

        print("\nCandidate Comparison Scorecard:")
        print(f"{'Model':<22} | {'Test R2':<8} | {'Test MAE (L)':<12} | {'Test MAPE (%)':<12}")
        print("-" * 62)
        for name, scores in meta["candidate_comparison"].items():
            print(f"{name:<22} | {scores['test_r2']:<8.4f} | {scores['test_mae_lakhs']:<12.2f} | {scores['test_mape_pct']:<12.2f}")
    else:
        print("Model metadata not found. Training models now...")
        trainer = ModelTrainer()
        trainer.train_and_evaluate()

if __name__ == "__main__":
    run_eda()
    run_model_experiment_summary()
