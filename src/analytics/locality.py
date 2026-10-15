"""
Locality Intelligence and Market Analytics Service for PropSight.
Aggregates micro-market KPIs, distributions, historical CAGR, and comparative statistics.
Enforces sample-size reliability warnings.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

from src.data.loader import DataLoader

class LocalityAnalytics:
    def __init__(self):
        self.df_props = DataLoader.load_properties(cleaned=True)
        self.df_hist = DataLoader.load_historical_trends()
        self.df_meta = DataLoader.load_locality_metadata()

    def get_available_localities(self) -> List[str]:
        """Return sorted list of supported localities."""
        return sorted(self.df_props["locality"].unique().tolist())

    def get_locality_kpis(self, locality: str) -> Dict[str, Any]:
        """Compute key real estate metrics for a specific micro-market."""
        loc_df = self.df_props[self.df_props["locality"] == locality]
        hist_df = self.df_hist[self.df_hist["locality"] == locality]
        meta_df = self.df_meta[self.df_meta["locality"] == locality]

        sample_size = len(loc_df)
        is_reliable = sample_size >= 25
        reliability_warning = None if is_reliable else f"Low sample size ({sample_size} listings). Metrics may have higher variance."

        if sample_size == 0:
            return {
                "locality": locality,
                "sample_size": 0,
                "is_reliable": False,
                "warning": "No records found for this locality."
            }

        avg_price = float(loc_df["price_lakhs"].mean())
        median_price = float(loc_df["price_lakhs"].median())
        avg_rate_sqft = float(loc_df["price_per_sqft"].mean())
        median_rate_sqft = float(loc_df["price_per_sqft"].median())
        min_price = float(loc_df["price_lakhs"].min())
        max_price = float(loc_df["price_lakhs"].max())

        # Historical growth calculation
        hist_cagr = None
        if len(hist_df) >= 8:
            first_rate = hist_df.sort_values(by="quarter").iloc[0]["avg_price_per_sqft"]
            last_rate = hist_df.sort_values(by="quarter").iloc[-1]["avg_price_per_sqft"]
            num_years = len(hist_df) / 4.0
            if first_rate > 0 and num_years > 0:
                hist_cagr = round(((last_rate / first_rate) ** (1.0 / num_years) - 1.0) * 100.0, 2)

        # Region & Tier
        region = loc_df["region"].iloc[0] if "region" in loc_df.columns else "Pune"
        tier = meta_df["tier"].iloc[0] if len(meta_df) > 0 and "tier" in meta_df.columns else "Mid-Market"

        return {
            "locality": locality,
            "region": region,
            "tier": tier,
            "sample_size": sample_size,
            "is_reliable": is_reliable,
            "warning": reliability_warning,
            "avg_price_lakhs": round(avg_price, 2),
            "median_price_lakhs": round(median_price, 2),
            "avg_rate_per_sqft": round(avg_rate_sqft, 1),
            "median_rate_per_sqft": round(median_rate_sqft, 1),
            "price_range_lakhs": (round(min_price, 1), round(max_price, 1)),
            "historical_cagr_pct": hist_cagr,
            "bhk_distribution": loc_df["bhk"].value_counts().to_dict(),
            "property_type_distribution": loc_df["property_type"].value_counts().to_dict()
        }

    def compare_localities(self, localities: List[str]) -> pd.DataFrame:
        """Compare multiple localities side-by-side in a structured summary table."""
        records = []
        for loc in localities:
            kpi = self.get_locality_kpis(loc)
            if kpi.get("sample_size", 0) > 0:
                records.append({
                    "Locality": loc,
                    "Region": kpi["region"],
                    "Market Tier": kpi["tier"],
                    "Listings Count": kpi["sample_size"],
                    "Avg Price (₹ Lakhs)": kpi["avg_price_lakhs"],
                    "Median Price (₹ Lakhs)": kpi["median_price_lakhs"],
                    "Avg Rate (₹/sq.ft.)": kpi["avg_rate_per_sqft"],
                    "Median Rate (₹/sq.ft.)": kpi["median_rate_per_sqft"],
                    "7-Year Historical CAGR": f"{kpi['historical_cagr_pct']:.1f}%" if kpi['historical_cagr_pct'] else "N/A",
                    "Sample Reliability": "Sufficient" if kpi["is_reliable"] else "Limited Data"
                })
        return pd.DataFrame(records)

    def plot_price_distribution(self, locality: str) -> go.Figure:
        """Plot price distribution histogram with median marker."""
        loc_df = self.df_props[self.df_props["locality"] == locality]
        median_val = loc_df["price_lakhs"].median()

        fig = px.histogram(
            loc_df,
            x="price_lakhs",
            nbins=25,
            title=f"Price Distribution in {locality} (₹ Lakhs)",
            labels={"price_lakhs": "Listing Price (₹ Lakhs)"},
            color_discrete_sequence=["#0EA5E9"]
        )
        fig.add_vline(
            x=median_val,
            line_dash="dash",
            line_color="#0F172A",
            annotation_text=f"Median: ₹{median_val:.1f}L",
            annotation_position="top right"
        )
        fig.update_layout(
            plot_bgcolor="#FFFFFF",
            height=340,
            margin=dict(l=20, r=20, t=40, b=30),
            yaxis_title="Count"
        )
        return fig

    def plot_historical_series(self, locality: str) -> go.Figure:
        """Plot quarterly average price per sq.ft. trend from 2018 to 2024."""
        hist_df = self.df_hist[self.df_hist["locality"] == locality].sort_values(by=["year", "quarter_num"])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["quarter"],
            y=hist_df["avg_price_per_sqft"],
            mode="lines+markers",
            name="Avg Rate (₹/sq.ft.)",
            line=dict(color="#0284C7", width=3),
            marker=dict(size=6, color="#0F172A")
        ))

        fig.update_layout(
            title=f"Historical Price per Sq.Ft. Trend — {locality} (2018-2024)",
            xaxis_title="Quarter",
            yaxis_title="Average Rate (₹/sq.ft.)",
            plot_bgcolor="#FFFFFF",
            height=360,
            margin=dict(l=20, r=20, t=40, b=40)
        )
        return fig

if __name__ == "__main__":
    analytics = LocalityAnalytics()
    kpis = analytics.get_locality_kpis("Baner")
    print(f"Locality: {kpis['locality']} ({kpis['region']})")
    print(f"Avg Price: INR {kpis['avg_price_lakhs']}L | Avg Rate: INR {kpis['avg_rate_per_sqft']}/sqft")
    print(f"Sample Size: {kpis['sample_size']} | CAGR: {kpis['historical_cagr_pct']}%")
    
    comp = analytics.compare_localities(["Baner", "Wakad", "Kharadi", "Kothrud", "Ravet"])
    print(comp[["Locality", "Region", "Sample Reliability"]])
