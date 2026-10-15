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
            color_discrete_sequence=["#0D9488"],
            opacity=0.75,
            labels={"price_lakhs": "Listing Price (₹ Lakhs)"}
        )

        q25 = float(loc_df["price_lakhs"].quantile(0.25))
        q75 = float(loc_df["price_lakhs"].quantile(0.75))

        fig.add_vline(
            x=median_val,
            line_dash="dash",
            line_color="#0F172A",
            line_width=2,
            annotation_text=f"Median: ₹{median_val:.1f}L",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#0F172A", weight="bold")
        )
        fig.add_vrect(
            x0=q25, x1=q75,
            fillcolor="rgba(13, 148, 136, 0.12)",
            line_width=0,
            annotation_text="IQR Core Band",
            annotation_position="top left",
            annotation_font=dict(size=9.5, color="#0D9488")
        )

        from src.utils.charts import apply_chart_theme
        apply_chart_theme(
            fig,
            title=f"Price Distribution in {locality} (₹ Lakhs)",
            xaxis_title="Listing Price (₹ Lakhs)",
            yaxis_title="Property Count",
            height=350
        )
        return fig

    def plot_historical_series(self, locality: str, show_moving_avg: bool = True) -> go.Figure:
        """Plot quarterly average price per sq.ft. trend with moving average and range slider."""
        hist_df = self.df_hist[self.df_hist["locality"] == locality].sort_values(by=["year", "quarter_num"])
        
        fig = go.Figure()

        # Primary rate line
        fig.add_trace(go.Scatter(
            x=hist_df["quarter"],
            y=hist_df["avg_price_per_sqft"],
            mode="lines+markers",
            name="Quarterly Rate",
            line=dict(color="#0D9488", width=2.5),
            marker=dict(size=6, color="#0F172A"),
            hovertemplate="Quarter: %{x}<br>Average Rate: ₹%{y:,.1f}/sq.ft.<extra></extra>"
        ))

        # 4-Quarter Rolling Average
        if show_moving_avg and len(hist_df) >= 4:
            ma = hist_df["avg_price_per_sqft"].rolling(window=4, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=hist_df["quarter"],
                y=ma,
                mode="lines",
                name="4Q Rolling Trend",
                line=dict(color="#0284C7", width=1.5, dash="dot"),
                hovertemplate="Rolling 4Q Trend: ₹%{y:,.1f}/sq.ft.<extra></extra>"
            ))

        from src.utils.charts import apply_chart_theme
        apply_chart_theme(
            fig,
            title=f"Historical Price per Sq.Ft. Evolution — {locality} (2018-2024)",
            xaxis_title="Timeline",
            yaxis_title="Rate (₹/sq.ft.)",
            height=370,
            hovermode="x unified"
        )

        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.06, bgcolor="#F8FAFC"),
            rangeselector=dict(
                buttons=list([
                    dict(count=4, label="1Y", step="all", stepmode="backward"),
                    dict(count=8, label="2Y", step="all", stepmode="backward"),
                    dict(count=16, label="4Y", step="all", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                font=dict(size=10, color="#1E293B"),
                bgcolor="#FFFFFF",
                activecolor="#0D9488"
            )
        )
        return fig

    def plot_locality_scatter(self, locality: str) -> go.Figure:
        """Plot price vs area scatter for the specific micro-market."""
        loc_df = self.df_props[self.df_props["locality"] == locality]
        from src.utils.charts import build_price_vs_area_scatter
        return build_price_vs_area_scatter(
            loc_df,
            x_col="carpet_area_sqft",
            y_col="price_lakhs",
            color_col="bhk",
            title=f"Listing Landscape in {locality}: Price vs. Carpet Area",
            sample_size=500
        )

if __name__ == "__main__":
    analytics = LocalityAnalytics()
    kpis = analytics.get_locality_kpis("Baner")
    print(f"Locality: {kpis['locality']} ({kpis['region']})")
    print(f"Avg Price: INR {kpis['avg_price_lakhs']}L | Avg Rate: INR {kpis['avg_rate_per_sqft']}/sqft")
    print(f"Sample Size: {kpis['sample_size']} | CAGR: {kpis['historical_cagr_pct']}%")
    
    comp = analytics.compare_localities(["Baner", "Wakad", "Kharadi", "Kothrud", "Ravet"])
    print(comp[["Locality", "Region", "Sample Reliability"]])
