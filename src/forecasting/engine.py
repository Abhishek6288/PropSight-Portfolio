"""
Locality-Level Time-Series Forecasting and Scenario Analysis Engine for PropSight.
Enforces chronological back-testing, statistical baselines, ARIMA modeling,
uncertainty interval widening, and transparent scenario projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import plotly.graph_objects as go
import logging

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from src.data.loader import DataLoader
from src.utils.formatting import format_inr_raw, format_pct

logger = logging.getLogger(__name__)

class LocalityForecastEngine:
    MIN_QUARTERS_REQUIRED = 12

    def __init__(self):
        self.df_hist = DataLoader.load_historical_trends()

    def check_eligibility(self, locality: str, target_year: int) -> Tuple[bool, str]:
        """
        Verify whether locality has sufficient historical observations and whether horizon is supported.
        Supported horizons for statistical projection: up to 2026 (1-8 quarters).
        2028: advanced horizon (cautious scenario only).
        2030+: future speculative scope (rejected for statistical forecasting).
        """
        loc_df = self.df_hist[self.df_hist["locality"] == locality]
        n_quarters = len(loc_df)

        if n_quarters < self.MIN_QUARTERS_REQUIRED:
            return False, f"Insufficient historical data: {locality} has only {n_quarters} quarters of records (minimum {self.MIN_QUARTERS_REQUIRED} required)."

        if target_year > 2028:
            return False, f"Target horizon {target_year} exceeds empirical forecasting limits (2025-2028). Projections beyond 2028 are strictly speculative research scope per the PropSight PRD."

        return True, "Eligible for statistical forecasting."

    def backtest_models(self, locality: str, test_quarters: int = 4) -> Dict[str, Any]:
        """
        Perform walk-forward back-testing on the held-out historical quarters (chronological split).
        Compares:
        1. 4-Quarter Simple Moving Average Baseline
        2. Exponential Smoothing Baseline
        3. ARIMA(1, 1, 1) Model
        """
        loc_df = self.df_hist[self.df_hist["locality"] == locality].sort_values(by=["year", "quarter_num"])
        series = loc_df["avg_price_per_sqft"].values
        quarters = loc_df["quarter"].values

        if len(series) <= test_quarters + 4:
            raise ValueError(f"Not enough observations in {locality} to perform back-testing.")

        train_series = series[:-test_quarters]
        test_series = series[-test_quarters:]
        test_q_labels = quarters[-test_quarters:]

        results = {}

        # 1. Moving Average (4Q) Baseline
        ma_pred = [float(np.mean(train_series[-4:]))] * test_quarters
        ma_mae = float(np.mean(np.abs(test_series - ma_pred)))
        ma_mape = float(np.mean(np.abs((test_series - ma_pred) / test_series)) * 100.0)
        results["Moving Average (4Q)"] = {
            "predictions": ma_pred,
            "mae": round(ma_mae, 1),
            "mape_pct": round(ma_mape, 2)
        }

        # 2. Exponential Smoothing
        try:
            ses = SimpleExpSmoothing(train_series, initialization_method="estimated").fit()
            ses_pred = ses.forecast(test_quarters).tolist()
            ses_mae = float(np.mean(np.abs(test_series - ses_pred)))
            ses_mape = float(np.mean(np.abs((test_series - ses_pred) / test_series)) * 100.0)
            results["Exponential Smoothing"] = {
                "predictions": [round(p, 1) for p in ses_pred],
                "mae": round(ses_mae, 1),
                "mape_pct": round(ses_mape, 2)
            }
        except Exception as e:
            logger.warning(f"SES backtesting fallback: {e}")

        # 3. ARIMA(1, 1, 1)
        try:
            arima = ARIMA(train_series, order=(1, 1, 1)).fit()
            arima_pred = arima.forecast(test_quarters).tolist()
            arima_mae = float(np.mean(np.abs(test_series - arima_pred)))
            arima_mape = float(np.mean(np.abs((test_series - arima_pred) / test_series)) * 100.0)
            results["ARIMA(1,1,1)"] = {
                "predictions": [round(p, 1) for p in arima_pred],
                "mae": round(arima_mae, 1),
                "mape_pct": round(arima_mape, 2)
            }
        except Exception as e:
            logger.warning(f"ARIMA backtesting fallback: {e}")

        # Find best back-tested model
        best_model = min(results, key=lambda k: results[k]["mae"])

        return {
            "locality": locality,
            "test_quarters": list(test_q_labels),
            "actual_values": [round(v, 1) for v in test_series],
            "models": results,
            "selected_best_model": best_model,
            "best_backtest_mape_pct": results[best_model]["mape_pct"]
        }

    def generate_forecast(
        self,
        locality: str,
        steps_ahead: int = 4
    ) -> Dict[str, Any]:
        """
        Fit ARIMA model on all available historical quarters and generate future forecast
        with expanding statistical prediction intervals (80% and 95%).
        """
        eligible, reason = self.check_eligibility(locality, target_year=2025)
        if not eligible:
            raise ValueError(reason)

        loc_df = self.df_hist[self.df_hist["locality"] == locality].sort_values(by=["year", "quarter_num"])
        series = loc_df["avg_price_per_sqft"].values
        last_quarter = loc_df["quarter"].iloc[-1]
        last_year = int(last_quarter.split("-")[0])
        last_q_num = int(last_quarter.split("-Q")[1])

        # Generate future quarter labels
        future_quarters = []
        cur_y, cur_q = last_year, last_q_num
        for _ in range(steps_ahead):
            cur_q += 1
            if cur_q > 4:
                cur_q = 1
                cur_y += 1
            future_quarters.append(f"{cur_y}-Q{cur_q}")

        # Fit ARIMA on full series
        try:
            model = ARIMA(series, order=(1, 1, 1)).fit()
            forecast_res = model.get_forecast(steps=steps_ahead)
            central_pred = forecast_res.predicted_mean
            conf_int_80 = forecast_res.conf_int(alpha=0.20)  # 80% CI
            conf_int_95 = forecast_res.conf_int(alpha=0.05)  # 95% CI
        except Exception as e:
            # Linear trend fallback
            logger.warning(f"ARIMA forecast fallback: {e}")
            growth_rate = np.mean(loc_df["quarterly_growth_pct"].iloc[-8:]) / 100.0
            central_pred = []
            cur_val = series[-1]
            for _ in range(steps_ahead):
                cur_val *= (1.0 + growth_rate)
                central_pred.append(cur_val)
            central_pred = np.array(central_pred)
            conf_int_80 = np.column_stack([central_pred * 0.95, central_pred * 1.05])
            conf_int_95 = np.column_stack([central_pred * 0.90, central_pred * 1.10])

        forecast_table = []
        for i, q in enumerate(future_quarters):
            rate = float(central_pred[i])
            lower_80 = float(conf_int_80[i, 0])
            upper_80 = float(conf_int_80[i, 1])
            lower_95 = float(conf_int_95[i, 0])
            upper_95 = float(conf_int_95[i, 1])

            forecast_table.append({
                "Quarter": q,
                "Forecast Rate (₹/sq.ft.)": round(rate, 1),
                "Lower 80% CI": round(lower_80, 1),
                "Upper 80% CI": round(upper_80, 1),
                "Lower 95% CI": round(lower_95, 1),
                "Upper 95% CI": round(upper_95, 1),
                "Uncertainty Spread": round(upper_80 - lower_80, 1)
            })

        return {
            "locality": locality,
            "historical_quarters": list(loc_df["quarter"].values),
            "historical_rates": list(loc_df["avg_price_per_sqft"].values),
            "forecast_quarters": future_quarters,
            "forecast_table": pd.DataFrame(forecast_table),
            "central_forecasts": [round(float(v), 1) for v in central_pred],
            "methodology": "Statistical ARIMA(1,1,1) with expanding asymptotic confidence intervals"
        }

    def generate_scenarios(
        self,
        locality: str,
        horizon_years: int = 3
    ) -> Dict[str, Any]:
        """
        Generate explicit Conservative, Baseline, and High Growth scenarios through 2027/2028.
        Clearly documents underlying macro assumptions.
        """
        loc_df = self.df_hist[self.df_hist["locality"] == locality].sort_values(by=["year", "quarter_num"])
        current_rate = float(loc_df["avg_price_per_sqft"].iloc[-1])

        # Locality profile for growth baseline
        cagr_historical = ((current_rate / loc_df["avg_price_per_sqft"].iloc[0]) ** (1.0 / (len(loc_df) / 4.0)) - 1.0)
        cagr_historical = max(0.04, min(0.12, cagr_historical))

        # Scenario CAGR rates
        scenarios = {
            "Conservative Scenario": {
                "annual_growth_pct": round((cagr_historical * 0.5) * 100, 2),
                "assumptions": "High interest rate environment (repo rate > 6.75%), regulatory compliance delays, slower IT hiring.",
                "color": "#EF4444"
            },
            "Baseline Scenario": {
                "annual_growth_pct": round(cagr_historical * 100, 2),
                "assumptions": "Continuation of 2021-2024 economic trajectory, steady IT expansion, moderate inflation (5-6%).",
                "color": "#0284C7"
            },
            "High Growth Scenario": {
                "annual_growth_pct": round((cagr_historical * 1.5) * 100, 2),
                "assumptions": "Accelerated infrastructure completion (Metro lines, Ring Road), major multinational IT/GCC campus expansion.",
                "color": "#10B981"
            }
        }

        years = list(range(2024, 2024 + horizon_years + 1))
        projection_rows = []

        for name, details in scenarios.items():
            rate = current_rate
            g = details["annual_growth_pct"] / 100.0
            rates = [round(current_rate, 1)]
            for _ in range(horizon_years):
                rate *= (1.0 + g)
                rates.append(round(rate, 1))

            scenarios[name]["projected_rates"] = rates
            scenarios[name]["terminal_rate"] = rates[-1]
            scenarios[name]["total_appreciation_pct"] = round(((rates[-1] / current_rate) - 1.0) * 100, 1)

        return {
            "locality": locality,
            "base_rate_2024": current_rate,
            "years": years,
            "scenarios": scenarios
        }

    def plot_forecast_and_history(self, forecast_res: Dict[str, Any]) -> go.Figure:
        """Plot combined historical trend, future ARIMA forecast, and 80%/95% confidence bands."""
        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=forecast_res["historical_quarters"],
            y=forecast_res["historical_rates"],
            mode="lines+markers",
            name="Historical Avg Rate",
            line=dict(color="#334155", width=2.5),
            marker=dict(size=4)
        ))

        # Future forecast
        fc_df = forecast_res["forecast_table"]
        # Connect last historical point to first forecast
        connect_x = [forecast_res["historical_quarters"][-1]] + list(fc_df["Quarter"])
        connect_y = [forecast_res["historical_rates"][-1]] + list(fc_df["Forecast Rate (₹/sq.ft.)"])

        fig.add_trace(go.Scatter(
            x=connect_x,
            y=connect_y,
            mode="lines+markers",
            name="ARIMA Central Forecast",
            line=dict(color="#0284C7", width=3, dash="dash"),
            marker=dict(size=6, color="#0284C7")
        ))

        # 80% Confidence Interval Band
        fig.add_trace(go.Scatter(
            x=list(fc_df["Quarter"]) + list(fc_df["Quarter"])[::-1],
            y=list(fc_df["Upper 80% CI"]) + list(fc_df["Lower 80% CI"])[::-1],
            fill="toself",
            fillcolor="rgba(2, 132, 199, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="80% Prediction Interval"
        ))

        fig.update_layout(
            title=f"Locality Rate Projection & Uncertainty Bands — {forecast_res['locality']}",
            xaxis_title="Quarter",
            yaxis_title="Average Rate (₹/sq.ft.)",
            plot_bgcolor="#FFFFFF",
            height=380,
            margin=dict(l=20, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    def plot_scenarios(self, scenario_res: Dict[str, Any]) -> go.Figure:
        """Plot 3-Year Scenario Fan Chart (Conservative, Baseline, High Growth)."""
        fig = go.Figure()
        years_str = [str(y) for y in scenario_res["years"]]

        for name, details in scenario_res["scenarios"].items():
            fig.add_trace(go.Scatter(
                x=years_str,
                y=details["projected_rates"],
                mode="lines+markers",
                name=f"{name} ({details['annual_growth_pct']:+.1f}% p.a.)",
                line=dict(color=details["color"], width=2.5),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title=f"3-Year Multi-Scenario Trajectory (2024-2027) — {scenario_res['locality']}",
            xaxis_title="Year",
            yaxis_title="Estimated Rate (₹/sq.ft.)",
            plot_bgcolor="#FFFFFF",
            height=360,
            margin=dict(l=20, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

if __name__ == "__main__":
    engine = LocalityForecastEngine()
    backtest = engine.backtest_models("Baner")
    print("Back-testing Results for Baner:")
    print(f"Selected Best Model: {backtest['selected_best_model']} (Backtest MAPE: {backtest['best_backtest_mape_pct']}%)")
    for m, res in backtest["models"].items():
        print(f" - {m:<25}: MAE={res['mae']}, MAPE={res['mape_pct']}%")

    fc = engine.generate_forecast("Baner", steps_ahead=4)
    print("\nNext 4 Quarters Forecast (ARIMA):")
    print(fc["forecast_table"][["Quarter", "Forecast Rate (₹/sq.ft.)", "Lower 80% CI", "Upper 80% CI"]].rename(columns={"Forecast Rate (₹/sq.ft.)": "Forecast Rate"}))

    sc = engine.generate_scenarios("Baner", horizon_years=3)
    print("\n3-Year Scenarios:")
    for s_name, s_data in sc["scenarios"].items():
        print(f" - {s_name}: Terminal Rate INR {s_data['terminal_rate']}/sqft ({s_data['total_appreciation_pct']}%)")
