"""
PropSight — Market Forecasts & Scenario Analysis
Time-series forecasting with statistical back-testing, ARIMA modeling,
expanding prediction intervals, and documented macroeconomic scenarios.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils.styles import apply_custom_css, render_header, render_kpi_card
from src.utils.formatting import format_rate_per_sqft, format_pct
from src.forecasting.engine import LocalityForecastEngine
from src.analytics.locality import LocalityAnalytics

st.set_page_config(page_title="Market Forecasts & Scenarios — PropSight", page_icon="📈", layout="wide")
apply_custom_css()

# Header
render_header(
    title="Market Forecasts & Scenario Projections",
    subtitle="ARIMA time-series projections, statistical back-testing, widening uncertainty intervals, and macroeconomic scenarios",
    page_tag="Forecasting Module"
)

@st.cache_resource
def get_forecast_engine():
    return LocalityForecastEngine()

@st.cache_resource
def get_analytics():
    return LocalityAnalytics()

forecast_engine = get_forecast_engine()
analytics = get_analytics()
localities = analytics.get_available_localities()

# Controls row
c_loc, c_steps, c_horizon = st.columns([1.5, 1.0, 1.0])

with c_loc:
    selected_loc = st.selectbox(
        "Select Micro-Market for Forecast:",
        options=localities,
        index=localities.index("Baner") if "Baner" in localities else 0
    )

with c_steps:
    forecast_steps = st.selectbox(
        "Projection Quarters Ahead:",
        options=[1, 2, 3, 4],
        index=3,
        format_func=lambda x: f"{x} Quarter{'s' if x > 1 else ''} ({(x/4)*12:.0f} Months)"
    )

with c_horizon:
    selected_horizon_year = st.selectbox(
        "Long-term Horizon Year:",
        options=[2025, 2026, 2027, 2028, 2030, 2035, 2045],
        index=0,
        help="PRD Scope Boundary: 2025-2028 supported. 2030+ is classified as future speculative scope."
    )

# Scope Boundary Check per PRD
eligible, reason = forecast_engine.check_eligibility(selected_loc, selected_horizon_year)

if not eligible:
    st.markdown(f"""
    <div class="callout-box callout-warning">
        ⛔ <strong>Forecasting Scope Boundary Enforced (PropSight PRD §5 & §8):</strong><br>
        {reason}<br><br>
        <em>Academic Note:</em> Econometric time-series extrapolations past 3-4 years compound unmodeled structural variance (policy shifts, interest rate regimes, infrastructural delays). PropSight intentionally restricts statistical time-series projections to near-term horizons to maintain scientific defensibility.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# 1. Walk-Forward Back-testing Card
st.markdown("### 🧪 Chronological Model Back-Testing (Held-out 4 Quarters)")
st.caption("Models were trained on 2018-Q1 to 2023-Q4 and validated against actual 2024 quarterly observations (no random splitting).")

with st.spinner("Executing walk-forward back-testing..."):
    backtest = forecast_engine.backtest_models(selected_loc, test_quarters=4)

b1, b2, b3 = st.columns(3)
with b1:
    render_kpi_card(
        label="Selected Model",
        value=backtest["selected_best_model"],
        caption=f"Lowest back-test error for {selected_loc}"
    )
with b2:
    render_kpi_card(
        label="Back-Test MAPE",
        value=f"{backtest['best_backtest_mape_pct']:.2f}%",
        caption="Mean Absolute % Error on 2024"
    )
with b3:
    best_mae = backtest["models"][backtest["selected_best_model"]]["mae"]
    render_kpi_card(
        label="Back-Test MAE",
        value=format_rate_per_sqft(best_mae),
        caption="Absolute deviation / sq.ft."
    )

# Back-testing comparison table
bt_rows = []
for model_name, m_res in backtest["models"].items():
    bt_rows.append({
        "Model": model_name,
        "Back-Test MAE (₹/sq.ft.)": m_res["mae"],
        "Back-Test MAPE (%)": f"{m_res['mape_pct']:.2f}%",
        "Evaluation": "Selected Estimator" if model_name == backtest["selected_best_model"] else "Baseline"
    })
st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Statistical ARIMA Forecast & Confidence Bands
st.markdown(f"### 📈 Statistical Rate Projections & Uncertainty Bands — {selected_loc}")

with st.spinner("Fitting ARIMA and computing expanding prediction intervals..."):
    fc_res = forecast_engine.generate_forecast(selected_loc, steps_ahead=forecast_steps)

col_chart, col_table = st.columns([1.4, 1.0])

with col_chart:
    fig_fc = forecast_engine.plot_forecast_and_history(fc_res)
    st.plotly_chart(fig_fc, use_container_width=True)

with col_table:
    st.markdown("#### Projected Quarterly Rates")
    fc_table_disp = fc_res["forecast_table"][[
        "Quarter", "Forecast Rate (₹/sq.ft.)", "Lower 80% CI", "Upper 80% CI", "Uncertainty Spread"
    ]].copy()
    st.dataframe(fc_table_disp, use_container_width=True, hide_index=True)
    
    st.caption(
        "Notice: The uncertainty spread widens monotonically over future horizons, "
        "faithfully modeling increasing epistemic uncertainty as required by the PRD."
    )

st.markdown("<br>", unsafe_allow_html=True)

# 3. Macroeconomic Scenario Analysis
st.markdown(f"### 🌐 3-Year Macroeconomic Scenario Analysis (2024–2027) — {selected_loc}")
st.caption("Assumption-driven scenarios clearly separated from statistical model extrapolations.")

scenario_res = forecast_engine.generate_scenarios(selected_loc, horizon_years=3)

fig_sc = forecast_engine.plot_scenarios(scenario_res)
st.plotly_chart(fig_sc, use_container_width=True)

# Scenario Assumptions and Metrics Table
sc_cards = st.columns(3)
for i, (s_name, s_info) in enumerate(scenario_res["scenarios"].items()):
    with sc_cards[i]:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 4px solid {s_info['color']};">
            <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">{s_name}</h4>
            <div style="font-size: 1.25rem; font-weight: 700; color: {s_info['color']}; margin-bottom: 0.25rem;">
                ₹{s_info['terminal_rate']:,.0f}/sq.ft.
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 0.75rem;">
                Growth: <strong>{s_info['annual_growth_pct']:+.1f}% p.a.</strong> ({s_info['total_appreciation_pct']:+.1f}% total)
            </div>
            <div style="font-size: 0.82rem; color: #334155; line-height: 1.4;">
                <strong>Assumptions:</strong><br>{s_info['assumptions']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Educational Limitations Banner
st.markdown("---")
st.markdown("""
<div class="callout-box">
    <strong>Forecasting Governance & Principles:</strong><br>
    • <strong>Chronological Separation:</strong> Time-series models never use random cross-validation, strictly preventing future information leakage.<br>
    • <strong>Uncertainty Communication:</strong> Forecasts present expanding confidence intervals rather than single-point deterministic guarantees.<br>
    • <strong>Decision Support:</strong> Projections do not constitute guaranteed investment returns or certified appreciation forecasts.
</div>
""", unsafe_allow_html=True)
