"""
PropSight — Centralized Chart Design System & Visualization Engine.
Provides consistent fintech-grade aesthetic themes, custom hover templates,
responsive layouts, and interactive Plotly visual builders across all pages.
"""

from typing import Dict, Any, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Semantically restrained palette adhering to PropSight visual identity
PROPSIGHT_COLORS = {
    "primary_teal": "#0D9488",      # Muted Teal (Primary Data Accent)
    "teal_light": "#14B8A6",
    "teal_dark": "#0F766E",
    "navy_primary": "#0F172A",     # Deep Slate / Navy
    "slate_dark": "#1E293B",
    "slate_muted": "#64748B",
    "slate_light": "#94A3B8",
    "blue_accent": "#0284C7",      # Corporate Blue
    "sky_accent": "#38BDF8",
    "indigo_accent": "#6366F1",    # Tech Accent
    "amber_accent": "#F59E0B",     # Warning / Mid-tier
    "rose_accent": "#E11D48",      # Danger / Overpriced
    "emerald_accent": "#10B981",   # Growth / Undervalued
    "card_bg": "#FFFFFF",
    "plot_bg": "#FFFFFF",
    "grid_color": "#F1F5F9",
    "border_color": "#E2E8F0",
    "font_family": "Plus Jakarta Sans, Inter, -apple-system, BlinkMacSystemFont, sans-serif"
}

# Modebar configuration for minimal, distraction-free interactivity
PROPSIGHT_MODEBAR_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "hoverCompareCartesian", "toggleSpikelines"
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "propsight_chart",
        "height": 600,
        "width": 1000,
        "scale": 2
    }
}

def apply_chart_theme(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 380,
    show_legend: bool = True,
    legend_orientation: str = "h",
    hovermode: str = "closest"
) -> go.Figure:
    """
    Apply unified PropSight visual design system to any Plotly figure:
    - Clean typography and restrained slate/teal color hierarchy.
    - Minimalist axes with subtle horizontal gridlines.
    - Responsive margins and floating legends.
    """
    fig.update_layout(
        title={
            "text": title,
            "font": {
                "family": PROPSIGHT_COLORS["font_family"],
                "size": 15,
                "color": PROPSIGHT_COLORS["navy_primary"],
                "weight": "bold"
            },
            "x": 0.0,
            "xanchor": "left",
            "y": 0.96
        } if title else None,
        font={
            "family": PROPSIGHT_COLORS["font_family"],
            "size": 12,
            "color": PROPSIGHT_COLORS["slate_dark"]
        },
        paper_bgcolor=PROPSIGHT_COLORS["card_bg"],
        plot_bgcolor=PROPSIGHT_COLORS["plot_bg"],
        height=height,
        hovermode=hovermode,
        hoverlabel={
            "bgcolor": PROPSIGHT_COLORS["navy_primary"],
            "font_size": 12,
            "font_family": PROPSIGHT_COLORS["font_family"],
            "font_color": "#FFFFFF",
            "bordercolor": "rgba(255,255,255,0.15)"
        },
        margin=dict(l=35, r=25, t=55 if title else 25, b=40),
        xaxis={
            "title": {"text": xaxis_title, "font": {"size": 12, "color": PROPSIGHT_COLORS["slate_muted"]}} if xaxis_title else None,
            "showgrid": False,
            "linecolor": PROPSIGHT_COLORS["border_color"],
            "linewidth": 1,
            "tickfont": {"size": 11, "color": PROPSIGHT_COLORS["slate_muted"]},
            "zeroline": False
        },
        yaxis={
            "title": {"text": yaxis_title, "font": {"size": 12, "color": PROPSIGHT_COLORS["slate_muted"]}} if yaxis_title else None,
            "showgrid": True,
            "gridcolor": PROPSIGHT_COLORS["grid_color"],
            "gridwidth": 1,
            "linecolor": PROPSIGHT_COLORS["border_color"],
            "linewidth": 1,
            "tickfont": {"size": 11, "color": PROPSIGHT_COLORS["slate_muted"]},
            "zeroline": False
        },
        legend={
            "orientation": legend_orientation,
            "yanchor": "bottom" if legend_orientation == "h" else "top",
            "y": 1.02 if legend_orientation == "h" else 0.98,
            "xanchor": "right" if legend_orientation == "h" else "right",
            "x": 1.0,
            "font": {"size": 11, "color": PROPSIGHT_COLORS["slate_dark"]},
            "bgcolor": "rgba(255, 255, 255, 0.7)",
            "bordercolor": "rgba(226, 232, 240, 0.6)",
            "borderwidth": 1
        } if show_legend else {"visible": False}
    )
    return fig

# --- SPECIFIC REUSABLE CHART BUILDERS ---

def build_timeseries_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    yaxis_title: str = "Rate (₹/sq.ft.)",
    color_map: Optional[Dict[str, str]] = None,
    show_range_slider: bool = False,
    show_moving_avg: bool = False,
    ma_window: int = 4
) -> go.Figure:
    """
    Build a multi-series interactive time-series chart with optional moving averages and range sliders.
    """
    fig = go.Figure()

    if color_col:
        unique_groups = df[color_col].unique()
        for group in unique_groups:
            grp_df = df[df[color_col] == group].sort_values(by=x_col)
            line_color = color_map.get(group, PROPSIGHT_COLORS["primary_teal"]) if color_map else None

            # Primary line
            fig.add_trace(go.Scatter(
                x=grp_df[x_col],
                y=grp_df[y_col],
                mode="lines+markers",
                name=str(group),
                line=dict(color=line_color, width=2.5),
                marker=dict(size=5),
                hovertemplate=f"<b>{group}</b><br>Quarter: %{{x}}<br>{yaxis_title}: ₹%{{y:,.1f}}<extra></extra>"
            ))

            # Optional moving average
            if show_moving_avg and len(grp_df) >= ma_window:
                ma_series = grp_df[y_col].rolling(window=ma_window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=grp_df[x_col],
                    y=ma_series,
                    mode="lines",
                    name=f"{group} ({ma_window}Q Trend)",
                    line=dict(color=line_color, width=1.5, dash="dot"),
                    opacity=0.6,
                    hoverinfo="skip"
                ))
    else:
        df_sorted = df.sort_values(by=x_col)
        fig.add_trace(go.Scatter(
            x=df_sorted[x_col],
            y=df_sorted[y_col],
            mode="lines+markers",
            name="Quarterly Rate",
            line=dict(color=PROPSIGHT_COLORS["primary_teal"], width=2.5),
            marker=dict(size=5, color=PROPSIGHT_COLORS["navy_primary"]),
            hovertemplate=f"Quarter: %{{x}}<br>{yaxis_title}: ₹%{{y:,.1f}}<extra></extra>"
        ))

        if show_moving_avg and len(df_sorted) >= ma_window:
            ma_series = df_sorted[y_col].rolling(window=ma_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df_sorted[x_col],
                y=ma_series,
                mode="lines",
                name=f"{ma_window}Q Rolling Trend",
                line=dict(color=PROPSIGHT_COLORS["blue_accent"], width=1.5, dash="dot"),
                opacity=0.8,
                hovertemplate="Trend: ₹%{y:,.1f}<extra></extra>"
            ))

    apply_chart_theme(fig, title=title, xaxis_title="Timeline", yaxis_title=yaxis_title, hovermode="x unified" if color_col else "closest")

    if show_range_slider:
        fig.update_xaxes(
            rangeslider=dict(visible=True, thickness=0.06, bgcolor="#F8FAFC"),
            rangeselector=dict(
                buttons=list([
                    dict(count=4, label="1Y", step="all", stepmode="backward"),
                    dict(count=8, label="2Y", step="all", stepmode="backward"),
                    dict(count=16, label="4Y", step="all", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                font=dict(size=10, color=PROPSIGHT_COLORS["slate_dark"]),
                bgcolor="#FFFFFF",
                activecolor=PROPSIGHT_COLORS["primary_teal"]
            )
        )

    return fig

def build_price_distribution_chart(
    df: pd.DataFrame,
    col_name: str = "price_lakhs",
    color_col: Optional[str] = "region",
    color_map: Optional[Dict[str, str]] = None,
    title: str = "Listing Price Distribution (₹ Lakhs)",
    xaxis_title: str = "Listing Price (₹ Lakhs)",
    nbins: int = 30
) -> go.Figure:
    """
    Build an elegant multi-layer price distribution chart with median marker and statistics.
    """
    fig = px.histogram(
        df,
        x=col_name,
        color=color_col,
        color_discrete_map=color_map or {"Pune": PROPSIGHT_COLORS["blue_accent"], "PCMC": PROPSIGHT_COLORS["emerald_accent"]},
        nbins=nbins,
        barmode="overlay",
        opacity=0.75,
        labels={col_name: xaxis_title}
    )

    # Median reference line
    median_val = float(df[col_name].median())
    mean_val = float(df[col_name].mean())

    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color=PROPSIGHT_COLORS["navy_primary"],
        line_width=2,
        annotation_text=f"Median: ₹{median_val:.1f}{'L' if 'lakh' in col_name.lower() else ''}",
        annotation_position="top right",
        annotation_font=dict(size=11, color=PROPSIGHT_COLORS["navy_primary"], weight="bold")
    )

    apply_chart_theme(fig, title=title, xaxis_title=xaxis_title, yaxis_title="Number of Properties")
    return fig

def build_horizontal_bar_rank(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    xaxis_title: str = "Average Rate (₹/sq.ft.)",
    color_scale: str = "Teal",
    top_n: int = 12
) -> go.Figure:
    """
    Build a ranked horizontal bar chart with color-intensity scaling and clean data labels.
    """
    df_plot = df.sort_values(by=x_col, ascending=True).tail(top_n)

    fig = px.bar(
        df_plot,
        x=x_col,
        y=y_col,
        orientation="h",
        color=color_col or x_col,
        color_continuous_scale=color_scale,
        labels={x_col: xaxis_title, y_col: "Locality"}
    )

    fig.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        textfont=dict(size=10, color=PROPSIGHT_COLORS["slate_dark"]),
        cliponaxis=False
    )

    apply_chart_theme(fig, title=title, xaxis_title=xaxis_title, yaxis_title="")
    fig.update_layout(coloraxis_showscale=False)
    return fig

def build_price_vs_area_scatter(
    df: pd.DataFrame,
    x_col: str = "carpet_area_sqft",
    y_col: str = "price_lakhs",
    color_col: str = "region",
    color_map: Optional[Dict[str, str]] = None,
    title: str = "Price vs. Carpet Area Relationship",
    sample_size: int = 1000
) -> go.Figure:
    """
    Build an interactive scatter plot with trendline and rich property inspection hover tooltips.
    """
    plot_df = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_map or {"Pune": PROPSIGHT_COLORS["blue_accent"], "PCMC": PROPSIGHT_COLORS["emerald_accent"]},
        opacity=0.68,
        hover_data={
            "locality": True,
            "bhk": True,
            "floor": True,
            "price_per_sqft": ":,.0f",
            x_col: ":,.0f",
            y_col: ":,.2f"
        },
        labels={
            x_col: "Carpet Area (sq.ft.)",
            y_col: "Listing Price (₹ Lakhs)",
            "locality": "Locality",
            "bhk": "BHK",
            "floor": "Floor",
            "price_per_sqft": "Rate (₹/sqft)"
        }
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))

    apply_chart_theme(fig, title=title, xaxis_title="Carpet Area (sq.ft.)", yaxis_title="Listing Price (₹ Lakhs)")
    return fig

def build_valuation_gauge_bullet(
    estimated_lakhs: float,
    lower_lakhs: float,
    upper_lakhs: float,
    asking_lakhs: Optional[float] = None,
    locality_median_lakhs: Optional[float] = None
) -> go.Figure:
    """
    Build an executive bullet / range gauge chart showing:
    - 80% Expected Prediction Interval (shaded band).
    - Model Point Valuation (diamond).
    - Locality Median benchmark (reference line).
    - Asking Price with status color (circle).
    """
    fig = go.Figure()

    # 1. 80% Confidence Interval Band
    fig.add_trace(go.Bar(
        name="80% Expected Interval",
        y=["Valuation Assessment"],
        x=[upper_lakhs - lower_lakhs],
        base=[lower_lakhs],
        orientation="h",
        marker=dict(
            color="rgba(13, 148, 136, 0.18)",
            line=dict(color=PROPSIGHT_COLORS["primary_teal"], width=1.5)
        ),
        text=f"Expected Band: ₹{lower_lakhs:.1f}L – ₹{upper_lakhs:.1f}L",
        textposition="inside",
        hoverinfo="text"
    ))

    # 2. Locality Median Benchmark (if provided)
    if locality_median_lakhs:
        fig.add_trace(go.Scatter(
            x=[locality_median_lakhs],
            y=["Valuation Assessment"],
            mode="markers",
            name="Locality Median",
            marker=dict(size=14, color=PROPSIGHT_COLORS["slate_muted"], symbol="line-ns", line=dict(width=3)),
            hovertemplate=f"Locality Median: ₹{locality_median_lakhs:.1f}L<extra></extra>"
        ))

    # 3. Model Point Estimate
    fig.add_trace(go.Scatter(
        x=[estimated_lakhs],
        y=["Valuation Assessment"],
        mode="markers",
        name="Model Valuation",
        marker=dict(size=15, color=PROPSIGHT_COLORS["navy_primary"], symbol="diamond"),
        hovertemplate=f"Model Estimate: ₹{estimated_lakhs:.2f}L<extra></extra>"
    ))

    # 4. Asking Price (if provided)
    if asking_lakhs:
        if asking_lakhs < lower_lakhs:
            ask_color = PROPSIGHT_COLORS["emerald_accent"]
            status_tag = "Undervalued"
        elif asking_lakhs > upper_lakhs:
            ask_color = PROPSIGHT_COLORS["rose_accent"]
            status_tag = "Overpriced"
        else:
            ask_color = PROPSIGHT_COLORS["blue_accent"]
            status_tag = "Within Expected Range"

        fig.add_trace(go.Scatter(
            x=[asking_lakhs],
            y=["Valuation Assessment"],
            mode="markers",
            name=f"Asking Price ({status_tag})",
            marker=dict(size=18, color=ask_color, symbol="circle", line=dict(color="#FFFFFF", width=2)),
            hovertemplate=f"Seller Asking Price: ₹{asking_lakhs:.2f}L ({status_tag})<extra></extra>"
        ))

    apply_chart_theme(fig, title="Valuation Range & Market Benchmark (₹ Lakhs)", xaxis_title="Price (₹ Lakhs)", yaxis_title="", height=230)
    fig.update_layout(margin=dict(l=20, r=25, t=45, b=35))
    return fig

def build_actual_vs_predicted_scatter(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    sample_size: int = 500
) -> go.Figure:
    """
    Build an Actual vs Predicted regression scatter plot with 45-degree ideal line
    and error-gradient encoding.
    """
    if len(y_actual) > sample_size:
        idx = np.random.choice(len(y_actual), size=sample_size, replace=False)
        act = y_actual[idx]
        pred = y_pred[idx]
    else:
        act, pred = y_actual, y_pred

    residuals = act - pred
    abs_errors = np.abs(residuals)

    fig = go.Figure()

    # 45-degree reference line
    min_val = min(float(np.min(act)), float(np.min(pred))) * 0.95
    max_val = max(float(np.max(act)), float(np.max(pred))) * 1.05

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Prediction (y = x)",
        line=dict(color=PROPSIGHT_COLORS["slate_muted"], dash="dash", width=1.5),
        hoverinfo="skip"
    ))

    # Scatter points colored by absolute error
    fig.add_trace(go.Scatter(
        x=act,
        y=pred,
        mode="markers",
        name="Test Properties",
        marker=dict(
            size=6,
            color=abs_errors,
            colorscale="Teal",
            colorbar=dict(title="Error (₹L)", thickness=10, len=0.7),
            showscale=True,
            line=dict(width=0.5, color="white")
        ),
        hovertemplate="Actual: ₹%{x:.1f}L<br>Predicted: ₹%{y:.1f}L<br>Error: ₹%{marker.color:.1f}L<extra></extra>"
    ))

    apply_chart_theme(
        fig,
        title="Actual vs. Model Predicted Prices (Held-Out Test Set)",
        xaxis_title="Actual Listing Price (₹ Lakhs)",
        yaxis_title="Predicted Price (₹ Lakhs)",
        height=400
    )
    return fig

def build_residuals_histogram(y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Build a distribution plot of model test residuals with normal distribution curve.
    """
    residuals = y_actual - y_pred

    fig = px.histogram(
        x=residuals,
        nbins=40,
        color_discrete_sequence=[PROPSIGHT_COLORS["primary_teal"]],
        opacity=0.75,
        labels={"x": "Prediction Residual (Actual − Predicted in ₹ Lakhs)"}
    )

    # Reference zero line
    fig.add_vline(x=0, line_dash="solid", line_color=PROPSIGHT_COLORS["navy_primary"], line_width=1.5)
    
    # 80% Quantiles
    q10 = float(np.percentile(residuals, 10))
    q90 = float(np.percentile(residuals, 90))
    fig.add_vline(x=q10, line_dash="dash", line_color=PROPSIGHT_COLORS["amber_accent"], line_width=1.5, annotation_text="Q10")
    fig.add_vline(x=q90, line_dash="dash", line_color=PROPSIGHT_COLORS["amber_accent"], line_width=1.5, annotation_text="Q90")

    apply_chart_theme(
        fig,
        title="Empirical Residual Distribution (Uncertainty Basis)",
        xaxis_title="Residual (₹ Lakhs)",
        yaxis_title="Frequency",
        height=350
    )
    return fig
