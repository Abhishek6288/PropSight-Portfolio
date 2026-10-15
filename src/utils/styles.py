"""
UI Styling, CSS Injection, and Theme helpers for PropSight.
Enforces an executive, mature real-estate decision-support aesthetic.
"""

import streamlit as st

def apply_custom_css():
    """Inject polished executive CSS styles for PropSight."""
    st.markdown("""
    <style>
    /* Global styling and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Executive Header Banner */
    .propsight-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 1.25rem 1.75rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #0EA5E9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .propsight-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #F8FAFC;
    }
    .propsight-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 0.25rem;
        font-weight: 400;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-caption {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.35rem;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-undervalued {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .badge-fair {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .badge-overpriced {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .badge-synthetic {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .badge-verified {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }

    /* Info / Warning Callout Box */
    .callout-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1.15rem;
        border-radius: 4px;
        margin: 1rem 0;
        font-size: 0.86rem;
        color: #334155;
    }
    .callout-warning {
        background-color: #FFFBEB;
        border-left-color: #F59E0B;
        color: #92400E;
    }
    
    /* Table headers */
    .dataframe th {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* Plotly chart container */
    .chart-container {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, page_tag: str = "Pune & PCMC Market Intelligence"):
    """Render the standard executive header across pages."""
    apply_custom_css()
    st.markdown(f"""
    <div class="propsight-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1 class="propsight-title">{title}</h1>
                <div class="propsight-subtitle">{subtitle}</div>
            </div>
            <div style="text-align: right;">
                <span class="badge badge-synthetic">Demonstration Baseline</span>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem;">{page_tag}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_card(label: str, value: str, caption: str = ""):
    """Render a clean, high-contrast metric card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-caption">{caption}</div>' if caption else ''}
    </div>
    """, unsafe_allow_html=True)
