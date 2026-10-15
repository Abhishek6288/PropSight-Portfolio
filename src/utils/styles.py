"""
UI Styling, CSS Injection, and Theme helpers for PropSight.
Enforces an executive, mature real-estate decision-support aesthetic
with 4K-grade infrastructure visuals, glassmorphism, and responsive components.
"""

import streamlit as st
import base64
from pathlib import Path
from src.utils.config import PROJECT_ROOT

ASSETS_DIR = PROJECT_ROOT / "assets" / "images"

def get_image_base64(filename: str) -> str:
    """Return base64 data URI for an asset image, or empty string if not found."""
    img_path = ASSETS_DIR / filename
    if img_path.exists():
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    return ""

def apply_custom_css():
    """Inject polished executive CSS styles for PropSight."""
    st.markdown("""
    <style>
    /* Global styling and typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 4K Hero Cinematic Banner */
    .hero-container {
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-bg {
        width: 100%;
        min-height: 280px;
        background-size: cover;
        background-position: center;
        position: relative;
        display: flex;
        align-items: center;
    }
    .hero-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.94) 0%, rgba(15, 23, 42, 0.82) 50%, rgba(30, 41, 59, 0.70) 100%);
        backdrop-filter: blur(2px);
    }
    .hero-content {
        position: relative;
        z-index: 2;
        padding: 2.25rem 2.5rem;
        color: #FFFFFF;
        width: 100%;
    }
    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(14, 165, 233, 0.18);
        border: 1px solid rgba(14, 165, 233, 0.45);
        color: #38BDF8;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0;
        color: #F8FAFC;
    }
    .hero-title span {
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #CBD5E1;
        margin-top: 0.5rem;
        max-width: 720px;
        line-height: 1.5;
        font-weight: 400;
    }
    .hero-stats-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin-top: 1.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-stat-item {
        display: flex;
        flex-direction: column;
    }
    .hero-stat-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .hero-stat-label {
        font-size: 0.72rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Executive Page Header Banner (for Subpages) */
    .propsight-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 1.25rem 1.75rem;
        border-radius: 10px;
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
    
    /* Modern Glass KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(0,0,0,0.08);
        border-color: #CBD5E1;
    }
    .kpi-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .kpi-caption {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 0.35rem;
    }

    /* Visual Showcase Card with Image & Hover Animation */
    .showcase-card {
        background: #FFFFFF;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.25s ease;
        margin-bottom: 1.25rem;
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .showcase-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #0EA5E9;
    }
    .showcase-img-wrap {
        position: relative;
        width: 100%;
        height: 160px;
        overflow: hidden;
    }
    .showcase-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    .showcase-card:hover .showcase-img {
        transform: scale(1.06);
    }
    .showcase-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(15, 23, 42, 0.85);
        color: #38BDF8;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        backdrop-filter: blur(4px);
        letter-spacing: 0.03em;
    }
    .showcase-body {
        padding: 1rem 1.15rem;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        justify-content: space-between;
    }
    .showcase-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.35rem 0;
    }
    .showcase-desc {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.45;
        margin-bottom: 0.75rem;
    }
    .showcase-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #F1F5F9;
        padding-top: 0.65rem;
        font-size: 0.76rem;
        color: #64748B;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.22rem 0.6rem;
        border-radius: 6px;
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

    /* Callout Boxes */
    .callout-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin: 1.25rem 0;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
    }
    .callout-warning {
        background-color: #FFFBEB;
        border-left-color: #F59E0B;
        color: #92400E;
    }
    
    /* Table styling */
    .dataframe th {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_hero_banner(
    title: str = "PropSight",
    subtitle: str = "Localized Real-Estate Intelligence & Decision Support",
    tag: str = "Pune & Pimpri-Chinchwad Market Engine",
    stats: list = None
):
    """Render a dramatic, 4K aesthetic urban infrastructure hero banner."""
    apply_custom_css()
    bg_uri = get_image_base64("hero_skyline.jpg")
    bg_style = f"background-image: url('{bg_uri}');" if bg_uri else "background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);"
    
    stats_html = ""
    if stats:
        items_html = "".join([
            f'<div class="hero-stat-item"><span class="hero-stat-value">{s.get("value", "")}</span><span class="hero-stat-label">{s.get("label", "")}</span></div>'
            for s in stats
        ])
        stats_html = f'<div class="hero-stats-row">{items_html}</div>'

    html = (
        f'<div class="hero-container">'
        f'<div class="hero-bg" style="{bg_style}">'
        f'<div class="hero-overlay"></div>'
        f'<div class="hero-content">'
        f'<div class="hero-badge-pill">🏙️ {tag}</div>'
        f'<h1 class="hero-title">PropSight <span>Intelligence</span></h1>'
        f'<div class="hero-subtitle">{subtitle}</div>'
        f'{stats_html}'
        f'</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_header(title: str, subtitle: str, page_tag: str = "Pune & PCMC Market Intelligence"):
    """Render the standard executive header across subpages."""
    apply_custom_css()
    html = (
        f'<div class="propsight-header">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
        f'<div><h1 class="propsight-title">{title}</h1><div class="propsight-subtitle">{subtitle}</div></div>'
        f'<div style="text-align: right;"><span class="badge badge-synthetic">Demonstration Baseline</span>'
        f'<div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem;">{page_tag}</div></div>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_kpi_card(label: str, value: str, caption: str = ""):
    """Render a clean, high-contrast metric card."""
    cap_html = f'<div class="kpi-caption">{caption}</div>' if caption else ''
    html = f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{cap_html}</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_showcase_card(
    title: str,
    desc: str,
    image_name: str,
    badge_text: str,
    meta_left: str,
    meta_right: str
):
    """Render an aesthetic visual card with image header and hover zoom."""
    img_uri = get_image_base64(image_name)
    img_src = img_uri if img_uri else "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800"

    html = (
        f'<div class="showcase-card">'
        f'<div class="showcase-img-wrap">'
        f'<img class="showcase-img" src="{img_src}" alt="{title}" />'
        f'<span class="showcase-badge">{badge_text}</span>'
        f'</div>'
        f'<div class="showcase-body">'
        f'<div><h4 class="showcase-title">{title}</h4><div class="showcase-desc">{desc}</div></div>'
        f'<div class="showcase-meta"><span>{meta_left}</span><span style="font-weight: 600; color: #0EA5E9;">{meta_right}</span></div>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_module_nav_card(icon: str, title: str, desc: str, page_label: str):
    """Render a clean navigation card for application modules."""
    html = (
        f'<div class="showcase-card" style="padding: 1.25rem;">'
        f'<div style="font-size: 1.5rem; margin-bottom: 0.35rem;">{icon}</div>'
        f'<h4 style="margin: 0 0 0.4rem 0; color: #0F172A;">{title}</h4>'
        f'<p style="font-size: 0.83rem; color: #475569; margin-bottom: 0.75rem; line-height: 1.45;">{desc}</p>'
        f'<div style="font-size: 0.78rem; font-weight: 600; color: #0EA5E9;">Open Page: {page_label} →</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_spotlight_card(label: str, name: str, rate_str: str, cagr_str: str, caption: str, border_color: str = "#0EA5E9"):
    """Render a micro-market spotlight card."""
    html = (
        f'<div class="kpi-card" style="border-top: 3px solid {border_color};">'
        f'<div class="kpi-label">{label}</div>'
        f'<h4 style="margin: 0.2rem 0; color: #0F172A;">{name}</h4>'
        f'<div style="font-size: 1.3rem; font-weight: 700; color: #0F172A;">{rate_str}</div>'
        f'<div style="font-size: 0.78rem; color: #10B981; font-weight: 600;">{cagr_str}</div>'
        f'<div class="kpi-caption">{caption}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

