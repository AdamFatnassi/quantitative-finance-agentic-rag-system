"""
theme.py — Design System and UI Utilities for Quant Agentic RAG
Modern dark institutional styling, color palettes, and Plotly templates.
"""

import streamlit as st

# ─── Color Palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg_dark": "#0a0d14",
    "bg_card": "#131826",
    "bg_card_alt": "#1a2136",
    "border": "#252d47",
    "border_highlight": "#3b4770",
    "primary": "#6366f1",        # Indigo
    "secondary": "#38bdf8",      # Sky blue
    "accent": "#a855f7",         # Purple
    "success": "#10b981",        # Emerald
    "danger": "#ef4444",         # Red
    "warning": "#f59e0b",        # Amber
    "text": "#f1f5f9",           # Slate light
    "text_muted": "#94a3b8",     # Slate muted
}

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Safely convert hex string (e.g. #5c6bc0 or 5c6bc0) to rgba(r, g, b, a)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    except Exception:
        return f"rgba(100, 116, 139, {alpha})"


# ─── Global Custom CSS ─────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Global application styling */
    .stApp {
        background: radial-gradient(circle at top right, #111728 0%, #090c14 100%);
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar aesthetics */
    section[data-testid="stSidebar"] {
        background-color: #0d111d !important;
        border-right: 1px solid #1e263d !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    /* Metric cards */
    .quant-card {
        background: linear-gradient(135deg, rgba(19, 24, 38, 0.85) 0%, rgba(26, 33, 54, 0.65) 100%);
        backdrop-filter: blur(8px);
        border: 1px solid #232c45;
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .quant-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.15);
    }
    .quant-title {
        color: #94a3b8;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .quant-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .quant-sub {
        font-size: 0.8rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .val-good { color: #10b981; }
    .val-bad  { color: #ef4444; }
    .val-neutral { color: #94a3b8; }
    .val-indigo { color: #818cf8; }

    /* Header banner styling */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 79, 0.45) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid #2a3454;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: 0; right: 0;
        width: 250px; height: 100%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Badges & Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .pill-green { background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .pill-red { background: rgba(239, 68, 68, 0.16); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
    .pill-indigo { background: rgba(99, 102, 241, 0.16); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.35); }
    .pill-amber { background: rgba(245, 158, 11, 0.16); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .pill-cyan { background: rgba(6, 182, 212, 0.16); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.35); }

    /* Modern tabs and inputs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1e263d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 500;
        padding: 6px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #232c45 !important;
        color: #f8fafc !important;
        font-weight: 600;
    }

    /* Streamlit native metric overrides */
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Pre-flight Backtest Panel */
    .preflight-box {
        background: linear-gradient(135deg, rgba(20, 27, 45, 0.9) 0%, rgba(13, 18, 30, 0.95) 100%);
        border: 1px solid #33416b;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
</style>
"""


def apply_theme():
    """Apply global CSS enhancements to current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "📈", badge: str = "PRODUCTION"):
    """Render a modern unified header banner."""
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1 class="hero-title">{icon} {title}</h1>
                <div class="hero-subtitle">{subtitle}</div>
            </div>
            <div>
                <span class="pill pill-indigo">{badge}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric(title: str, value: str, sub: str = "", status: str = "neutral", icon: str = "") -> str:
    """Return HTML for a modern styled metric card."""
    color_cls = f"val-{status}"
    icon_html = f"<span>{icon}</span> " if icon else ""
    sub_html = f'<div class="quant-sub {color_cls}">{sub}</div>' if sub else ""
    return f"""
    <div class="quant-card">
        <div class="quant-title">{icon_html}{title}</div>
        <div class="quant-value">{value}</div>
        {sub_html}
    </div>
    """


def get_plotly_layout(**kwargs) -> dict:
    """Return standard dark theme layout options for Plotly charts."""
    base_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 20, 32, 0.55)",
        font=dict(color="#cbd5e1", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
        xaxis=dict(
            gridcolor="#1e263d",
            zerolinecolor="#2d3755",
            showline=True,
            linecolor="#2d3755",
        ),
        yaxis=dict(
            gridcolor="#1e263d",
            zerolinecolor="#2d3755",
            showline=True,
            linecolor="#2d3755",
        ),
        margin=dict(l=15, r=15, t=35, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
    )
    base_layout.update(kwargs)
    return base_layout
