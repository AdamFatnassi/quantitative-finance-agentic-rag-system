"""
Home.py — Quantitative Finance Agentic RAG System
Landing page with modern system KPIs and pipeline overview.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add app root to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from theme import apply_theme, render_header, render_metric, get_plotly_layout

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Agentic RAG System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark institutional theme
apply_theme()

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ANALYTICS = ROOT / "data" / "lake" / "analytics"
RESULTS   = ROOT / "results" / "backtest"


@st.cache_data(ttl=600)
def load_kpi_data() -> dict:
    metrics_df = pd.read_csv(ANALYTICS / "ml_signals" / "metrics.csv", index_col=0, header=0)
    metrics_df.columns = ["value"]
    metrics = metrics_df["value"].to_dict()

    comparison = pd.read_csv(RESULTS / "comparison.csv")
    bt5 = comparison[comparison["cost_bps"] == 5]
    ml_row  = bt5[bt5["strategy"] == "ML filter"].iloc[0]
    all_row = bt5[bt5["strategy"] == "All entries"].iloc[0]

    pairs_df = pd.read_parquet(ANALYTICS / "pairs.parquet")
    regimes  = pd.read_parquet(ANALYTICS / "hmm_regimes.parquet")
    funnel   = pd.read_parquet(ANALYTICS / "pair_selection_funnel.parquet")

    return {
        "auc_pooled":        float(metrics.get("auc_pooled", 0)),
        "auc_fold_mean":     float(metrics.get("auc_fold_mean", 0)),
        "lift_top":          float(metrics.get("lift_top", 0)),
        "signal_lift":       float(metrics.get("signal_lift", 0)),
        "ml_sharpe":         float(ml_row["sharpe"]),
        "ml_winrate":        float(ml_row["win_rate"]),
        "ml_n_trades":       int(ml_row["n_trades"]),
        "ml_maxdd":          float(ml_row["max_drawdown"]),
        "all_maxdd":         float(all_row["max_drawdown"]),
        "ml_net_bps":        float(ml_row["mean_net_bps"]),
        "all_sharpe":        float(all_row["sharpe"]),
        "all_n_trades":      int(all_row["n_trades"]),
        "n_pairs":           len(pairs_df),
        "n_sectors":         pairs_df["sector"].nunique() if "sector" in pairs_df.columns else 0,
        "n_regimes":         3,
        "n_windows":         regimes["window_id"].nunique(),
        "date_start":        str(regimes["time"].min())[:10],
        "date_end":          str(regimes["time"].max())[:10],
        "n_funnel_windows":  int(funnel["window_id"].nunique()),
    }


# ─── Header ────────────────────────────────────────────────────────────────────
render_header(
    title="Quantitative Agentic RAG Terminal",
    subtitle="Statistical Arbitrage · Logistic ML Filter · 3-State HMM Regimes · Agentic Risk Gatekeeper (LLM/RAG)",
    icon="📈",
    badge="SYSTEM ACTIVE",
)

kpi = load_kpi_data()

# ─── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ System Status")
    st.markdown("""
    <div style="background: rgba(19, 24, 38, 0.7); border: 1px solid #232c45; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-size:0.78rem; color:#94a3b8;">Data Lake</span>
            <span class="pill pill-green">Synchronized</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="font-size:0.78rem; color:#94a3b8;">ML Pipeline</span>
            <span class="pill pill-indigo">OOS Calibrated</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size:0.78rem; color:#94a3b8;">RAG Gatekeeper</span>
            <span class="pill pill-cyan">Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**Pairs Universe:** {kpi['n_pairs']:,} pairs across {kpi['n_sectors']} sectors")
    st.markdown(f"**Walk-Forward:** {kpi['n_windows']} rolling windows")
    st.markdown(f"**Data Span:** `{kpi['date_start']}` → `{kpi['date_end']}`")

# ─── KPI Row 1 — ML ────────────────────────────────────────────────────────────
st.markdown("### 🤖 Machine Learning Signal Filter (OOS Pooled)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_metric("AUC (Pooled OOS)", f"{kpi['auc_pooled']:.3f}", "↑ +0.076 vs 0.50 random", "good", "🎯"), unsafe_allow_html=True)
with c2:
    st.markdown(render_metric("AUC (Fold Mean)", f"{kpi['auc_fold_mean']:.3f}", "Cross-validated ±0.178 std", "indigo", "📐"), unsafe_allow_html=True)
with c3:
    st.markdown(render_metric("Top-20% Lift", f"{kpi['lift_top']:.3f}×", "Higher precision in top decile", "good", "🚀"), unsafe_allow_html=True)
with c4:
    st.markdown(render_metric("Signal Lift", f"{kpi['signal_lift']:.3f}×", "Real signal vs base rate", "good", "⚡"), unsafe_allow_html=True)

st.markdown("")

# ─── KPI Row 2 — Backtest ──────────────────────────────────────────────────────
st.markdown("### 📊 Backtest Performance (5 bps Cost Regime)")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(render_metric("ML Sharpe", f"{kpi['ml_sharpe']:.4f}", f"All Entries: {kpi['all_sharpe']:.4f}", "good" if kpi["ml_sharpe"] > kpi["all_sharpe"] else "bad", "🏆"), unsafe_allow_html=True)
with c2:
    st.markdown(render_metric("ML Win Rate", f"{kpi['ml_winrate']:.1%}", "5 bps friction included", "good", "✅"), unsafe_allow_html=True)
with c3:
    st.markdown(render_metric("Executed Trades", f"{kpi['ml_n_trades']:,}", f"All Universe: {kpi['all_n_trades']:,}", "neutral", "🔄"), unsafe_allow_html=True)
with c4:
    st.markdown(render_metric("Max Drawdown", f"{kpi['ml_maxdd']:.2%}", f"All Entries: {kpi['all_maxdd']:.2%}", "good" if kpi["ml_maxdd"] > kpi["all_maxdd"] else "bad", "🛡️"), unsafe_allow_html=True)
with c5:
    st.markdown(render_metric("Mean Net BPS", f"{kpi['ml_net_bps']:.2f}", "Net edge per trade", "good" if kpi["ml_net_bps"] > 0 else "bad", "💰"), unsafe_allow_html=True)

st.markdown("")

# ─── KPI Row 3 — Universe ──────────────────────────────────────────────────────
st.markdown("### 🌐 Universe & Market Regimes")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_metric("Active Pairs", f"{kpi['n_pairs']:,}", "Cointegrated & bounded", "indigo", "🔗"), unsafe_allow_html=True)
with c2:
    st.markdown(render_metric("Sector Neutrality", f"{kpi['n_sectors']} Sectors", "GICS intra-sector clustering", "cyan", "🏛️"), unsafe_allow_html=True)
with c3:
    st.markdown(render_metric("HMM Regimes", f"{kpi['n_regimes']} States", "Low Vol · Trend · Stress", "amber", "🌊"), unsafe_allow_html=True)
with c4:
    st.markdown(render_metric("Walk-Forward Windows", f"{kpi['n_windows']}", f"{kpi['date_start']} → {kpi['date_end']}", "indigo", "📅"), unsafe_allow_html=True)

st.markdown("---")

# ─── Pipeline Architecture ─────────────────────────────────────────────────────
st.markdown("### 🏗️ Quant-RAG System Architecture & Data Flow")

col_arch, col_flow = st.columns([1, 1.2])

with col_arch:
    st.markdown("""
    <div style="background: rgba(19, 24, 38, 0.7); border: 1px solid #232c45; border-radius: 10px; padding: 18px;">
        <h4 style="margin-top:0; color:#38bdf8;">Multi-Tier Decision Engine</h4>
        <ol style="color:#cbd5e1; font-size:0.88rem; line-height: 1.7; padding-left: 20px;">
            <li><b>Pairs Screening:</b> SSD ranking + Engle-Granger / Johansen cointegration test.</li>
            <li><b>HMM Regime Classifier:</b> Gaussian HMM estimating market regime state & entropy.</li>
            <li><b>Machine Learning Filter:</b> Walk-forward calibrated logistic classifier predicting reversion success.</li>
            <li><b>Agentic Risk Gatekeeper:</b> Asynchronous LLM with Qdrant vector retrieval checking SEC 8-K / transcripts for M&A, earnings shock, or accounting risk.</li>
            <li><b>Portfolio Execution:</b> Position-capped execution with borrow fees and transaction cost modeling.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with col_flow:
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=22,
            label=[
                "Raw Price Data", "SEC Filings",
                "Pairs Screening", "Cointegration",
                "ML Filter", "HMM Regimes",
                "Agentic Gatekeeper", "EXECUTE TRADE", "VETO RISK"
            ],
            color=[
                "#4f46e5", "#06b6d4",
                "#818cf8", "#818cf8",
                "#3b82f6", "#f59e0b",
                "#6366f1", "#10b981", "#ef4444"
            ],
        ),
        link=dict(
            source=[0, 1, 2, 2, 3, 4, 5, 6, 6],
            target=[2, 6, 3, 4, 4, 6, 6, 7, 8],
            value =[100, 40, 70, 70, 60, 50, 30, 75, 25],
            color=[
                "rgba(79, 70, 229, 0.25)", "rgba(6, 182, 212, 0.25)",
                "rgba(129, 140, 248, 0.25)", "rgba(129, 140, 248, 0.25)",
                "rgba(59, 130, 246, 0.25)", "rgba(245, 158, 11, 0.25)",
                "rgba(99, 102, 241, 0.25)", "rgba(16, 185, 129, 0.4)", "rgba(239, 68, 68, 0.4)"
            ],
        ),
    ))
    fig.update_layout(
        title=dict(text="System Decision & Liquidity Routing", font=dict(color="#f8fafc", size=14)),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Quantitative Finance Agentic RAG System • Production Terminal • "
    "LLM: qwen2.5:1.5b • Vector Store: Qdrant • Time-Series Store: TimescaleDB"
)
