"""
1_ML_Performance.py — ML Signal Filter Analysis
Displays ROC/AUC, feature importance, calibration, and ablation study.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add app root to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from theme import apply_theme, render_header, render_metric, get_plotly_layout

st.set_page_config(page_title="ML Performance", page_icon="🤖", layout="wide")
apply_theme()

ROOT      = Path(__file__).parent.parent.parent
ANALYTICS = ROOT / "data" / "lake" / "analytics"

# ─── Load ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_ml_data():
    metrics_raw = pd.read_csv(ANALYTICS / "ml_signals" / "metrics.csv", index_col=0, header=0)
    metrics_raw.columns = ["value"]
    metrics = metrics_raw["value"].to_dict()

    fi = pd.read_csv(ANALYTICS / "ml_signals" / "feature_importance.csv")
    fi = fi.sort_values("importance", ascending=True)

    signals = pd.read_parquet(ANALYTICS / "ml_signals" / "signals_oos.parquet")
    ablation = pd.read_csv(ROOT / "results" / "ablation" / "summary.csv")
    return metrics, fi, signals, ablation


metrics, fi, signals, ablation = load_ml_data()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🤖 ML Signal Filter")
st.sidebar.markdown(
    "Walk-forward cross-validated **L2-penalized Logistic Classifier** trained strictly on trailing data."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**OOS Observations:** {len(signals):,}")
st.sidebar.markdown(f"**Selected Signals:** {(signals['selected'] == True).sum():,}")
st.sidebar.markdown(f"**Base Reversion Rate:** {float(metrics.get('base_rate', 0.28)):.1%}")

# ─── Header ────────────────────────────────────────────────────────────────────
render_header(
    title="ML Signal Filter — Performance & Diagnostics",
    subtitle="Walk-forward out-of-sample evaluation · ROC & Precision-Recall · Feature attribution · Noise ablation",
    icon="🤖",
    badge="CALIBRATED",
)

# ─── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(render_metric("AUC (Pooled OOS)", f"{float(metrics['auc_pooled']):.4f}", "All OOS folds", "good", "🎯"), unsafe_allow_html=True)
with k2:
    st.markdown(render_metric("AUC (Fold Mean)", f"{float(metrics['auc_fold_mean']):.4f}", f"±{float(metrics['auc_fold_std']):.4f}", "indigo", "📐"), unsafe_allow_html=True)
with k3:
    st.markdown(render_metric("Brier Score", f"{float(metrics['brier']):.4f}", f"Baseline: {float(metrics['brier_baseline']):.4f}", "good", "⚖️"), unsafe_allow_html=True)
with k4:
    st.markdown(render_metric("Top-20% Precision", f"{float(metrics['precision_top20']):.4f}", "Precision in top quintile", "good", "💎"), unsafe_allow_html=True)
with k5:
    st.markdown(render_metric("Top-20% Lift", f"{float(metrics['lift_top']):.4f}×", "vs random selection", "good", "🚀"), unsafe_allow_html=True)
with k6:
    st.markdown(render_metric("Signal Lift", f"{float(metrics['signal_lift']):.4f}×", "Real signal vs base rate", "good", "⚡"), unsafe_allow_html=True)

st.markdown("---")

# ─── Row 1: Feature Importance & Calibration ──────────────────────────────────
col_fi, col_cal = st.columns([1, 1.3])

with col_fi:
    st.markdown("### 📊 Feature Importance")
    fig_fi = go.Figure(go.Bar(
        y=fi["feature"],
        x=fi["importance"],
        orientation="h",
        marker=dict(
            color=fi["importance"],
            colorscale="Viridis",
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in fi["importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        **get_plotly_layout(
            xaxis_title="Importance (mean |coef| × std)",
            height=340,
            margin=dict(l=10, r=60, t=20, b=30),
        )
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with col_cal:
    st.markdown("### 🎯 Predicted Probability Calibration")
    pos = signals[signals["y"] == 1]["proba"].dropna()
    neg = signals[signals["y"] == 0]["proba"].dropna()

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Histogram(
        x=neg, name="Non-revert (y=0)",
        histnorm="probability density",
        nbinsx=40,
        marker_color="rgba(239, 68, 68, 0.65)",
        opacity=0.75,
    ))
    fig_cal.add_trace(go.Histogram(
        x=pos, name="Revert (y=1)",
        histnorm="probability density",
        nbinsx=40,
        marker_color="rgba(16, 185, 129, 0.65)",
        opacity=0.75,
    ))
    fig_cal.update_layout(
        **get_plotly_layout(
            barmode="overlay",
            xaxis_title="Predicted Probability",
            yaxis_title="Density",
            height=340,
            legend=dict(orientation="h", y=1.05, x=0),
            margin=dict(l=10, r=10, t=30, b=30),
        )
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# ─── Row 2: Simulated ROC & Precision-Recall ──────────────────────────────────
st.markdown("### 📈 Discrimination Curves (ROC & Precision-Recall)")

col_roc, col_pr = st.columns(2)

valid = signals.dropna(subset=["proba", "y"])
y_true = valid["y"].values
y_score = valid["proba"].values

thresholds = np.linspace(0, 1, 200)
tpr_list, fpr_list = [], []
for t in thresholds:
    pred = (y_score >= t).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    tpr_list.append(tp / (tp + fn + 1e-9))
    fpr_list.append(fp / (fp + tn + 1e-9))

fpr_arr = np.array(fpr_list)
tpr_arr = np.array(tpr_list)
sort_idx = np.argsort(fpr_arr)

# NumPy 2.x safe trapezoid integration
x_sorted = fpr_arr[sort_idx]
y_sorted = tpr_arr[sort_idx]
if hasattr(np, "trapezoid"):
    auc_val = float(np.trapezoid(y_sorted, x_sorted))
elif hasattr(np, "trapz"):
    auc_val = float(np.trapz(y_sorted, x_sorted))
else:
    auc_val = float(np.sum((x_sorted[1:] - x_sorted[:-1]) * (y_sorted[1:] + y_sorted[:-1]) / 2.0))

with col_roc:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=x_sorted, y=y_sorted,
        mode="lines", name=f"ROC (AUC = {auc_val:.3f})",
        line=dict(color="#6366f1", width=2.5),
        fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.15)",
    ))
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="#64748b", dash="dash", width=1.2))
    fig_roc.update_layout(
        **get_plotly_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=320,
            margin=dict(l=10, r=10, t=10, b=30),
        )
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# Precision-Recall
prec_list, rec_list = [], []
for t in thresholds:
    pred = (y_score >= t).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    prec_list.append(tp / (tp + fp + 1e-9))
    rec_list.append(tp / (tp + fn + 1e-9))

base_rate = float(metrics.get("base_rate", y_true.mean()))

with col_pr:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rec_list, y=prec_list,
        mode="lines", name="Precision-Recall",
        line=dict(color="#10b981", width=2.5),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.15)",
    ))
    fig_pr.add_shape(type="line", x0=0, y0=base_rate, x1=1, y1=base_rate,
                     line=dict(color="#64748b", dash="dash", width=1.2))
    fig_pr.add_annotation(x=0.8, y=base_rate + 0.015, text=f"Baseline Rate {base_rate:.3f}",
                          showarrow=False, font=dict(color="#94a3b8", size=10))
    fig_pr.update_layout(
        **get_plotly_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=320,
            margin=dict(l=10, r=10, t=10, b=30),
        )
    )
    st.plotly_chart(fig_pr, use_container_width=True)

# ─── Row 3: Ablation Study ────────────────────────────────────────────────────
st.markdown("### 🔬 Ablation Study — Feature Robustness (Noise Injection Test)")
st.caption("K=0: Full active feature set; K>0: Injects K uninformative gaussian noise features to stress-test stability.")

col_abl1, col_abl2 = st.columns(2)

with col_abl1:
    fig_abl = go.Figure()
    fig_abl.add_trace(go.Scatter(
        x=ablation["K"], y=ablation["auc_pooled"],
        mode="lines+markers", name="AUC (Pooled)",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=8, color="#818cf8"),
    ))
    fig_abl.add_trace(go.Scatter(
        x=ablation["K"], y=ablation["auc_fold_mean"],
        mode="lines+markers", name="AUC (Fold Mean)",
        line=dict(color="#06b6d4", width=2, dash="dash"),
        marker=dict(size=8, color="#22d3ee"),
    ))
    fig_abl.update_layout(
        **get_plotly_layout(
            xaxis_title="K (Synthetic Noise Features Added)",
            yaxis_title="AUC",
            height=300,
            xaxis=dict(tickmode="array", tickvals=ablation["K"].tolist(), gridcolor="#1e263d"),
            yaxis=dict(range=[0.5, 0.7], gridcolor="#1e263d"),
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(orientation="h", y=1.05),
        )
    )
    st.plotly_chart(fig_abl, use_container_width=True)

with col_abl2:
    fig_abl2 = go.Figure()
    fig_abl2.add_trace(go.Bar(
        x=ablation["K"], y=ablation["precision_top20"],
        name="Top-20% Precision",
        marker_color="#8b5cf6",
        text=[f"{v:.4f}" for v in ablation["precision_top20"]],
        textposition="outside",
    ))
    fig_abl2.add_trace(go.Bar(
        x=ablation["K"], y=ablation["lift_top"],
        name="Top-20% Lift",
        marker_color="#38bdf8",
        text=[f"{v:.3f}×" for v in ablation["lift_top"]],
        textposition="outside",
    ))
    fig_abl2.update_layout(
        **get_plotly_layout(
            barmode="group",
            xaxis_title="K (Synthetic Noise Features Added)",
            yaxis_title="Metric Value",
            height=300,
            xaxis=dict(tickmode="array", tickvals=ablation["K"].tolist(), gridcolor="#1e263d"),
            yaxis=dict(gridcolor="#1e263d"),
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(orientation="h", y=1.05),
        )
    )
    st.plotly_chart(fig_abl2, use_container_width=True)

st.markdown("---")

# ─── Signal timeline ──────────────────────────────────────────────────────────
st.markdown("### 📅 Signal Volume Distribution Over Time")
sig_daily = (
    signals[signals["selected"] == True]
    .groupby("time")
    .size()
    .reset_index(name="n_signals")
)
fig_vol = go.Figure(go.Bar(
    x=sig_daily["time"], y=sig_daily["n_signals"],
    marker_color="#4f46e5", opacity=0.85, name="ML Selected Signals",
))
fig_vol.update_layout(
    **get_plotly_layout(
        xaxis_title="Date", yaxis_title="Daily Signals",
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
    )
)
st.plotly_chart(fig_vol, use_container_width=True)
