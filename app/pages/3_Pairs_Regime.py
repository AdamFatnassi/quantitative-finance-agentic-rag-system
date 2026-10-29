"""
3_Pairs_Regime.py — HMM Regime Classification Dashboard
Visualizes latent market regime states, soft probabilities,
entropy, and pair-level overlays.
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
from theme import apply_theme, render_header, render_metric, get_plotly_layout, hex_to_rgba

st.set_page_config(page_title="Pairs & Regime", page_icon="🌊", layout="wide")
apply_theme()

ROOT      = Path(__file__).parent.parent.parent
ANALYTICS = ROOT / "data" / "lake" / "analytics"

REGIME_COLORS = {0: "#4f46e5", 1: "#06b6d4", 2: "#f43f5e"}
REGIME_NAMES  = {0: "State 0 (Low Vol / Mean-Reverting)", 1: "State 1 (Trending / Momentum)", 2: "State 2 (High Stress / Volatile)"}


@st.cache_data(ttl=600)
def load_regime_data():
    regimes_path = ANALYTICS / "assets" / "hmm_regimes.parquet"
    if not regimes_path.exists():
        regimes_path = ANALYTICS / "hmm_regimes.parquet"
    df = pd.read_parquet(regimes_path)
    df["time"] = pd.to_datetime(df["time"])
    df["pair"] = df["stock_a"] + " / " + df["stock_b"]
    return df


@st.cache_data(ttl=600)
def load_pairs_meta():
    pairs_path = ANALYTICS / "assets" / "pairs.parquet"
    if not pairs_path.exists():
        pairs_path = ANALYTICS / "pairs.parquet"
    pairs = pd.read_parquet(pairs_path)
    return pairs


df = load_regime_data()
pairs_meta = load_pairs_meta()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌊 Regime Explorer")

all_pairs = sorted(df["pair"].unique())
sel_pair  = st.sidebar.selectbox("Select Pair", all_pairs, index=0)
sel_window = st.sidebar.selectbox(
    "Walk-Forward Window",
    sorted(df["window_id"].unique()),
    index=int(df["window_id"].max()),
)
show_entropy  = st.sidebar.checkbox("Show Regime Entropy", value=True)
show_soft     = st.sidebar.checkbox("Show Soft Probabilities", value=True)
rolling_days  = st.sidebar.slider("Regime Distribution Rolling Window (days)", 30, 365, 90)

# Filter to selected pair
pair_df = df[df["pair"] == sel_pair].sort_values("time")

# ─── Header ────────────────────────────────────────────────────────────────────
render_header(
    title=f"HMM Market Regimes — {sel_pair}",
    subtitle="Gaussian Hidden Markov Model (3 latent states) fitted on z-score, spread volatility, momentum, and skew.",
    icon="🌊",
    badge="REGIME ENGINE",
)

# ─── Global KPIs ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
dominant_state = int(pair_df["regime_state"].mode().iloc[0])
with col1:
    st.markdown(render_metric("Observations", f"{len(pair_df):,}", "Historical days", "indigo", "📊"), unsafe_allow_html=True)
with col2:
    st.markdown(render_metric("Date Span", f"{str(pair_df['time'].min())[:7]} → {str(pair_df['time'].max())[:7]}", "Active history", "neutral", "📅"), unsafe_allow_html=True)
with col3:
    st.markdown(render_metric("Dominant Regime", f"State {dominant_state}", REGIME_NAMES[dominant_state].split('(')[1].rstrip(')'), "good", "🎯"), unsafe_allow_html=True)
with col4:
    st.markdown(render_metric("Mean Entropy", f"{pair_df['regime_entropy'].mean():.4f}", "State uncertainty", "amber", "⚖️"), unsafe_allow_html=True)
with col5:
    st.markdown(render_metric("Regime Switches", f"{(pair_df['regime_state'].diff().ne(0)).sum():,}", "State transitions", "cyan", "🔄"), unsafe_allow_html=True)

st.markdown("---")

# ─── Main Chart: Z-Score with Regime Coloring ─────────────────────────────────
st.markdown("### 📈 Z-Score Time Series with HMM Regime Shading")

fig_main = go.Figure()

# Background shading by regime
for state in [0, 1, 2]:
    seg = pair_df[pair_df["regime_state"].round().astype(int) == state]
    if seg.empty:
        continue
    seg_copy = seg.copy()
    seg_copy["block"] = (seg_copy["time"].diff() > pd.Timedelta("5D")).cumsum()
    for _, blk in seg_copy.groupby("block"):
        fig_main.add_vrect(
            x0=blk["time"].iloc[0], x1=blk["time"].iloc[-1],
            fillcolor=REGIME_COLORS[state], opacity=0.14,
            layer="below", line_width=0,
        )

# Entry / exit thresholds
for level, color, dash in [(2.0, "#38bdf8", "dash"), (-2.0, "#38bdf8", "dash"),
                            (0.5, "#34d399", "dot"), (-0.5, "#34d399", "dot"),
                            (4.0, "#f87171", "longdash"), (-4.0, "#f87171", "longdash")]:
    fig_main.add_hline(y=level, line_color=color, line_dash=dash, line_width=1, opacity=0.6)

# Z-score line
fig_main.add_trace(go.Scatter(
    x=pair_df["time"], y=pair_df["z_score"],
    mode="lines", name="Z-Score",
    line=dict(color="#f8fafc", width=1.5),
))

# Regime state scatter (colored dots)
for state in [0, 1, 2]:
    seg = pair_df[pair_df["regime_state"].round().astype(int) == state]
    fig_main.add_trace(go.Scatter(
        x=seg["time"], y=seg["z_score"],
        mode="markers", name=REGIME_NAMES[state].split(' ')[0] + f" ({state})",
        marker=dict(color=REGIME_COLORS[state], size=3.5, opacity=0.7),
    ))

fig_main.update_layout(
    **get_plotly_layout(
        xaxis_title="Date", yaxis_title="Spread Z-Score",
        height=420,
        margin=dict(l=10, r=10, t=30, b=30),
    )
)
st.plotly_chart(fig_main, use_container_width=True)

# ─── Soft Probabilities & Entropy ─────────────────────────────────────────────
if show_soft or show_entropy:
    n_rows = (1 if show_soft else 0) + (1 if show_entropy else 0)
    fig_soft = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            (["Soft Regime Posterior Probabilities (Stacked Area)"] if show_soft else []) +
            (["Regime Entropy (Uncertainty Metric)"] if show_entropy else [])
        ),
        vertical_spacing=0.1,
    )
    row = 1

    if show_soft:
        for state, col_name in [(0, "regime_p0"), (1, "regime_p1"), (2, "regime_p2")]:
            fig_soft.add_trace(go.Scatter(
                x=pair_df["time"], y=pair_df[col_name],
                mode="lines", name=REGIME_NAMES[state].split('(')[0].strip(),
                stackgroup="one",
                line=dict(color=REGIME_COLORS[state], width=0.5),
                fillcolor=hex_to_rgba(REGIME_COLORS[state], 0.45),
            ), row=row, col=1)
        fig_soft.update_yaxes(title_text="Probability", range=[0, 1], row=row, col=1)
        row += 1

    if show_entropy:
        fig_soft.add_trace(go.Scatter(
            x=pair_df["time"], y=pair_df["regime_entropy"],
            mode="lines", name="Entropy",
            line=dict(color="#f59e0b", width=1.5),
            fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.15)",
            showlegend=True,
        ), row=row, col=1)
        fig_soft.update_yaxes(title_text="Entropy", row=row, col=1)

    fig_soft.update_layout(
        **get_plotly_layout(
            height=360 * n_rows // 2 + 100,
            margin=dict(l=10, r=10, t=50, b=30),
        )
    )
    st.plotly_chart(fig_soft, use_container_width=True)

# ─── Regime Distribution Over Time ────────────────────────────────────────────
st.markdown(f"### 🌊 Rolling Regime Distribution ({rolling_days}d window, all pairs)")

global_roll = (
    df.sort_values("time")
    .groupby("time")["regime_state"]
    .apply(lambda x: x.round().astype(int).value_counts(normalize=True))
    .unstack(fill_value=0)
)
if not isinstance(global_roll.index, pd.DatetimeIndex):
    global_roll.index = pd.to_datetime(global_roll.index)

roll_dist = global_roll.rolling(rolling_days).mean().dropna()

fig_roll = go.Figure()
for state in [0, 1, 2]:
    if state in roll_dist.columns:
        fig_roll.add_trace(go.Scatter(
            x=roll_dist.index, y=roll_dist[state],
            mode="lines", name=REGIME_NAMES[state].split('(')[0].strip(),
            stackgroup="one",
            line=dict(color=REGIME_COLORS[state], width=0.5),
            fillcolor=hex_to_rgba(REGIME_COLORS[state], 0.5),
        ))
fig_roll.update_layout(
    **get_plotly_layout(
        xaxis_title="Date", yaxis_title="Regime Share",
        yaxis=dict(range=[0, 1], gridcolor="#1e263d"),
        height=320,
        margin=dict(l=10, r=10, t=30, b=30),
    )
)
st.plotly_chart(fig_roll, use_container_width=True)

# ─── Regime Transition Heatmap & Sector Distribution ──────────────────────────
col_hm, col_sector = st.columns(2)

with col_hm:
    st.markdown("### 🔄 Regime Transition Probability Matrix")
    states = pair_df["regime_state"].round().astype(int)
    trans_mat = np.zeros((3, 3))
    for i, j in zip(states.values[:-1], states.values[1:]):
        if 0 <= i <= 2 and 0 <= j <= 2:
            trans_mat[i, j] += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    trans_prob = np.divide(trans_mat, row_sums, out=np.zeros_like(trans_mat), where=row_sums != 0)

    fig_hm = go.Figure(go.Heatmap(
        z=trans_prob,
        x=[f"To State {i}" for i in range(3)],
        y=[f"From State {i}" for i in range(3)],
        colorscale="Blues",
        text=[[f"{v:.3f}" for v in r] for r in trans_prob],
        texttemplate="%{text}",
        hovertemplate="From %{y}<br>To %{x}<br>Transition Prob = %{z:.4f}<extra></extra>",
    ))
    fig_hm.update_layout(
        **get_plotly_layout(
            height=300, margin=dict(l=10, r=10, t=20, b=10),
        )
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with col_sector:
    st.markdown("### 🏛️ Regime Distribution by Sector")
    if "sector" in df.columns:
        sec_regime = (
            df.groupby(["sector", df["regime_state"].round().astype(int).rename("state")])
            .size()
            .unstack(fill_value=0)
        )
        sec_regime_pct = sec_regime.div(sec_regime.sum(axis=1), axis=0)
        fig_sec = go.Figure()
        for state in [0, 1, 2]:
            if state in sec_regime_pct.columns:
                fig_sec.add_trace(go.Bar(
                    x=sec_regime_pct.index,
                    y=sec_regime_pct[state],
                    name=f"State {state}",
                    marker_color=REGIME_COLORS[state],
                ))
        fig_sec.update_layout(
            **get_plotly_layout(
                barmode="stack", xaxis_title="Sector", yaxis_title="Regime Share",
                height=300, margin=dict(l=10, r=10, t=20, b=60),
                xaxis=dict(tickangle=-30, gridcolor="#1e263d"),
                yaxis=dict(range=[0, 1], gridcolor="#1e263d"),
            )
        )
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info("Sector data not available in regime dataset.")

# ─── Feature Distributions by Regime ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Distribution of Feature Attributes by Latent Regime")
feats = ["z_score", "spread_vol_20", "rolling_half_life", "spread_slope_5"]
feat_names = ["Spread Z-Score", "Spread Volatility (20d)", "Rolling Half-Life (days)", "Spread Slope (5d)"]

fig_box = make_subplots(rows=1, cols=4, subplot_titles=feat_names)
for i, (feat, fname) in enumerate(zip(feats, feat_names), 1):
    for state in [0, 1, 2]:
        seg = pair_df[pair_df["regime_state"].round().astype(int) == state]
        fig_box.add_trace(go.Box(
            y=seg[feat].dropna(),
            name=f"State {state}",
            marker_color=REGIME_COLORS[state],
            showlegend=(i == 1),
        ), row=1, col=i)

fig_box.update_layout(
    **get_plotly_layout(
        height=360, boxmode="group",
        margin=dict(l=10, r=10, t=40, b=30),
    )
)
st.plotly_chart(fig_box, use_container_width=True)
