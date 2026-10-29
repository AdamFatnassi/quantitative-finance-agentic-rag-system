"""
evaluation.py — leakage-safe evaluation of the reversion classifier.

Pure functions, no I/O. Used by run_ablation.py (and importable from notebooks).

Design rules
------------
1. Only evaluate on ENTRY situations (|z| >= z_entry): those are the only days
   a pairs trader would actually act on. Scoring on all days is inflated by
   trivially-true labels (|z| already small -> "reverts below 0.5" is easy).
2. Walk-forward by window: test on window i, train only on EARLIER windows.
3. Purge: the label looks `horizon` bars ahead, so training rows within
   `horizon` business days of the test start are dropped (their labels would
   peek into the test period).
4. Every feature set is evaluated on the SAME rows and the SAME folds, so
   differences are attributable to the features alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

RAW_COLS = ["z_score", "dz_1", "spread_vol_20", "spread_skew_60", "spread_slope_5", "vol_ratio"]


def regime_cols(n_states: int) -> list[str]:
    return ["regime_state", "regime_entropy"] + [f"regime_p{k}" for k in range(n_states)]


@dataclass(frozen=True)
class EvalConfig:
    z_entry: float = 2.0            # only score rows where |z| >= this
    horizon: int = 10               # must match TargetConfig.horizon (purge width)
    min_train_windows: int = 3      # first fold needs at least this many earlier windows
    min_train_rows: int = 200
    min_test_rows: int = 30
    top_quantile: float = 0.2       # "trade the top 20% most confident signals"


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    """Deliberately small/regularised: the signal is weak, overfitting is the main risk."""
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200,
        l2_regularization=1.0, min_samples_leaf=50, random_state=seed,
    )


# ─────────────────────────────────────────────
# Data prep
# ─────────────────────────────────────────────

def prepare(df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """Keep labelled entry-situation rows only, sorted by time."""
    out = df[df["y"].notna() & (df["z_score"].abs() >= cfg.z_entry)].copy()
    return out.sort_values(["time", "stock_a", "stock_b"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# Purged walk-forward folds
# ─────────────────────────────────────────────

def walk_forward_folds(df: pd.DataFrame, cfg: EvalConfig) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (window_id, train_mask, test_mask). Train = earlier windows, purged near test start."""
    window_ids = sorted(df["window_id"].unique())
    for i, wid in enumerate(window_ids):
        if i < cfg.min_train_windows:
            continue
        test_mask = (df["window_id"] == wid).to_numpy()
        test_start = df.loc[test_mask, "time"].min()
        cutoff = test_start - pd.offsets.BDay(cfg.horizon)       # purge/embargo
        train_mask = ((df["window_id"] < wid) & (df["time"] < cutoff)).to_numpy()
        if train_mask.sum() < cfg.min_train_rows or test_mask.sum() < cfg.min_test_rows:
            continue
        yield wid, train_mask, test_mask


def out_of_sample_predictions(df: pd.DataFrame, feature_cols: list[str],
                              cfg: EvalConfig, seed: int = 0) -> pd.DataFrame:
    """Stack out-of-sample probabilities from every fold. Columns: window_id, y, p (+ row index)."""
    parts = []
    X_all, y_all = df[feature_cols], df["y"].astype(int)
    for wid, tr, te in walk_forward_folds(df, cfg):
        y_tr = y_all[tr]
        if y_tr.nunique() < 2:
            continue
        model = make_model(seed).fit(X_all[tr], y_tr)
        p = model.predict_proba(X_all[te])[:, 1]
        parts.append(pd.DataFrame({"window_id": wid, "y": y_all[te].to_numpy(), "p": p},
                                  index=df.index[te]))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["window_id", "y", "p"])


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def _safe_auc(y, p) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def per_fold_auc(oos: pd.DataFrame) -> pd.Series:
    return oos.groupby("window_id").apply(lambda g: _safe_auc(g["y"], g["p"]), include_groups=False)


def summarize(oos: pd.DataFrame, cfg: EvalConfig) -> dict:
    """Pooled + per-fold metrics. Precision@top-q is the trading-relevant one."""
    if oos.empty:
        return {}
    base = oos["y"].mean()
    thresh = oos["p"].quantile(1 - cfg.top_quantile)
    top = oos[oos["p"] >= thresh]
    fold_auc = per_fold_auc(oos).dropna()
    return {
        "n_oos": len(oos),
        "n_folds": oos["window_id"].nunique(),
        "base_rate": base,
        "auc_pooled": _safe_auc(oos["y"], oos["p"]),
        "auc_fold_mean": fold_auc.mean(),
        "auc_fold_std": fold_auc.std(),
        "brier": brier_score_loss(oos["y"], oos["p"]),
        "brier_baseline": base * (1 - base),             # brier of always predicting base rate
        f"precision_top{int(cfg.top_quantile * 100)}": top["y"].mean(),
        "lift_top": top["y"].mean() / base if base > 0 else np.nan,
    }


def paired_comparison(oos_a: pd.DataFrame, oos_b: pd.DataFrame) -> dict:
    """Compare B with A on matching prediction rows and folds."""
    if not oos_a.index.equals(oos_b.index):
        raise ValueError("Paired predictions must have identical row indexes.")
    paired = oos_a[["window_id", "y", "p"]].join(
        oos_b[["window_id", "y", "p"]], how="left", lsuffix="_a", rsuffix="_b",
    )
    if paired.empty:
        return {}
    if not (paired["window_id_a"].equals(paired["window_id_b"])
            and paired["y_a"].equals(paired["y_b"])):
        raise ValueError("Paired predictions must have matching folds and labels.")

    differences = {}
    for window_id, group in paired.groupby("window_id_a"):
        auc_a = _safe_auc(group["y_a"], group["p_a"])
        auc_b = _safe_auc(group["y_b"], group["p_b"])
        if np.isfinite(auc_a) and np.isfinite(auc_b):
            differences[window_id] = auc_b - auc_a
    d = pd.Series(differences, dtype=float)
    if d.empty:
        return {}
    return {
        "n_folds": len(d),
        "mean_auc_diff": d.mean(),
        "folds_improved": int((d > 0).sum()),
        "se": d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan,
    }
