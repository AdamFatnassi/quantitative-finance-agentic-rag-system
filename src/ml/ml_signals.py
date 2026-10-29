"""
ml_signals.py — pairs-trading reversion signal using raw spread features.

Question answered: "this spread is stretched (|z| >= z_entry) — will it revert
within `horizon` bars?"  The model ranks entry situations by P(revert); the
top slice becomes a trade signal.

Input
-----
Pair-metric rows from pairs_selection, labelled here within each pair/window.
The model uses six raw spread features only (no HMM — it did not help).

Leakage rules (all enforced here)
---------------------------------
1. Rows are restricted to entry situations (|z| >= z_entry), labelled rows only.
2. Walk-forward by window_id; training rows within `horizon` business days of the
   test start are purged (their labels look into the test period).
3. The scaler lives inside the sklearn Pipeline -> fit on training rows only.
4. The signal threshold for a fold comes ONLY from out-of-sample probabilities of
   EARLIER folds (expanding history). No signals until enough history exists.
   Nothing about a fold's own predictions or labels sets its threshold.

Usage
-----
    pair_metrics = pd.read_parquet("data/lake/analytics/pair_metrics.parquet")
    result = MLSignal(SignalConfig(model_type="logistic")).fit_predict(pair_metrics)
    print(result.metrics, result.signal_metrics)
    result.oos.to_parquet("signals.parquet")                 # input for the backtest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from diagnostics.evaluation import (RAW_COLS, EvalConfig, per_fold_auc, prepare, summarize,
                        walk_forward_folds)

ID_COLS = ["time", "window_id", "stock_a", "stock_b", "sector", "half_life",
           "pair_active", "z_score", "y"]


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

@dataclass
class SignalConfig:
    model_type: str = "logistic"                  # logistic | hgb | xgboost | lightgbm
    features: tuple = tuple(RAW_COLS)
    model_params: dict = field(default_factory=dict)   # overrides for the chosen model
    seed: int = 42
    # Signal selection: trade the top `top_quantile` of probabilities, where the
    # cut-off is the (1 - top_quantile) quantile of PAST out-of-sample probabilities.
    top_quantile: float = 0.2
    min_history_for_threshold: int = 500          # OOS rows needed before any signal
    # Evaluation / label settings (must match TargetConfig used to build the data)
    eval: EvalConfig = field(default_factory=EvalConfig)
    z_exit: float = 0.5


def build_signal_dataset(pair_metrics: pd.DataFrame, z_exit: float = 0.5,
                         horizon: int = 10) -> pd.DataFrame:
    """Attach forward reversion labels without crossing pair/window boundaries."""
    keys = ["window_id", "stock_a", "stock_b"]
    required = set(keys + ["time", "z_score"])
    missing = required - set(pair_metrics.columns)
    if missing:
        raise ValueError(f"pair metrics are missing signal columns: {sorted(missing)}")
    if z_exit <= 0 or horizon < 1:
        raise ValueError("z_exit must be positive and horizon must be >= 1")

    data = pair_metrics.copy()
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values(keys + ["time"]).copy()
    data["y"] = np.nan

    for _, group in data.groupby(keys, sort=False):
        absolute_z = group["z_score"].abs().to_numpy()
        active = (group["pair_active"].fillna(False).to_numpy(dtype=bool)
                  if "pair_active" in group else np.ones(len(group), dtype=bool))
        labels = np.full(len(group), np.nan)
        for pos in range(len(group)):
            if not active[pos]:
                continue
            stop = min(pos + horizon, len(group) - 1)
            for future_pos in range(pos + 1, stop + 1):
                if not active[future_pos]:
                    labels[pos] = 0.0
                    break
                if absolute_z[future_pos] < z_exit:
                    labels[pos] = 1.0
                    break
            else:
                if pos + horizon < len(group):
                    labels[pos] = 0.0
        data.loc[group.index, "y"] = labels

    return data.sort_values(["time", "stock_a", "stock_b"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. MODEL
# ─────────────────────────────────────────────

def build_pipeline(cfg: SignalConfig) -> Pipeline:
    """RobustScaler -> classifier. Small, regularised: the signal is weak, overfit is the risk."""
    p, seed = cfg.model_params, cfg.seed

    if cfg.model_type == "logistic":
        clf = LogisticRegression(**{"C": 0.1, "max_iter": 1000, "class_weight": "balanced", **p})
    elif cfg.model_type == "hgb":
        clf = HistGradientBoostingClassifier(**{
            "max_depth": 3, "learning_rate": 0.05, "max_iter": 200,
            "l2_regularization": 1.0, "min_samples_leaf": 50, "random_state": seed, **p})
    elif cfg.model_type == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**{
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 20, "reg_lambda": 5.0,
            "eval_metric": "logloss", "random_state": seed, "n_jobs": -1, **p})
    elif cfg.model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**{
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_samples": 50, "reg_lambda": 5.0,
            "random_state": seed, "n_jobs": -1, "verbose": -1, **p})
    else:
        raise ValueError(f"unknown model_type: {cfg.model_type}")

    return Pipeline([("scaler", RobustScaler()), ("clf", clf)])


# ─────────────────────────────────────────────
# 3. RESULT
# ─────────────────────────────────────────────

@dataclass
class SignalResult:
    """
    oos : one row per out-of-sample entry situation, columns
          time, window_id, stock_a, stock_b, z_score, y, proba, threshold, selected, signal
          signal: +1 = long spread (z < 0), -1 = short spread (z > 0), 0 = no trade.
    """
    oos: pd.DataFrame
    metrics: dict               # ranking quality of ALL oos rows (AUC, Brier, lift ...)
    signal_metrics: dict        # quality of the rows actually selected as signals
    fold_auc: pd.Series
    feature_importance: pd.DataFrame
    model: Pipeline             # final model fit on all labelled entry rows (for live use)
    config: SignalConfig


# ─────────────────────────────────────────────
# 4. SIGNAL ENGINE
# ─────────────────────────────────────────────

class MLSignal:
    def __init__(self, config: Optional[SignalConfig] = None):
        self.cfg = config or SignalConfig()

    def fit_predict(self, df: pd.DataFrame) -> SignalResult:
        cfg, ecfg = self.cfg, self.cfg.eval
        if "pair_active" not in df.columns:
            raise KeyError("input is missing required column: pair_active")
        if "y" not in df.columns:
            df = build_signal_dataset(df, z_exit=cfg.z_exit, horizon=ecfg.horizon)
        feats = list(cfg.features)
        missing = [c for c in feats + ["z_score", "y", "time", "window_id"] if c not in df.columns]
        if missing:
            raise KeyError(f"input is missing columns: {missing}")

        active_rows = df[df["pair_active"].fillna(False)].copy()
        data = prepare(active_rows.dropna(subset=feats), ecfg)  # active entry rows only
        if data.empty:
            raise RuntimeError("no active labelled entry rows are available for signal training")
        X, y = data[feats], data["y"].astype(int)
        print(f"{len(data):,} entry rows | base rate {y.mean():.2%} | "
              f"{data['window_id'].nunique()} windows | model={cfg.model_type}")

        # ── purged walk-forward, causal threshold ──
        parts, history = [], np.empty(0)
        for wid, tr, te in walk_forward_folds(data, ecfg):
            if y[tr].nunique() < 2:
                continue
            model = build_pipeline(cfg).fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])[:, 1]

            if len(history) > 0 and len(history) >= cfg.min_history_for_threshold:
                thr = float(np.quantile(history, 1 - cfg.top_quantile))   # earlier folds only
                selected = proba >= thr
            else:
                thr, selected = np.nan, np.zeros(len(proba), dtype=bool)

            fold = data.loc[te, [c for c in ID_COLS if c in data.columns]].copy()
            fold["proba"], fold["threshold"], fold["selected"] = proba, thr, selected
            parts.append(fold)
            history = np.concatenate([history, proba])                   # update AFTER use

        if not parts:
            raise RuntimeError("no valid folds — check min_train_windows / min_train_rows / data size")

        oos = pd.concat(parts).sort_values(["time", "stock_a", "stock_b"]).reset_index(drop=True)
        # z > 0: spread is high -> short the spread; z < 0 -> long the spread.
        # (Assumes z = (spread - mean)/std. Verify against your backtest's spread definition.)
        oos["signal"] = np.where(
            oos["selected"] & oos["pair_active"].fillna(False), -np.sign(oos["z_score"]), 0
        ).astype(int)

        # ── metrics ──
        scored = oos.rename(columns={"proba": "p"})
        metrics = summarize(scored[["window_id", "y", "p"]], ecfg)
        fold_auc = per_fold_auc(scored[["window_id", "y", "p"]])
        sel = oos[oos["selected"]]
        signal_metrics = {
            "n_signals": len(sel),
            "signal_precision": sel["y"].mean() if len(sel) else np.nan,
            "base_rate_same_period": oos.loc[oos["threshold"].notna(), "y"].mean(),
        }
        signal_metrics["signal_lift"] = (signal_metrics["signal_precision"]
                                         / signal_metrics["base_rate_same_period"]
                                         if len(sel) else np.nan)

        # ── final model for live inference (cannot be evaluated; do not report its score) ──
        final = build_pipeline(cfg).fit(X, y)
        imp = self._importance(final, feats)

        self._print(metrics, signal_metrics)
        return SignalResult(oos, metrics, signal_metrics, fold_auc, imp, final, cfg)

    # ── live inference on new rows (same features, already causal) ──
    @staticmethod
    def predict_proba(model: Pipeline, X_new: pd.DataFrame, features: tuple) -> pd.Series:
        return pd.Series(model.predict_proba(X_new[list(features)])[:, 1], index=X_new.index, name="proba")

    @staticmethod
    def _importance(model: Pipeline, feats: list[str]) -> pd.DataFrame:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])                      # on scaled features -> comparable
        elif hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        else:                                               # HistGradientBoosting: no built-in
            imp = np.full(len(feats), np.nan)
        return (pd.DataFrame({"feature": feats, "importance": imp})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    @staticmethod
    def _print(m: dict, s: dict) -> None:
        print(f"OOS: n={m['n_oos']:,} folds={m['n_folds']} base={m['base_rate']:.3f} "
              f"AUC={m['auc_pooled']:.3f} (fold mean {m['auc_fold_mean']:.3f} ± {m['auc_fold_std']:.3f}) "
              f"Brier={m['brier']:.4f} vs {m['brier_baseline']:.4f}")
        print(f"Signals: n={s['n_signals']:,} precision={s['signal_precision']:.3f} "
              f"vs base {s['base_rate_same_period']:.3f} -> lift {s['signal_lift']:.2f}")


# ─────────────────────────────────────────────
# 5. DIAGNOSTICS & PERSISTENCE
# ─────────────────────────────────────────────

def decile_table(result: SignalResult) -> pd.DataFrame:
    """Hit rate by predicted-probability decile. A useful model is monotonic: top deciles > bottom."""
    o = result.oos
    o = o.assign(decile=pd.qcut(o["proba"], 10, labels=False, duplicates="drop"))
    t = o.groupby("decile").agg(n=("y", "size"), mean_proba=("proba", "mean"), hit_rate=("y", "mean"))
    t["lift"] = t["hit_rate"] / o["y"].mean()
    return t


def plot_diagnostics(result: SignalResult):
    """Fold-by-fold AUC, hit rate by decile, and calibration. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    fa = result.fold_auc.dropna()
    ax[0].bar(range(len(fa)), fa.values, color=np.where(fa.values >= 0.5, "#4CAF50", "#E53935"))
    ax[0].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[0].set(title="AUC per walk-forward fold", xlabel="fold", ylabel="AUC")

    dt = decile_table(result)
    ax[1].bar(dt.index, dt["hit_rate"], color="#2196F3")
    ax[1].axhline(result.oos["y"].mean(), color="k", lw=0.8, ls="--", label="base rate")
    ax[1].set(title="Reversion rate by probability decile", xlabel="decile (9 = most confident)")
    ax[1].legend()

    ax[2].plot(dt["mean_proba"], dt["hit_rate"], "s-", color="#9C27B0")
    lim = [0, max(dt["mean_proba"].max(), dt["hit_rate"].max()) * 1.1]
    ax[2].plot(lim, lim, "k--", lw=0.8)
    ax[2].set(title="Calibration (by decile)", xlabel="mean predicted", ylabel="observed")

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_result(result: SignalResult, output_dir: str = "results/ml_signal") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.oos.to_parquet(out / "signals_oos.parquet", index=False)
    result.feature_importance.to_csv(out / "feature_importance.csv", index=False)
    joblib.dump(result.model, out / "model_final.pkl")
    pd.Series({**result.metrics, **result.signal_metrics}).to_csv(out / "metrics.csv")
    print(f"saved -> {out}/")