from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant.pairs_selection import PairScreenConfig, WalkForwardOrchestrator


def _pair_key(stock_a: str, stock_b: str) -> str:
    return f"{stock_a}/{stock_b}"


def _safe_var(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.var(ddof=1))


def _compute_beta_drift(repo_root: Path) -> pd.DataFrame:
    prices_path = repo_root / "data" / "lake" / "processed" / "assets" / "prices_clean.parquet"
    if not prices_path.exists():
        prices_path = repo_root / "data" / "lake" / "processed" / "prices_clean.parquet"
    prices = pd.read_parquet(prices_path)
    pairs_path = repo_root / "data" / "lake" / "analytics" / "assets" / "pairs.parquet"
    if not pairs_path.exists():
        pairs_path = repo_root / "data" / "lake" / "analytics" / "pairs.parquet"
    pairs = pd.read_parquet(pairs_path)
    config = PairScreenConfig()
    orchestrator = WalkForwardOrchestrator(config)

    rows: list[dict[str, float | int | str]] = []
    for window in orchestrator.iter_windows(prices.index):
        pairs_w = pairs[pairs["window_id"] == window.window_id].copy()
        if pairs_w.empty:
            continue

        history = prices.iloc[window.pos_start : window.pos_end + 1]
        wide = orchestrator.engine.compute_wide(history, pairs_w)
        wide.pop("_active_today", None)

        trading_mask = history.index > window.t0
        if not trading_mask.any():
            continue

        trading_dates = history.index[trading_mask]
        for _, pair_row in pairs_w.iterrows():
            stock_a = str(pair_row["stock_a"])
            stock_b = str(pair_row["stock_b"])
            key = _pair_key(stock_a, stock_b)
            if key not in wide["hedge_ratio"].columns:
                continue

            beta = wide["hedge_ratio"][key].loc[trading_dates]
            spread = wide["spread"][key].loc[trading_dates]
            z_roll = wide["z_score"][key].loc[trading_dates]

            if beta.empty or spread.empty or z_roll.empty:
                continue

            Y = np.log(history[stock_a]).loc[trading_dates]
            X = np.log(history[stock_b]).loc[trading_dates]
            ds_total = spread.diff().dropna()
            ds_price = (Y.diff() - beta.shift(1) * X.diff()).loc[ds_total.index].dropna()
            common_idx = ds_total.index.intersection(ds_price.index)
            if common_idx.empty:
                beta_share = np.nan
            else:
                ds_total2 = ds_total.loc[common_idx]
                ds_price2 = ds_price.loc[common_idx]
                ds_beta = ds_total2 - ds_price2
                total_var = _safe_var(ds_total2)
                beta_share = np.nan if not np.isfinite(total_var) or np.isclose(total_var, 0.0) else _safe_var(ds_beta) / total_var

            spread_f = Y - float(pair_row["hedge_ratio"]) * X
            z_frozen = (spread_f - float(pair_row["intercept"])) / float(pair_row["spread_std"])
            corr_z = z_roll.corr(z_frozen)
            if np.isnan(corr_z):
                corr_z = np.nan

            abs_z_roll = z_roll.abs()
            entry_mask = abs_z_roll >= 2.0
            if entry_mask.any():
                phantom_entry = z_frozen.loc[entry_mask].abs() < 1.5
                phantom_entry_share = float(phantom_entry.mean()) if entry_mask.any() else np.nan
            else:
                phantom_entry_share = np.nan

            reversion_events = 0
            reversion_matches = 0
            for idx in range(len(z_roll) - 10):
                if abs_z_roll.iloc[idx] < 2.0:
                    continue
                future = z_roll.iloc[idx + 1 : idx + 11]
                rev_idx = future.index[future.abs() < 0.5]
                if rev_idx.empty:
                    continue
                reversion_events += 1
                reversion_bar = rev_idx[0]
                if abs(z_frozen.loc[reversion_bar]) >= 1.5:
                    reversion_matches += 1
            phantom_reversion_share = (reversion_matches / reversion_events) if reversion_events else np.nan

            rows.append({
                "window_id": int(window.window_id),
                "pair": key,
                "n_rows": int(len(trading_dates)),
                "beta_share": float(beta_share) if np.isfinite(beta_share) else np.nan,
                "corr_z": float(corr_z) if np.isfinite(corr_z) else np.nan,
                "phantom_entry_share": float(phantom_entry_share) if np.isfinite(phantom_entry_share) else np.nan,
                "phantom_reversion_share": float(phantom_reversion_share) if np.isfinite(phantom_reversion_share) else np.nan,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["window_id", "pair", "n_rows", "beta_share", "corr_z", "phantom_entry_share", "phantom_reversion_share"])
    return out.sort_values(["window_id", "pair"]).reset_index(drop=True)


def _print_beta_summary(df: pd.DataFrame) -> None:
    print(f"pair_windows={len(df)}")
    if df.empty:
        print("metric median mean p90")
        return
    metrics = ["beta_share", "corr_z", "phantom_entry_share", "phantom_reversion_share"]
    summary_rows = []
    for metric in metrics:
        vals = df[metric].dropna()
        if vals.empty:
            continue
        summary_rows.append({
            "metric": metric,
            "median": float(vals.median()),
            "mean": float(vals.mean()),
            "p90": float(vals.quantile(0.9)),
        })
    if not summary_rows:
        print("metric median mean p90")
        return
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    out_dir = ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _compute_beta_drift(ROOT)
    df.to_csv(out_dir / "beta_drift.csv", index=False)
    _print_beta_summary(df)


if __name__ == "__main__":
    main()
