"""
backtest.py — does the ML filter make money after costs?

Compares, with IDENTICAL trade logic, on the SAME out-of-sample period:
    "ML filter"    : enter only on signals chosen by ml_signal (top-q probabilities)
    "All entries"  : enter on every eligible |z| >= z_entry situation (no filter)
    "Random match" : random entries, same count as the ML filter (the null test —
                     is the ML selection better than picking the same number blindly?)

Trade logic (all decisions use information available at the close of the decision day)
    signal at close of day i  ->  enter at close of day i+1
    exit decision at close of day k when:  |z| < z_exit (revert)
                                       or  |z| >= z_stop on the adverse side (stop)
                                       or  held >= max_hold bars (time stop)
                                       or  end of the walk-forward window (forced)
    exit fills at close of day k+1.  One open position per pair at a time.

P&L model (stated simplifications — adjust if your live implementation differs)
        * Spread = A - beta*B in log terms, gross notional 1 per trade:
            weights (A, B) = side * (1, -beta) / (1 + |beta|), using the rolling beta known
            at the signal close and held fixed for that trade.
    * Costs: `cost_bps` per side on gross notional (commission + slippage + half-spread),
      charged at entry and exit; borrow fee `borrow_bps_annual` on the short leg per day.
    * Portfolio: each trade gets 1/max_positions of capital; unused slots earn 0.

Inputs
    signals : ml_signal output  (signals_oos.parquet)
    dataset : pipeline pair_metrics.parquet — daily z_score, hedge_ratio, and pair state
    prices  : pipeline processed/prices_clean.parquet — wide adjusted prices

Usage
    python -m quant.backtest --output-dir data/lake
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

@dataclass
class BTConfig:
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    max_hold: int = 20               # bars (label horizon is 10; allow some slack)
    cost_bps: float = 5.0            # per side, per unit gross notional
    borrow_bps_annual: float = 50.0  # on the short leg
    max_positions: int = 50          # capital is split into this many equal slots
    entry_delay: int = 1             # >= 1 (0 would trade on the same close the signal uses)
    exit_delay: int = 1
    n_random: int = 30               # random-matched draws for the null test
    seed: int = 0

    def __post_init__(self):
        if self.entry_delay < 1:
            raise ValueError("entry_delay must be >= 1 (0 = lookahead)")
        if self.exit_delay < 1:
            raise ValueError("exit_delay must be >= 1 (0 = lookahead)")
        if self.max_hold < 1 or self.max_positions < 1 or self.n_random < 1:
            raise ValueError("max_hold, max_positions, and n_random must be >= 1")
        if self.z_entry <= self.z_exit or self.z_exit <= 0 or self.z_stop <= self.z_exit:
            raise ValueError("z_stop must be greater than z_exit, and z_exit must be positive")
        if self.cost_bps < 0 or self.borrow_bps_annual < 0:
            raise ValueError("transaction costs and borrow fees cannot be negative")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily_ret: pd.Series
    stats: dict


# ─────────────────────────────────────────────
# 2. LOCAL PAIR RESILIENCE
# ─────────────────────────────────────────────


def _pair_data_paths(root: Optional[Path] = None) -> tuple[Path, Path]:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    dataset = base / "data" / "lake" / "analytics" / "assets" / "pair_metrics.parquet"
    if not dataset.exists():
        dataset = base / "data" / "lake" / "analytics" / "pair_metrics.parquet"
    prices = base / "data" / "lake" / "processed" / "assets" / "prices_clean.parquet"
    if not prices.exists():
        prices = base / "data" / "lake" / "processed" / "prices_clean.parquet"
    return dataset, prices


def _resolve_pair_row(dataset: pd.DataFrame, pair_id: str) -> pd.DataFrame:
    if dataset.empty:
        return dataset.copy()

    pair_key = str(pair_id).strip()
    if pair_key == "":
        return pd.DataFrame(columns=dataset.columns)

    numeric = pd.to_numeric(pair_key, errors="coerce")
    if pd.notna(numeric) and "pair_id" in dataset.columns:
        return dataset[dataset["pair_id"].astype(str) == str(int(numeric))].copy()

    ticker_pairs = (
        dataset["stock_a"].astype(str) + "/" + dataset["stock_b"].astype(str)
    )
    alt_pairs = (
        dataset["stock_a"].astype(str) + "_" + dataset["stock_b"].astype(str)
    )
    mask = ticker_pairs.eq(pair_key) | alt_pairs.eq(pair_key)
    if "pair_name" in dataset.columns:
        mask |= dataset["pair_name"].astype(str).eq(pair_key)
    return dataset.loc[mask].copy()


def check_pair_resilience(pair_id: str, lookback_days: int = 365) -> str:
    """Localized pair backtest summary for a single pair.

    The function is intentionally lightweight: it loads the project's local parquet
    artifacts, filters the pair to a trailing lookback window, runs the same
    backtesting logic as the global run, and returns a compact summary string with
    historical Sharpe, max drawdown, and breakdown rate.

    Expected pair identifiers:
      * integer-like `pair_id` values from the pipeline DB metadata, or
      * ticker-form strings such as `AAPL/MSFT` or `AAPL_MSFT`.
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    try:
        dataset_path, prices_path = _pair_data_paths()
        if not dataset_path.exists() or not prices_path.exists():
            return (
                f"Pair resilience unavailable: expected local artifacts not found at "
                f"{dataset_path} and {prices_path}."
            )
        dataset = pd.read_parquet(dataset_path)
        prices = pd.read_parquet(prices_path)
    except Exception as exc:  # pragma: no cover - local artifact loading is environment-specific
        return f"Pair resilience unavailable: could not load local backtest data ({exc})."

    pair_df = _resolve_pair_row(dataset, pair_id)
    if pair_df.empty:
        return f"No local pair data found for '{pair_id}'."

    if "time" not in pair_df.columns:
        return f"Pair '{pair_id}' is missing a time column in local pair_metrics."

    latest_time = pd.to_datetime(pair_df["time"]).max()
    min_time = latest_time - pd.Timedelta(days=lookback_days)
    local = pair_df[pd.to_datetime(pair_df["time"]) >= min_time].copy()
    if local.empty:
        local = pair_df.copy()

    required = {"time", "window_id", "stock_a", "stock_b", "z_score", "pair_active"}
    missing = sorted(required - set(local.columns))
    if missing:
        return (
            f"Pair '{pair_id}' is missing required local backtest columns: {', '.join(missing)}."
        )

    if "hedge_ratio" not in local.columns:
        return f"Pair '{pair_id}' has no hedge_ratio in local pair_metrics; global backtest cannot run."
    if not local["hedge_ratio"].notna().any():
        return f"Pair '{pair_id}' has no valid hedge_ratio to backtest with."

    bt = PairsBacktester(prices, local, config=BTConfig())
    eligible = local[
        local["pair_active"].fillna(False)
        & local["z_score"].abs().ge(bt.cfg.z_entry)
    ].copy()
    if eligible.empty:
        return f"Pair '{pair_id}' has no active local observations in the trailing {lookback_days}-day window."

    eval_period = (pd.Timestamp(eligible["time"].min()), pd.Timestamp(eligible["time"].max()))
    result = bt.run(eligible, evaluation_period=eval_period)
    if result.trades.empty:
        return (
            f"Pair '{pair_id}' has no valid local trades in the last {lookback_days} days; "
            "historical Sharpe = n/a, max drawdown = n/a, breakdown rate = n/a."
        )

    stats = result.stats
    sharpe = float(stats.get("sharpe", np.nan))
    max_dd = float(stats.get("max_drawdown", np.nan))
    breakdown = float(stats.get("pct_pair_breakdown", np.nan))
    return (
        f"Pair {pair_id}: sharpe={sharpe:.2f}, max_drawdown={max_dd:.2%}, "
        f"breakdown_rate={breakdown:.2%}, trades={stats.get('n_trades', 0)}, "
        f"mean_net_bps={stats.get('mean_net_bps', np.nan):.1f}, win_rate={stats.get('win_rate', np.nan):.1%}."
    )


# ─────────────────────────────────────────────
# 3. BACKTESTER
# ─────────────────────────────────────────────

class PairsBacktester:
    def __init__(self, prices: pd.DataFrame, dataset: pd.DataFrame,
                 config: BTConfig = BTConfig()):
        self.cfg = config
        if "hedge_ratio" not in dataset.columns:
            raise ValueError("pair_metrics must include hedge_ratio used to compute z_score")

        # Per (window, pair): aligned daily z-score, its rolling hedge ratio, prices, and active state.
        self.paths = {}
        for (wid, a, b), g in dataset.sort_values("time").groupby(["window_id", "stock_a", "stock_b"]):
            if a not in prices.columns or b not in prices.columns:
                continue
            t = pd.DatetimeIndex(g["time"])
            active = (g["pair_active"].fillna(False).to_numpy()
                      if "pair_active" in g else np.ones(len(g), dtype=bool))
            path_data = {
                "z": g["z_score"].to_numpy(),
                "active": active,
                "pa": prices[a].reindex(t).to_numpy(),
                "pb": prices[b].reindex(t).to_numpy(),
            }
            path_data["beta"] = g["hedge_ratio"].to_numpy()
            df = pd.DataFrame(path_data, index=t).dropna()
            if len(df) > 2:
                self.paths[(wid, a, b)] = df
        self.calendar = pd.DatetimeIndex(sorted(dataset["time"].unique()))

    # ── single trade ──
    def _simulate(self, key, i: int, side: int, cfg: BTConfig):
        path = self.paths[key]
        t, z, pa, pb = path.index, path["z"].to_numpy(), path["pa"].to_numpy(), path["pb"].to_numpy()
        n = len(z)
        e = i + cfg.entry_delay
        if e >= n - 1:
            return None

        entry_sign, d, reason = np.sign(z[i]), n - 1, "end"
        if not path["active"].iloc[e]:
            d, reason = e, "pair_breakdown"
        else:
            for k in range(e + 1, n):
                if not path["active"].iloc[k]:
                    d, reason = k, "pair_breakdown"; break
                if abs(z[k]) < cfg.z_exit:
                    d, reason = k, "revert"; break
                if abs(z[k]) >= cfg.z_stop and np.sign(z[k]) == entry_sign:
                    d, reason = k, "stop"; break
                if k - e >= cfg.max_hold:
                    d, reason = k, "time"; break
        f = n - 1 if reason == "end" else min(d + cfg.exit_delay, n - 1)

        beta = float(path["beta"].iloc[i])
        wa, wb = side / (1 + abs(beta)), -side * beta / (1 + abs(beta))
        ra = pa[e + 1: f + 1] / pa[e: f] - 1
        rb = pb[e + 1: f + 1] / pb[e: f] - 1
        gross = wa * ra + wb * rb
        short_w = -(min(wa, 0) + min(wb, 0))
        borrow = short_w * cfg.borrow_bps_annual / 1e4 / 252

        daily = np.zeros(f - e + 1)
        daily[1:] = gross - borrow
        daily[0] -= cfg.cost_bps / 1e4                    # entry cost (gross notional = 1)
        daily[-1] -= cfg.cost_bps / 1e4                   # exit cost
        return {"pair": f"{key[1]}/{key[2]}", "window_id": key[0], "side": side, "beta": beta,
                "entry_time": t[e], "exit_time": t[f], "hold": f - e, "reason": reason,
                "gross": gross.sum(), "net": daily.sum()}, pd.Series(daily, index=t[e: f + 1])

    # ── run a set of candidate entries ──
    def run(self, cand: pd.DataFrame, cost_bps: Optional[float] = None,
            evaluation_period: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None) -> BacktestResult:
        cfg = self.cfg if cost_bps is None else replace(self.cfg, cost_bps=cost_bps)
        if "pair_active" in cand.columns:
            cand = cand[cand["pair_active"].fillna(False)]
        cand = cand.assign(side=-np.sign(cand["z_score"]).astype(int),
                           abs_z=cand["z_score"].abs())
        cand = cand.sort_values(["time", "abs_z"], ascending=[True, False])

        trades, series, busy_until, open_exits = [], [], {}, []
        skipped_busy = skipped_cap = 0
        for r in cand.itertuples():
            key = (r.window_id, r.stock_a, r.stock_b)
            path = self.paths.get(key)
            if path is None or r.side == 0:
                continue
            try:
                i = path.index.get_loc(r.time)
            except KeyError:
                continue
            if i + cfg.entry_delay >= len(path) - 1:
                continue
            e_time = path.index[i + cfg.entry_delay]
            if busy_until.get((r.stock_a, r.stock_b), pd.Timestamp.min) > e_time:
                skipped_busy += 1; continue
            open_exits = [x for x in open_exits if x > e_time]
            if len(open_exits) >= cfg.max_positions:
                skipped_cap += 1; continue

            out = self._simulate(key, i, int(r.side), cfg)
            if out is None:
                continue
            tr, s = out
            trades.append(tr); series.append(s)
            busy_until[(r.stock_a, r.stock_b)] = tr["exit_time"]
            open_exits.append(tr["exit_time"])

        trades = pd.DataFrame(trades)
        if trades.empty:
            return BacktestResult(trades, pd.Series(dtype=float), {"n_trades": 0})

        if evaluation_period is None:
            lo, hi = cand["time"].min(), trades["exit_time"].max()
        else:
            lo, hi = map(pd.Timestamp, evaluation_period)
        idx = self.calendar[(self.calendar >= lo) & (self.calendar <= hi)]
        pnl = pd.concat(series).groupby(level=0).sum().reindex(idx, fill_value=0.0)
        n_open = pd.concat([pd.Series(1, index=s.index) for s in series]).groupby(level=0).sum().reindex(idx, fill_value=0)
        daily_ret = pnl / cfg.max_positions
        stats = self._stats(trades, daily_ret, n_open)
        stats.update(skipped_busy=skipped_busy, skipped_capacity=skipped_cap)
        return BacktestResult(trades, daily_ret, stats)

    @staticmethod
    def _stats(trades: pd.DataFrame, r: pd.Series, n_open: pd.Series) -> dict:
        eq = (1 + r).cumprod()
        net = trades["net"]
        wins, losses = net[net > 0].sum(), -net[net < 0].sum()
        return {
            "n_trades": len(trades),
            "win_rate": (net > 0).mean(),
            "mean_net_bps": net.mean() * 1e4,
            "mean_gross_bps": trades["gross"].mean() * 1e4,
            "median_net_bps": net.median() * 1e4,
            "profit_factor": wins / losses if losses > 0 else np.inf,
            "avg_hold": trades["hold"].mean(),
            "pct_revert": (trades["reason"] == "revert").mean(),
            "pct_stop": (trades["reason"] == "stop").mean(),
            "pct_pair_breakdown": (trades["reason"] == "pair_breakdown").mean(),
            "ann_return": r.mean() * 252,
            "ann_vol": r.std() * np.sqrt(252),
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
            "avg_open": n_open.mean(),
        }


# ─────────────────────────────────────────────
# 4. COMPARISON
# ─────────────────────────────────────────────

def compare_strategies(signals: pd.DataFrame, bt: PairsBacktester, costs=(0, 5, 10, 20)):
    """ML filter vs all entries across cost levels, plus the random-matched null at the base cost."""
    # Eligible = rows from folds where a threshold existed (same days for every strategy).
    elig = signals[signals["threshold"].notna()]
    sel = elig[elig["selected"]]
    if elig.empty:
        raise ValueError("No OOS rows have a past-only signal threshold; more folds/history are needed.")
    if sel.empty:
        raise ValueError("No OOS signals were selected; inspect threshold history or signal outputs.")
    print(f"eligible entries: {len(elig):,} | ML-selected: {len(sel):,}")
    common_period = (pd.Timestamp(elig["time"].min()), bt.calendar.max())

    rows = []
    for c in costs:
        for name, cand in (("ML filter", sel), ("All entries", elig)):
            rows.append({"strategy": name, "cost_bps": c,
                         **bt.run(cand, cost_bps=c, evaluation_period=common_period).stats})
    table = pd.DataFrame(rows)

    # null test at the configured cost: random entries, same count as the ML filter
    cfg, rng = bt.cfg, np.random.default_rng(bt.cfg.seed)
    ml = bt.run(sel, evaluation_period=common_period).stats
    draws = [bt.run(elig.iloc[rng.choice(len(elig), len(sel), replace=False)],
                    evaluation_period=common_period).stats
             for _ in range(cfg.n_random)]
    null = pd.DataFrame(draws)
    null_summary = {
        "ml_mean_net_bps": ml.get("mean_net_bps"),
        "random_mean_net_bps": null["mean_net_bps"].mean(),
        "random_p5": null["mean_net_bps"].quantile(0.05),
        "random_p95": null["mean_net_bps"].quantile(0.95),
        "ml_percentile_vs_random": (null["mean_net_bps"] < ml.get("mean_net_bps", np.nan)).mean(),
        "ml_sharpe": ml.get("sharpe"),
        "random_sharpe_mean": null["sharpe"].mean(),
    }
    return table, null_summary


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format: {path.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/lake", help="root used by run_pipeline.py")
    ap.add_argument("--signals", default=None, help="override signals_oos.parquet path")
    ap.add_argument("--dataset", default=None, help="override pair_metrics.parquet path")
    ap.add_argument("--prices", default=None, help="override clean wide-price path")
    ap.add_argument("--pair-id", default=None, help="run a localized resilience check for a single pair")
    ap.add_argument("--lookback-days", type=int, default=365, help="trailing lookback window for pair resilience checks")
    ap.add_argument("--costs", type=float, nargs="+", default=[0, 5, 10, 20])
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--max-positions", type=int, default=50)
    ap.add_argument("--z-stop", type=float, default=4.0)
    ap.add_argument("--max-hold", type=int, default=20)
    ap.add_argument("--out", default="results/backtest")
    a = ap.parse_args()

    if a.pair_id:
        print(check_pair_resilience(a.pair_id, lookback_days=a.lookback_days))
        return

    root = Path(a.output_dir)
    signals_path = Path(a.signals) if a.signals else root / "analytics" / "ml_signals" / "signals_oos.parquet"
    dataset_path = Path(a.dataset) if a.dataset else root / "analytics" / "pair_metrics.parquet"
    prices_path = Path(a.prices) if a.prices else root / "processed" / "prices_clean.parquet"
    for path in (signals_path, dataset_path, prices_path):
        if not path.exists():
            ap.error(f"required pipeline artifact not found: {path}")

    signals = _read_frame(signals_path)
    dataset = _read_frame(dataset_path)
    prices = _read_frame(prices_path)
    cfg = BTConfig(cost_bps=a.cost_bps, max_positions=a.max_positions, z_stop=a.z_stop, max_hold=a.max_hold)

    bt = PairsBacktester(prices, dataset, cfg)
    table, null = compare_strategies(signals, bt, tuple(a.costs))

    show = ["strategy", "cost_bps", "n_trades", "win_rate", "mean_gross_bps", "mean_net_bps",
            "profit_factor", "pct_stop", "pct_pair_breakdown", "avg_hold", "ann_return",
            "sharpe", "max_drawdown", "avg_open"]
    print("\n=== ML filter vs all entries, by cost level ===")
    print(table[show].round(3).to_string(index=False))
    print("\n=== Null test: ML filter vs random entries of the same count ===")
    print(pd.Series(null).round(3).to_string())
    print("\nRead: ML must beat 'All entries' AND sit above the random p95 (percentile > ~0.95)\n"
          "with net bps > 0 at a realistic cost. Gross > 0 but net < 0 means costs eat the edge.")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(a.out) / "comparison.csv", index=False)
    pd.Series(null).to_csv(Path(a.out) / "null_test.csv")


if __name__ == "__main__":
    main()
