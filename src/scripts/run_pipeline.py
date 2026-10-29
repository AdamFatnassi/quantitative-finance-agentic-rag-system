"""Run the yfinance -> Parquet -> pair/HMM -> TimescaleDB market-data pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.data_loader import MarketDataLoader
from data.data_processor import DataProcessor
from diagnostics.evaluation import EvalConfig
from ml.ml_signals import MLSignal, SignalConfig, save_result
from data.market_data_io import (
    resolve_pair_ids,
    upsert_assets,
    write_market_data,
    register_ml_model,
    write_pair_candidates,
    write_pair_metrics,
    write_regime_features,
    write_ml_signals,
)
from quant.pairs_selection import PairScreenConfig, WalkForwardOrchestrator
from ml.regime_classification import HMMConfig, RegimeFeatureBuilder


def run_pipeline(
    start_date: str,
    end_date: str,
    output_dir: str,
    database_url: str | None,
    use_cache: bool = False,
    hmm_n_states: int = 3,
    signal_model_type: str = "logistic",
    signal_z_entry: float = 2.0,
    signal_z_exit: float = 0.5,
    signal_horizon: int = 10,
    run_signals: bool = True,
) -> dict[str, int]:
    """Build raw/clean/feature Parquets, then optionally load them into TimescaleDB."""
    root = Path(output_dir)
    raw_dir = root / "raw" / "assets"
    raw_dir.mkdir(parents=True, exist_ok=True)
    loader = MarketDataLoader(output_dir=str(raw_dir))
    raw = loader.download(start_date=start_date, end_date=end_date, use_cache=use_cache)

    processor = (
        DataProcessor(raw)
        .extract_ohlcv()
        .validate_adjusted_prices()
        .align_time_series()
        .clean_bad_ticks()
        .filter_history()
        .filter_liquidity()
    )
    # Keep prices linear here: the pair and HMM engines apply log prices themselves.
    processed_dir = root / "processed" / "assets"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processor.save(str(processed_dir))
    clean = processor.build()
    universe_path = root.parent / "universe.parquet"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe = clean.universe_frame()
    universe.to_parquet(universe_path, index=False)

    pair_config = PairScreenConfig()
    orchestrator = WalkForwardOrchestrator(pair_config)
    selection = orchestrator.run(clean.prices, clean.sectors)

    analytics_dir = root / "analytics" / "assets"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selection_model_version = f"stat-arb-{run_stamp}-{uuid4().hex[:8]}"
    all_candidates = selection.all_candidates.copy()
    if not all_candidates.empty:
        all_candidates["model_version"] = selection_model_version
    else:
        all_candidates = pd.DataFrame(columns=[
            "window_id", "t0", "stock_a", "stock_b", "sector", "ssd", "ssd_rank",
            "correlation", "hedge_ratio", "intercept", "spread_mean", "spread_std",
            "adf_pvalue", "coint_pvalue", "coint_qvalue", "half_life", "hurst",
            "formation_n", "leg_a_adf_pvalue", "leg_b_adf_pvalue",
            "passes_correlation", "passes_cointegration", "passes_i1_legs",
            "passes_hedge_bounds", "passes_half_life", "passes_hurst", "is_valid",
            "model_version",
        ])
    all_candidates.to_parquet(analytics_dir / "pair_candidates.parquet", index=False)
    selection.pairs.to_parquet(analytics_dir / "pairs.parquet", index=False)
    selection.pair_metrics.to_parquet(analytics_dir / "pair_metrics.parquet", index=False)
    selection.funnel.to_parquet(analytics_dir / "pair_selection_funnel.parquet", index=False)

    hmm_config = HMMConfig(n_states=hmm_n_states)
    hmm_model_version = f"hmm-k{hmm_config.n_states}-{run_stamp}-{uuid4().hex[:8]}"
    regimes = RegimeFeatureBuilder(hmm_config).build(
        orchestrator, clean.prices, clean.sectors, selected_pairs=selection.pairs
    )
    regimes.to_parquet(analytics_dir / "hmm_regimes.parquet", index=False)

    signal_count = 0
    signal_result = None
    signal_model_version = None
    if run_signals and not selection.pair_metrics.empty:
        signal_config = SignalConfig(
            model_type=signal_model_type,
            eval=EvalConfig(z_entry=signal_z_entry, horizon=signal_horizon),
            z_exit=signal_z_exit,
        )
        signal_result = MLSignal(signal_config).fit_predict(selection.pair_metrics)
        signal_model_version = f"{signal_model_type}-{run_stamp}-{uuid4().hex[:8]}"
        versioned_signal_dir = analytics_dir.parent / "ml_models" / signal_model_version
        save_result(signal_result, str(versioned_signal_dir))
        stable_signal_dir = analytics_dir.parent / "ml_signals"
        stable_signal_dir.mkdir(parents=True, exist_ok=True)
        signal_result.oos.to_parquet(stable_signal_dir / "signals_oos.parquet", index=False)
        signal_count = int(signal_result.oos["signal"].ne(0).sum())
    elif run_signals:
        print("Skipping ML signals: pair selection produced no pair metrics.")

    counts = {
        "assets": clean.n_tickers,
        "universe_tickers": len(universe),
        "market_data": len(clean.prices) * clean.n_tickers,
        "pair_candidates": len(all_candidates),
        "pairs": len(selection.pairs),
        "pair_metrics": len(selection.pair_metrics),
        "hmm_regimes": len(regimes),
        "ml_signals": signal_count,
        "ml_models": 0,
    }
    if database_url is None:
        return counts

    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        asset_id_map = upsert_assets(engine, clean.sectors)
        if not all_candidates.empty:
            counts["pair_candidates"] = write_pair_candidates(
                engine, all_candidates, asset_id_map, selection_model_version
            )
        market_data = clean.to_long_format(asset_id_map)
        counts["market_data"] = write_market_data(engine, market_data)

        if not selection.pairs.empty:
            pairs = resolve_pair_ids(engine, selection.pairs)
            if not selection.pair_metrics.empty:
                metrics = resolve_pair_ids(engine, selection.pair_metrics)
                counts["pair_metrics"] = write_pair_metrics(engine, metrics)
            if not regimes.empty:
                hmm_model_id = register_ml_model(
                    engine,
                    model_name="pair_spread_regime",
                    model_version=hmm_model_version,
                    model_type="gaussian_hmm",
                    task="regime_classification",
                    status="trained",
                    feature_names=list(hmm_config.obs_cols),
                    hyperparameters={
                        "n_states": hmm_config.n_states,
                        "fit_scope": hmm_config.fit_scope,
                        "covariance_type": hmm_config.covariance_type,
                        "n_iter": hmm_config.n_iter,
                        "random_state": hmm_config.random_state,
                    },
                    training_start=clean.prices.index.min().date().isoformat(),
                    training_end=clean.prices.index.max().date().isoformat(),
                    source_revision=os.environ.get("GIT_COMMIT"),
                )
                regime_rows = resolve_pair_ids(engine, regimes)
                counts["hmm_regimes"] = write_regime_features(
                    engine, regime_rows, hmm_config.n_states, hmm_model_id,
                    model_version=hmm_model_version,
                )
                counts["ml_models"] += 1
            if signal_result is not None and signal_model_version is not None:
                signal_model_id = register_ml_model(
                    engine,
                    model_name="pairs_reversion_signal",
                    model_version=signal_model_version,
                    model_type=signal_model_type,
                    task="mean_reversion_classification",
                    status="trained",
                    feature_names=list(signal_result.config.features),
                    hyperparameters={
                        "model_params": signal_result.config.model_params,
                        "seed": signal_result.config.seed,
                        "z_entry": signal_result.config.eval.z_entry,
                        "z_exit": signal_result.config.z_exit,
                        "horizon": signal_result.config.eval.horizon,
                        "top_quantile": signal_result.config.top_quantile,
                        "min_history_for_threshold": signal_result.config.min_history_for_threshold,
                    },
                    evaluation_metrics={**signal_result.metrics, **signal_result.signal_metrics},
                    training_start=clean.prices.index.min().date().isoformat(),
                    training_end=clean.prices.index.max().date().isoformat(),
                    artifact_uri=str((analytics_dir / "ml_models" / signal_model_version).resolve()),
                    source_revision=os.environ.get("GIT_COMMIT"),
                )
                signal_rows = resolve_pair_ids(engine, signal_result.oos)
                counts["ml_signals"] = write_ml_signals(engine, signal_rows, signal_model_id)
                counts["ml_models"] += 1
        return counts
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default="2025-01-01")
    parser.add_argument("--output-dir", default="data/lake")
    parser.add_argument("--use-cache", action="store_true", help="reuse existing raw yfinance Parquets")
    parser.add_argument("--skip-database", action="store_true", help="build Parquets without a database")
    parser.add_argument("--n-states", type=int, default=3, help="number of HMM states (minimum 2)")
    parser.add_argument("--signal-model", choices=["logistic", "hgb", "xgboost", "lightgbm"],
                        default="logistic", help="classifier used for mean-reversion signals")
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--skip-signals", action="store_true", help="skip ML signal training")
    args = parser.parse_args()
    if args.n_states < 2:
        parser.error("--n-states must be >= 2")
    if args.z_entry <= 0 or args.z_exit <= 0 or args.horizon < 1:
        parser.error("--z-entry and --z-exit must be positive; --horizon must be >= 1")

    database_url = None if args.skip_database else os.environ.get("DATABASE_URL")
    if not args.skip_database and not database_url:
        parser.error("set DATABASE_URL or pass --skip-database")

    counts = run_pipeline(
        args.start_date, args.end_date, args.output_dir, database_url,
        use_cache=args.use_cache, hmm_n_states=args.n_states,
        signal_model_type=args.signal_model, signal_z_entry=args.z_entry,
        signal_z_exit=args.z_exit, signal_horizon=args.horizon,
        run_signals=not args.skip_signals,
    )
    print("Pipeline complete:", counts)


if __name__ == "__main__":
    main()