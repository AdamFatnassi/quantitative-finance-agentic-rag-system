"""
run_ablation.py — does the HMM regime layer add out-of-sample value?

Compares, on identical rows and identical purged walk-forward folds:
    K=0 : six raw spread features only (baseline, no HMM)
    K=2,3,4 : raw features + regime features from an HMM with K states

Usage:
    python run_ablation.py --ks 0 2 3 4 --z-entry 2.0 --seeds 3 --out results/ablation

Outputs (in --out):
    summary.csv        one row per K (pooled metrics, averaged over seeds)
    paired_vs_k0.csv   fold-by-fold AUC difference of each K vs the baseline
    dataset_K{k}.pkl   cached built datasets (delete to force a rebuild)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data.data_loader import MarketDataLoader
from data.data_processor import DataProcessor
from diagnostics.evaluation import (RAW_COLS, EvalConfig, out_of_sample_predictions,
                        paired_comparison, prepare, regime_cols, summarize)
from quant.pairs_selection import PairScreenConfig, WalkForwardOrchestrator
from ml.regime_classification import HMMConfig, RegimeFeatureBuilder, TargetConfig

logger = logging.getLogger("ablation")
ROW_KEYS = ["window_id", "time", "stock_a", "stock_b"]


# ─────────────────────────────────────────────
# TODO: wire this to your project. It must return:
#   orchestrator    : pairs_selection.WalkForwardOrchestrator
#   prices          : DataFrame (index = dates, columns = symbols)
#   sectors_df      : DataFrame with the symbol/sector columns the orchestrator config expects
#   selected_pairs  : DataFrame with window_id, stock_a, stock_b, sector, half_life
#                     (recommended: select pairs ONCE so every K sees identical pairs;
#                      or None to re-screen inside each build)
# ─────────────────────────────────────────────
def load_inputs(args):
    """Load cleaned prices and screen pairs once for all K values."""
    root = Path(args.data_dir)
    raw = MarketDataLoader(output_dir=str(root / "raw")).download(
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=args.use_cache,
    )
    clean = (
        DataProcessor(raw)
        .extract_ohlcv()
        .validate_adjusted_prices()
        .align_time_series()
        .clean_bad_ticks()
        .filter_history()
        .filter_liquidity()
        .build()
    )
    orchestrator = WalkForwardOrchestrator(PairScreenConfig())
    selected_pairs = orchestrator.run(clean.prices, clean.sectors).pairs
    if selected_pairs.empty:
        raise RuntimeError("Pair screening produced no valid pairs; cannot run the ablation.")
    return orchestrator, clean.prices, clean.sectors, selected_pairs


def build_dataset(k: int, args, inputs) -> pd.DataFrame:
    """Build (or load cached) the feature table. K=0 reuses a K=2 table, using raw columns only."""
    build_k = k if k > 0 else args.baseline_k
    cache = Path(args.out) / (
        f"dataset_{args.start_date}_{args.end_date}_z{args.z_exit:g}_h{args.horizon}_"
        f"{args.fit_scope}_K{build_k}.pkl"
    )
    if cache.exists() and not args.rebuild:
        logger.info("loading cached %s", cache)
        return pd.read_pickle(cache)

    orchestrator, prices, sectors_df, selected_pairs = inputs
    builder = RegimeFeatureBuilder(
        HMMConfig(n_states=build_k, fit_scope=args.fit_scope),
        TargetConfig(z_exit=args.z_exit, horizon=args.horizon),
    )
    df = builder.build(orchestrator, prices, sectors_df, selected_pairs)
    if df.empty:
        raise RuntimeError(f"No feature rows were built for K={build_k}.")
    df.to_pickle(cache)
    return df


def align_common_rows(datasets: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Restrict every K to the same (window, date, pair) observations."""
    key_sets = [set(map(tuple, df[ROW_KEYS].itertuples(index=False, name=None)))
                for df in datasets.values()]
    common = set.intersection(*key_sets) if key_sets else set()
    if not common:
        raise RuntimeError("The tested K values have no common labelled feature rows.")
    keys = pd.DataFrame(sorted(common), columns=ROW_KEYS)
    return {
        k: (df.merge(keys, on=ROW_KEYS, how="inner", validate="one_to_one")
            .sort_values(ROW_KEYS).reset_index(drop=True))
        for k, df in datasets.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 3, 4])
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--fit-scope", choices=["pair", "window"], default="pair")
    ap.add_argument("--seeds", type=int, default=3, help="downstream-model seeds to average over")
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--data-dir", default="data/lake", help="raw market-data cache directory")
    ap.add_argument("--start-date", default="2005-01-01")
    ap.add_argument("--end-date", default="2025-01-01")
    ap.add_argument("--use-cache", action="store_true", help="reuse raw yfinance Parquets")
    ap.add_argument("--rebuild", action="store_true", help="rebuild cached K datasets")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.ks = list(dict.fromkeys(args.ks))
    if not args.ks or any(k == 1 or k < 0 for k in args.ks):
        ap.error("--ks must contain 0 (baseline) and/or state counts >= 2")
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    args.baseline_k = min((k for k in args.ks if k > 0), default=2)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ecfg = EvalConfig(z_entry=args.z_entry, horizon=args.horizon)
    inputs = load_inputs(args)

    prepared = {k: prepare(build_dataset(k, args, inputs), ecfg) for k in args.ks}
    prepared = align_common_rows(prepared)

    summaries, oos_by_k = [], {}
    for k in args.ks:
        df = prepared[k]
        cols = RAW_COLS + (regime_cols(k) if k > 0 else [])
        logger.info("K=%d: %d common entry rows, %d features", k, len(df), len(cols))

        # average downstream-model randomness; HMM randomness is fixed by HMMConfig.random_state
        oos_seeds = [out_of_sample_predictions(df, cols, ecfg, seed=s) for s in range(args.seeds)]
        oos = pd.concat(oos_seeds).groupby(level=0).agg({"window_id": "first", "y": "first", "p": "mean"})
        oos_by_k[k] = oos
        summaries.append({"K": k, **summarize(oos, ecfg)})

    summary = pd.DataFrame(summaries).set_index("K")
    summary.to_csv(Path(args.out) / "summary.csv")
    print("\n=== Out-of-sample summary (entry situations only) ===")
    print(summary.round(4).to_string())

    if 0 in oos_by_k:
        paired = {k: paired_comparison(oos_by_k[0], oos_by_k[k]) for k in args.ks if k != 0}
        paired_df = pd.DataFrame(paired).T
        paired_df.to_csv(Path(args.out) / "paired_vs_k0.csv")
        print("\n=== Paired fold-by-fold AUC: K vs baseline (K=0) ===")
        print(paired_df.round(4).to_string())
        print("\nRead: mean_auc_diff > ~2*se and most folds improved => regime features help.")


if __name__ == "__main__":
    main()
