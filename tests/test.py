"""Focused regression checks for the spread-mode behavior."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.quant.pairs_selection import PairScreenConfig, SpreadFeatureEngine


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=150, freq="B")
    time = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "AAA": np.exp(4.0 + 0.001 * time + 0.02 * np.sin(time / 7.0)),
            "BBB": np.exp(3.0 + 0.0005 * time + 0.03 * np.cos(time / 9.0)),
        },
        index=index,
    )
    pairs = pd.DataFrame(
        {
            "stock_a": ["AAA"],
            "stock_b": ["BBB"],
            "hedge_ratio": [0.75],
            "intercept": [1.25],
            "spread_std": [0.02],
            "half_life": [10.0],
            "adf_pvalue": [0.01],
            "coint_qvalue": [0.02],
        }
    )
    return prices, pairs


def test_frozen_spread_has_no_beta_drift_component() -> None:
    prices, pairs = _fixture()
    engine = SpreadFeatureEngine(PairScreenConfig(spread_mode="frozen"))
    wide = engine.compute_wide(prices, pairs)
    y = np.log(prices["AAA"])
    x = np.log(prices["BBB"])
    beta = float(pairs.loc[0, "hedge_ratio"])
    spread_diff = wide["spread"]["AAA/BBB"].diff()
    fixed_weight_return = y.diff() - beta * x.diff()

    np.testing.assert_allclose(
        spread_diff.to_numpy(),
        fixed_weight_return.to_numpy(),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    valid = spread_diff.notna() & fixed_weight_return.notna()
    beta_component = spread_diff[valid] - fixed_weight_return[valid]
    beta_share = beta_component.var(ddof=1) / spread_diff[valid].var(ddof=1)
    assert beta_share < 1e-20


def test_frozen_spread_features_are_causal() -> None:
    prices, pairs = _fixture()
    engine = SpreadFeatureEngine(PairScreenConfig(spread_mode="frozen"))
    full = engine.compute_wide(prices, pairs)
    cutoff = prices.index[100]
    truncated = engine.compute_wide(prices.loc[:cutoff], pairs)

    for name in full:
        assert_frame_equal(full[name].loc[:cutoff], truncated[name])


def test_pair_metrics_hedge_ratio_is_constant_within_pair_window() -> None:
    prices, pairs = _fixture()
    engine = SpreadFeatureEngine(PairScreenConfig(spread_mode="frozen"))
    metrics = engine.compute(prices, pairs, prices.index[80], window_id=7)

    assert not metrics.empty
    counts = metrics.groupby(["window_id", "stock_a", "stock_b"])["hedge_ratio"].nunique()
    assert counts.eq(1).all()
    np.testing.assert_array_equal(metrics["hedge_ratio"].to_numpy(), 0.75)


def test_spread_mode_validation_and_rolling_option() -> None:
    assert PairScreenConfig().spread_mode == "frozen"
    assert PairScreenConfig(spread_mode="rolling").spread_mode == "rolling"
    try:
        PairScreenConfig(spread_mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid spread_mode should fail validation")


if __name__ == "__main__":
    test_frozen_spread_has_no_beta_drift_component()
    test_frozen_spread_features_are_causal()
    test_pair_metrics_hedge_ratio_is_constant_within_pair_window()
    test_spread_mode_validation_and_rolling_option()
    print("All spread-mode checks passed.")