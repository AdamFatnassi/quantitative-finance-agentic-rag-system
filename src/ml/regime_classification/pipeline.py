"""Walk-forward orchestration for market-regime classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quant.pairs_selection import SpreadFeatureEngine, WalkForwardOrchestrator, WalkForwardWindow
from ml.regime_classification.hmm_engine import HMMConfig, HMMRegimeEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetConfig:
    z_exit: float = 0.5
    horizon: int = 10


def compute_target(z_score: pd.Series, config: TargetConfig = TargetConfig()) -> pd.Series:
    """Label whether |Z| falls below z_exit within the next horizon bars."""
    abs_z = z_score.abs()
    future = pd.concat([abs_z.shift(-k) for k in range(1, config.horizon + 1)], axis=1)
    min_forward = future.min(axis=1)
    y = (min_forward < config.z_exit).astype(float)
    y[future.isna().any(axis=1)] = float("nan")
    return y.rename("y")


class RegimeFeatureBuilder:
    """Fit on formation data, then causally classify trading-window observations."""

    def __init__(self, hmm_config: HMMConfig = HMMConfig(), target_config: TargetConfig = TargetConfig()):
        self.hmm_cfg = hmm_config
        self.target_cfg = target_config
        self._hmm = HMMRegimeEngine(hmm_config)

    def build(self, orchestrator: WalkForwardOrchestrator, prices: pd.DataFrame,
              sectors_df: pd.DataFrame,
              selected_pairs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        cfg = orchestrator.cfg
        sector_map = (sectors_df.drop_duplicates(cfg.symbol_col)
                      .set_index(cfg.symbol_col)[cfg.sector_col])

        if selected_pairs is not None and selected_pairs.empty:
            return pd.DataFrame()

        rows = []
        for window in orchestrator.iter_windows(prices.index):
            if selected_pairs is None:
                formation = prices.iloc[window.pos_start: window.pos_t0 + 1]
                result = orchestrator.screener.screen(formation, sector_map)
                valid = result.valid
            else:
                valid = selected_pairs.loc[selected_pairs["window_id"] == window.window_id]
            if valid.empty:
                continue

            history = prices.iloc[window.pos_start: window.pos_end + 1]
            frame = self._build_window(orchestrator.engine, history, valid, window)
            if not frame.empty:
                rows.append(frame)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _build_window(self, engine: SpreadFeatureEngine, history: pd.DataFrame,
                      valid_pairs: pd.DataFrame, window: WalkForwardWindow) -> pd.DataFrame:
        wide = engine.compute_wide(history, valid_pairs)
        wide.pop("_active_today", None)
        keys = (valid_pairs["stock_a"] + "/" + valid_pairs["stock_b"]).tolist()
        formation_mask = history.index <= window.t0
        trading_mask = history.index > window.t0

        obs_by_key = {key: pd.DataFrame({name: frame[key] for name, frame in wide.items()},
                                        index=history.index)
                      for key in keys}
        regime_models = self._fit_window_models(obs_by_key, formation_mask, window.window_id)

        frames = []
        for key, (_, pair_row) in zip(keys, valid_pairs.iterrows()):
            regime_model = regime_models.get(key)
            if regime_model is None:
                logger.debug("window %d %s: formation sample too thin, skipped", window.window_id, key)
                continue

            obs = obs_by_key[key]
            regime = self._hmm.filter(regime_model, obs)
            target = compute_target(obs["z_score"], self.target_cfg)
            frame = pd.concat([obs, regime, target], axis=1).loc[trading_mask]
            frame = frame.dropna(subset=list(self.hmm_cfg.obs_cols) + ["regime_state"])
            if frame.empty:
                continue

            frame.insert(0, "time", frame.index)
            frame.insert(1, "window_id", window.window_id)
            frame.insert(2, "stock_a", pair_row["stock_a"])
            frame.insert(3, "stock_b", pair_row["stock_b"])
            frame.insert(4, "sector", pair_row["sector"])
            frame["half_life"] = pair_row["half_life"]
            frames.append(frame.reset_index(drop=True))

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fit_window_models(self, obs_by_key: dict, formation_mask: pd.Series, window_id: int) -> dict:
        """Fit pooled or per-pair HMMs, falling back when a pool is too small."""
        formation_by_pair = {key: obs.loc[formation_mask] for key, obs in obs_by_key.items()}
        if self.hmm_cfg.fit_scope == "window":
            pooled = self._hmm.fit_pooled(formation_by_pair)
            if pooled:
                return pooled
            logger.debug("window %d: too few pairs to pool (need >= %d), falling back to per-pair fit",
                         window_id, self.hmm_cfg.min_pool_pairs)

        return {key: model for key, obs in formation_by_pair.items()
                if (model := self._hmm.fit(obs)) is not None}
