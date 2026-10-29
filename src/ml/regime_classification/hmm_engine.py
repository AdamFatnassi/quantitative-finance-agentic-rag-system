"""Gaussian HMM engine for causal market-regime classification.

Models fit on formation observations only; trading-window probabilities use a
forward filter, so a regime label at t cannot depend on later observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 3
    covariance_type: str = "diag"
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 42
    min_formation_obs: int = 60
    fit_scope: str = "pair"
    min_pool_pairs: int = 3
    obs_cols: tuple = (
        "z_score", "dz_1", "spread_vol_20", "spread_skew_60",
        "spread_slope_5", "vol_ratio",
    )

    def __post_init__(self):
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if not self.obs_cols:
            raise ValueError("obs_cols must be non-empty")
        if self.fit_scope not in ("pair", "window"):
            raise ValueError("fit_scope must be 'pair' or 'window'")


@dataclass
class RegimeModel:
    """A fitted HMM and its standardization parameters."""
    model: GaussianHMM
    mean: np.ndarray
    scale: np.ndarray
    state_order: np.ndarray


class HMMRegimeEngine:
    """Fit-once, filter-forward Gaussian HMM regime detector."""

    def __init__(self, config: HMMConfig = HMMConfig()):
        self.cfg = config

    def fit(self, formation_obs: pd.DataFrame) -> Optional[RegimeModel]:
        """Fit one pair's HMM on formation data, or return None if too thin."""
        cfg = self.cfg
        obs = formation_obs[list(cfg.obs_cols)].dropna()
        if len(obs) < cfg.min_formation_obs:
            return None

        mean, scale = self._standardize_params(obs)
        model = self._new_model()
        model.fit(self._standardize(obs, mean, scale))
        return RegimeModel(model, mean, scale, self._order_states(model, cfg.obs_cols))

    def fit_pooled(self, formation_by_pair: dict) -> dict:
        """Fit one shared HMM using independent formation sequences per pair."""
        cfg = self.cfg
        seqs, lengths, scalers, keys = [], [], {}, []
        for key, formation_obs in formation_by_pair.items():
            obs = formation_obs[list(cfg.obs_cols)].dropna()
            if len(obs) < cfg.min_formation_obs:
                continue
            mean, scale = self._standardize_params(obs)
            scalers[key] = (mean, scale)
            seqs.append(self._standardize(obs, mean, scale))
            lengths.append(len(obs))
            keys.append(key)

        if len(seqs) < cfg.min_pool_pairs:
            return {}

        model = self._new_model()
        model.fit(np.concatenate(seqs, axis=0), lengths=lengths)
        state_order = self._order_states(model, cfg.obs_cols)
        return {key: RegimeModel(model, *scalers[key], state_order) for key in keys}

    def filter(self, regime_model: RegimeModel, obs: pd.DataFrame) -> pd.DataFrame:
        """Return causal filtered regime probabilities, using observations only through t."""
        cfg, model = self.cfg, regime_model.model
        cols = list(cfg.obs_cols)
        valid = obs[cols].notna().all(axis=1)
        n_states = model.n_components
        out = pd.DataFrame(
            np.nan, index=obs.index,
            columns=[f"regime_p{k}" for k in range(n_states)] + ["regime_state", "regime_entropy"],
        )
        if not valid.any():
            return out

        X = self._standardize(obs.loc[valid, cols], regime_model.mean, regime_model.scale)
        log_b = model._compute_log_likelihood(X)
        log_pi = np.log(model.startprob_ + 1e-300)
        log_A = np.log(model.transmat_ + 1e-300)

        T, K = log_b.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]

        m = log_alpha.max(axis=1, keepdims=True)
        post = np.exp(log_alpha - m)
        post /= post.sum(axis=1, keepdims=True)
        post = post[:, regime_model.state_order]

        prob_cols = [f"regime_p{k}" for k in range(n_states)]
        out.loc[valid, prob_cols] = post
        out.loc[valid, "regime_state"] = post.argmax(axis=1)
        entropy = -(post * np.log(np.clip(post, 1e-12, 1.0))).sum(axis=1)
        out.loc[valid, "regime_entropy"] = np.clip(entropy, 0.0, None)
        return out

    def _new_model(self) -> GaussianHMM:
        cfg = self.cfg
        return GaussianHMM(
            n_components=cfg.n_states, covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter, tol=cfg.tol, random_state=cfg.random_state,
        )

    @staticmethod
    def _standardize_params(obs: pd.DataFrame) -> tuple:
        mean = obs.mean().to_numpy(dtype=float)
        scale = obs.std().to_numpy(dtype=float).copy()
        scale[scale == 0] = 1.0
        return mean, scale

    @staticmethod
    def _standardize(obs: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (obs.to_numpy(dtype=float) - mean) / scale

    @staticmethod
    def _order_states(model: GaussianHMM, obs_cols: tuple) -> np.ndarray:
        """Sort hidden states by absolute z-score mean for stable regime labels."""
        z_idx = obs_cols.index("z_score") if "z_score" in obs_cols else 0
        return np.argsort(np.abs(model.means_[:, z_idx]))
