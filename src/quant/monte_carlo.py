"""monte_carlo.py

Ornstein-Uhlenbeck (OU) Monte Carlo spread path simulator for quantitative
stress-testing of cointegrated pair-trading signals.

All price slices must be strictly point-in-time (no lookahead bias). The caller
is responsible for slicing `prices_a` and `prices_b` up to the signal timestamp
before calling these functions.

Key design choices
------------------
- Spread = log(P_A) - beta * log(P_B), where beta is the hedge ratio.
- OU parameters (theta, mu, sigma) are fitted via AR(1) OLS on the demeaned spread.
- Exact discretized OU scheme:
      S_{t+1} = S_t * exp(-theta * dt)
               + mu * (1 - exp(-theta * dt))
               + sigma * sqrt((1 - exp(-2*theta*dt)) / (2*theta)) * Z
- Minimum 60 bars of overlapping data required; raises ValueError otherwise.
- NumPy 2.x compatible (uses np.trapezoid, not np.trapz).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────
_MIN_BARS: int = 60          # Minimum history bars required for OU fitting
_DT: float = 1.0            # Daily time step (1 trading day)
_THETA_FLOOR: float = 1e-4  # Prevent division-by-zero in discretised scheme
_THETA_CEIL: float = 5.0    # Cap mean-reversion speed at 5 (unrealistically fast)


# ─── OU Parameter Estimation ───────────────────────────────────────────────────

def _fit_ou_parameters(
    spread: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate OU parameters (theta, mu, sigma_ou) via AR(1) OLS on spread.

    Parameters
    ----------
    spread : np.ndarray
        1-D array of spread values (log-price spread or z-scored spread).

    Returns
    -------
    theta : float
        Mean-reversion speed (per trading day).
    mu : float
        Long-run mean of the spread.
    sigma_ou : float
        OU diffusion coefficient (per sqrt(day)).
    """
    n = len(spread)
    s_t  = spread[:-1]          # S_t
    s_t1 = spread[1:]           # S_{t+1}

    # OLS: S_{t+1} = a + b * S_t  =>  b = exp(-theta), a = mu*(1-b)
    x_mat = np.column_stack([np.ones(n - 1), s_t])
    coeffs, *_ = np.linalg.lstsq(x_mat, s_t1, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    # AR(1) coefficient must be in (0, 1) for mean-reversion; clamp defensively
    b = np.clip(b, np.exp(-_THETA_CEIL), 1.0 - 1e-8)

    theta = float(np.clip(-np.log(b), _THETA_FLOOR, _THETA_CEIL))
    mu = float(a / (1.0 - b)) if abs(1.0 - b) > 1e-10 else float(np.mean(spread))

    # Residual standard deviation → OU sigma
    residuals = s_t1 - (a + b * s_t)
    sigma_eps = float(np.std(residuals, ddof=1))

    # sigma_ou from exact discretised scheme: sigma_eps = sigma_ou * sqrt((1-exp(-2*theta))/2*theta)
    denom = np.sqrt((1.0 - np.exp(-2.0 * theta * _DT)) / (2.0 * theta + 1e-12))
    sigma_ou = sigma_eps / (denom + 1e-12)

    return theta, mu, sigma_ou


# ─── Core Simulation Engine ─────────────────────────────────────────────────────

def run_ou_simulation(
    spread: pd.Series,
    horizon_days: int = 10,
    num_simulations: int = 1000,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    random_seed: int = 42,
) -> dict:
    """Run Monte Carlo OU simulation directly on a pre-computed spread series.

    Parameters
    ----------
    spread : pd.Series
        Point-in-time spread series (log-price or z-scored). Must have >= 60 obs.
    horizon_days : int
        Forward simulation horizon in trading days. Default 10.
    num_simulations : int
        Number of Monte Carlo paths to simulate. Default 1000.
    z_exit : float
        Convergence threshold: |z| <= z_exit counts as a reversion hit. Default 0.5.
    z_stop : float
        Adverse stop threshold: |z| >= z_stop counts as a stop-loss breach. Default 4.0.
    random_seed : int
        NumPy random seed for reproducibility. Default 42.

    Returns
    -------
    dict with keys:
        probability_of_convergence : float   Fraction of paths that hit |z| <= z_exit.
        breach_probability         : float   Fraction of paths that hit |z| >= z_stop.
        expected_max_drawdown_bps  : float   Mean peak-to-trough spread excursion (bps).
        median_final_z             : float   Median spread at horizon end.
        half_life_days             : float   Estimated OU half-life in trading days.
        theta                      : float   Mean-reversion speed (per day).
        mu                         : float   Long-run mean of spread.
        sigma_ou                   : float   OU diffusion coefficient.
        num_simulations            : int     Number of paths run.
        horizon_days               : int     Simulation horizon.
        simulation_paths           : np.ndarray  Shape [num_simulations, horizon_days+1].
        spread_history             : np.ndarray  Historical spread values.

    Raises
    ------
    ValueError
        If `spread` has fewer than `_MIN_BARS` observations.
    """
    arr = spread.dropna().values.astype(float)
    if len(arr) < _MIN_BARS:
        raise ValueError(
            f"Spread series has only {len(arr)} observations; "
            f"minimum {_MIN_BARS} required for reliable OU fitting."
        )

    # ── Fit OU Parameters ─────────────────────────────────────────────────────
    theta, mu, sigma_ou = _fit_ou_parameters(arr)

    half_life = float(np.log(2.0) / (theta + 1e-12))

    # ── Z-score normalise the spread for threshold comparisons ────────────────
    spread_std = float(np.std(arr, ddof=1))
    spread_mean = float(np.mean(arr))
    spread_std = spread_std if spread_std > 1e-8 else 1.0

    current_level = arr[-1]                  # Most recent spread value (S_0)

    # ── Simulate Paths (exact OU discretisation) ──────────────────────────────
    rng = np.random.default_rng(random_seed)

    exp_neg_theta = np.exp(-theta * _DT)
    mean_term = mu * (1.0 - exp_neg_theta)
    noise_std = sigma_ou * np.sqrt((1.0 - np.exp(-2.0 * theta * _DT)) / (2.0 * theta + 1e-12))

    # paths shape: [num_simulations, horizon_days + 1]
    paths = np.empty((num_simulations, horizon_days + 1), dtype=np.float64)
    paths[:, 0] = current_level

    shocks = rng.standard_normal((num_simulations, horizon_days))

    for t in range(horizon_days):
        paths[:, t + 1] = (
            paths[:, t] * exp_neg_theta
            + mean_term
            + noise_std * shocks[:, t]
        )

    # ── Convert paths to z-score space for threshold checks ───────────────────
    z_paths = (paths - spread_mean) / spread_std

    # ── Convergence: any step in path where |z| <= z_exit ─────────────────────
    converged = np.any(np.abs(z_paths[:, 1:]) <= z_exit, axis=1)
    prob_convergence = float(converged.mean())

    # ── Breach: any step in path where |z| >= z_stop ──────────────────────────
    breached = np.any(np.abs(z_paths[:, 1:]) >= z_stop, axis=1)
    prob_breach = float(breached.mean())

    # ── Expected Max Drawdown (spread excursion from start, in bps-equivalent) ─
    # Using the range of spread excursion relative to starting level, scaled to bps
    excursions = np.max(np.abs(paths[:, 1:] - paths[:, 0:1]), axis=1)
    # Normalise by spread_std and scale to bps for display
    expected_max_drawdown_bps = float(np.mean(excursions) / (spread_std + 1e-12) * 100.0)

    # ── Median final z-score ───────────────────────────────────────────────────
    median_final_z = float(np.median(z_paths[:, -1]))

    return {
        "probability_of_convergence": prob_convergence,
        "breach_probability": prob_breach,
        "expected_max_drawdown_bps": expected_max_drawdown_bps,
        "median_final_z": median_final_z,
        "half_life_days": half_life,
        "theta": theta,
        "mu": mu,
        "sigma_ou": sigma_ou,
        "num_simulations": num_simulations,
        "horizon_days": horizon_days,
        "simulation_paths": paths,
        "spread_history": arr,
    }


# ─── Public API: Accepts raw price series ──────────────────────────────────────

def simulate_spread_paths(
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: Optional[float] = None,
    horizon_days: int = 10,
    num_simulations: int = 1000,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    random_seed: int = 42,
) -> dict:
    """Simulate OU spread paths for a cointegrated pair from raw price series.

    Enforces strict point-in-time discipline: prices_a and prices_b must already
    be sliced up to the signal timestamp by the caller (no look-ahead bias).

    Parameters
    ----------
    prices_a : pd.Series
        Point-in-time price history for asset A. Must overlap with prices_b.
    prices_b : pd.Series
        Point-in-time price history for asset B. Must overlap with prices_a.
    hedge_ratio : float, optional
        Known hedge ratio (beta). If None, estimated via OLS on log-prices.
    horizon_days : int
        Forward simulation horizon in trading days. Default 10.
    num_simulations : int
        Number of Monte Carlo paths. Default 1000.
    z_exit : float
        Reversion threshold (|z| <= z_exit = converged). Default 0.5.
    z_stop : float
        Adverse stop threshold (|z| >= z_stop = breached). Default 4.0.
    random_seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    dict  — Same schema as `run_ou_simulation`.

    Raises
    ------
    ValueError
        If prices_a or prices_b is empty, overlap < _MIN_BARS, or prices contain
        non-positive values (log transform undefined).
    """
    # ── Align on common index and drop NaN ────────────────────────────────────
    df = pd.DataFrame({"a": prices_a, "b": prices_b}).dropna()
    if len(df) == 0:
        raise ValueError("No overlapping observations between prices_a and prices_b.")
    if (df["a"] <= 0).any() or (df["b"] <= 0).any():
        raise ValueError("Prices must be strictly positive for log transform.")
    if len(df) < _MIN_BARS:
        raise ValueError(
            f"Overlapping price history has only {len(df)} bars; "
            f"minimum {_MIN_BARS} required."
        )

    log_a = np.log(df["a"].values)
    log_b = np.log(df["b"].values)

    # ── Hedge Ratio ───────────────────────────────────────────────────────────
    if hedge_ratio is None:
        # OLS: log_a = alpha + beta * log_b
        x_ols = np.column_stack([np.ones(len(log_b)), log_b])
        coeffs, *_ = np.linalg.lstsq(x_ols, log_a, rcond=None)
        hedge_ratio = float(coeffs[1])
        logger.debug("OLS hedge ratio estimated: %.4f", hedge_ratio)
    else:
        hedge_ratio = float(hedge_ratio)

    # ── Compute spread ────────────────────────────────────────────────────────
    spread_arr = log_a - hedge_ratio * log_b
    spread_series = pd.Series(spread_arr, index=df.index)

    return run_ou_simulation(
        spread=spread_series,
        horizon_days=horizon_days,
        num_simulations=num_simulations,
        z_exit=z_exit,
        z_stop=z_stop,
        random_seed=random_seed,
    )
