"""test_monte_carlo.py

Unit and integration tests for:
- src.quant.monte_carlo (Ornstein-Uhlenbeck spread simulator)
- src.agent.tools.run_monte_carlo_simulation (LLM tool wrapper)
- src.agent.schemas (GatekeeperResponse Monte Carlo extensions)
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.agent.schemas import GatekeeperResponse
from src.agent.tools import run_monte_carlo_simulation
from src.quant.monte_carlo import (
    _fit_ou_parameters,
    run_ou_simulation,
    simulate_spread_paths,
)


class TestMonteCarloOU:
    """Test suite for the Ornstein-Uhlenbeck spread simulation engine."""

    def test_fit_ou_parameters_positive_speed(self):
        """OU fitting on an AR(1) mean-reverting process should produce theta > 0."""
        rng = np.random.default_rng(42)
        n = 250
        theta_true = 0.15
        mu_true = 0.05
        spread = np.zeros(n)
        spread[0] = mu_true
        for t in range(1, n):
            spread[t] = (
                spread[t - 1] * np.exp(-theta_true)
                + mu_true * (1.0 - np.exp(-theta_true))
                + rng.normal(0, 0.02)
            )

        theta, mu, sigma_ou = _fit_ou_parameters(spread)
        assert theta > 0.0, "Fitted theta must be positive"
        assert abs(mu - mu_true) < 0.1, "Fitted mu should be close to true mean"
        assert sigma_ou > 0.0, "Sigma OU must be positive"

    def test_run_ou_simulation_metrics(self):
        """Simulation metrics should satisfy probability bounds [0, 1] and path shape."""
        rng = np.random.default_rng(123)
        spread = pd.Series(rng.normal(0, 0.03, size=200).cumsum())

        res = run_ou_simulation(
            spread=spread,
            horizon_days=10,
            num_simulations=500,
            z_exit=0.5,
            z_stop=4.0,
            random_seed=42,
        )

        assert 0.0 <= res["probability_of_convergence"] <= 1.0
        assert 0.0 <= res["breach_probability"] <= 1.0
        assert res["half_life_days"] > 0
        assert res["expected_max_drawdown_bps"] >= 0
        assert res["simulation_paths"].shape == (500, 11)
        assert res["num_simulations"] == 500
        assert res["horizon_days"] == 10

    def test_simulate_spread_paths_positive_prices(self):
        """simulate_spread_paths should compute spread and output valid risk metrics."""
        rng = np.random.default_rng(77)
        dates = pd.date_range("2024-01-01", periods=150, freq="B")
        pa = pd.Series(100.0 * np.exp(rng.normal(0, 0.01, 150).cumsum()), index=dates)
        pb = pd.Series(50.0 * np.exp(rng.normal(0, 0.01, 150).cumsum()), index=dates)

        res = simulate_spread_paths(pa, pb, horizon_days=5, num_simulations=200)
        assert "probability_of_convergence" in res
        assert "breach_probability" in res
        assert 0.0 <= res["probability_of_convergence"] <= 1.0
        assert res["simulation_paths"].shape == (200, 6)

    def test_simulate_spread_paths_insufficient_bars_raises_error(self):
        """simulate_spread_paths must raise ValueError if fewer than 60 bars are provided."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        pa = pd.Series(np.ones(30) * 100, index=dates)
        pb = pd.Series(np.ones(30) * 50, index=dates)

        with pytest.raises(ValueError, match="minimum 60 required"):
            simulate_spread_paths(pa, pb)

    def test_simulate_spread_paths_non_positive_price_raises_error(self):
        """Non-positive prices must be rejected (cannot take logarithm)."""
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        pa = pd.Series(np.ones(80) * 100, index=dates)
        pb = pd.Series(np.ones(80) * 50, index=dates)
        pb.iloc[10] = -1.0

        with pytest.raises(ValueError, match="strictly positive"):
            simulate_spread_paths(pa, pb)


class TestAgentMonteCarloTool:
    """Test suite for run_monte_carlo_simulation tool and schema extensions."""

    def test_run_monte_carlo_simulation_known_pair(self):
        """Tool should succeed on known tickers in price universe without look-ahead."""
        as_of = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)
        res = run_monte_carlo_simulation(
            ticker_a="JPM",
            ticker_b="BAC",
            as_of=as_of,
            lookback_days=252,
            horizon_days=10,
            num_simulations=250,
        )

        assert "error" not in res, f"Expected success but got error: {res.get('error')}"
        assert res["ticker_a"] == "JPM"
        assert res["ticker_b"] == "BAC"
        assert res["as_of"] == "2024-01-15"
        assert 0.0 <= res["probability_of_convergence"] <= 1.0
        assert 0.0 <= res["breach_probability"] <= 1.0
        assert isinstance(res["probability_of_convergence"], float)
        assert isinstance(res["breach_probability"], float)

    def test_run_monte_carlo_simulation_missing_ticker_returns_error_dict(self):
        """Tool should gracefully return error dict when ticker is invalid."""
        as_of = datetime(2024, 1, 15, tzinfo=timezone.utc)
        res = run_monte_carlo_simulation(
            ticker_a="NONEXISTENT_TICKER_XYZ",
            ticker_b="BAC",
            as_of=as_of,
        )
        assert "error" in res
        assert "not found in price universe" in res["error"]

    def test_gatekeeper_response_monte_carlo_fields(self):
        """GatekeeperResponse supports optional Monte Carlo diagnostics fields."""
        resp = GatekeeperResponse(
            decision="GO",
            confidence_score=0.92,
            risk_category="none",
            justification="Transitory divergence verified; Monte Carlo paths confirm mean-reversion.",
            filing_ids_inspected=["chunk-1"],
            simulation_performed=True,
            convergence_probability=0.84,
            breach_probability=0.03,
            simulation_notes="OU Half-Life = 8.4 days; 84% convergence probability within 10 days.",
        )
        assert resp.simulation_performed is True
        assert resp.convergence_probability == 0.84
        assert resp.breach_probability == 0.03
        assert "84%" in (resp.simulation_notes or "")

    def test_gatekeeper_response_probability_bounds(self):
        """Probabilities outside [0, 1] must raise ValidationError."""
        with pytest.raises(ValidationError):
            GatekeeperResponse(
                decision="GO",
                confidence_score=0.9,
                risk_category="none",
                justification="Testing bounds",
                simulation_performed=True,
                convergence_probability=1.5,  # Invalid
            )
