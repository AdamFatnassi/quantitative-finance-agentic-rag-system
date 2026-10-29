"""Integration tests for the Agentic Risk Gatekeeper FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from src.agent.schemas import GatekeeperResponse
from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant_connected" in data
    assert "collection_name" in data


def test_evaluate_signal_endpoint_prefilter_bypass(client):
    # Mock prefilter to bypass
    mock_agent = MagicMock()
    mock_agent.evaluate.return_value = GatekeeperResponse(
        decision="GO",
        confidence_score=1.0,
        risk_category="none",
        justification="Fast Pre-Filter Bypass: No recent filings found.",
        filing_ids_inspected=[],
    )

    with patch("src.api.main.get_agent", return_value=mock_agent):
        payload = {
            "ticker_a": "AAPL",
            "ticker_b": "MSFT",
            "signal_timestamp": "2024-06-01T16:00:00Z",
            "z_score": 2.15,
        }
        res = client.post("/api/v1/evaluate-signal", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["decision"] == "GO"
        assert data["confidence_score"] == 1.0
        assert data["risk_category"] == "none"
        assert data["filing_ids_inspected"] == []


def test_evaluate_signal_endpoint_veto(client):
    mock_agent = MagicMock()
    mock_agent.evaluate.return_value = GatekeeperResponse(
        decision="VETO",
        confidence_score=0.9,
        risk_category="operational",
        justification="Boeing 737 Max 9 emergency grounding Item 8-K indicates structural manufacturing disruption.",
        filing_ids_inspected=["chunk-ba-1"],
    )

    with patch("src.api.main.get_agent", return_value=mock_agent):
        payload = {
            "ticker_a": "BA",
            "ticker_b": "CAT",
            "signal_timestamp": "2024-01-10T16:00:00Z",
            "z_score": -2.8,
        }
        res = client.post("/api/v1/evaluate-signal", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["decision"] == "VETO"
        assert data["risk_category"] == "operational"
        assert "Boeing" in data["justification"]


def test_batch_evaluate_endpoint_with_deduplication(client):
    mock_agent = MagicMock()
    mock_agent.evaluate.return_value = GatekeeperResponse(
        decision="GO",
        confidence_score=1.0,
        risk_category="none",
        justification="Fast Pre-Filter Bypass: No recent filings.",
        filing_ids_inspected=[],
    )

    with patch("src.api.main.get_agent", return_value=mock_agent):
        # Two identical signals in batch (should deduplicate / cache)
        payload = [
            {
                "ticker_a": "AAPL",
                "ticker_b": "MSFT",
                "signal_timestamp": "2024-06-01T16:00:00Z",
                "z_score": 2.1,
            },
            {
                "ticker_a": "AAPL",
                "ticker_b": "MSFT",
                "signal_timestamp": "2024-06-01T16:00:00Z",
                "z_score": 2.1,
            },
            {
                "ticker_a": "GOOG",
                "ticker_b": "AMZN",
                "signal_timestamp": "2024-06-02T16:00:00Z",
                "z_score": -2.4,
            },
        ]
        res = client.post("/api/v1/batch-evaluate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 3
        assert data[0]["decision"] == "GO"
        assert data[1]["decision"] == "GO"
        assert data[2]["decision"] == "GO"


def test_simulate_spread_endpoint(client):
    """Test the Monte Carlo simulation endpoint."""
    payload = {
        "ticker_a": "JPM",
        "ticker_b": "BAC",
        "as_of_timestamp": "2024-01-15T16:00:00Z",
        "lookback_days": 252,
        "horizon_days": 10,
        "num_simulations": 200,
    }
    res = client.post("/api/v1/simulate-spread", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["ticker_a"] == "JPM"
    assert data["ticker_b"] == "BAC"
    assert "probability_of_convergence" in data
    assert 0.0 <= data["probability_of_convergence"] <= 1.0
    assert "breach_probability" in data
    assert 0.0 <= data["breach_probability"] <= 1.0
    assert data["num_simulations"] == 200
    assert data["horizon_days"] == 10


def test_simulate_spread_endpoint_invalid_ticker(client):
    """Endpoint should return 400 Bad Request if ticker is unknown in price universe."""
    payload = {
        "ticker_a": "ZZZZ",
        "ticker_b": "BAC",
        "as_of_timestamp": "2024-01-15T16:00:00Z",
    }
    res = client.post("/api/v1/simulate-spread", json=payload)
    assert res.status_code == 400
    assert "not found" in res.json()["detail"].lower()

