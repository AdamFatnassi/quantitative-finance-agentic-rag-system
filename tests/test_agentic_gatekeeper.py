"""Unit tests for the Agentic Risk Gatekeeper layer (schemas, prefilter, tools, agent)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agent.agent import AgenticRiskGatekeeper
from src.agent.prefilter import count_recent_filings, evaluate_prefilter
from src.agent.schemas import FilingChunk, GatekeeperResponse, SignalRequest
from src.agent.tools import search_sec_filings


class TestSchemas:
    def test_signal_request_normalizes_tickers_and_tz(self):
        sig = SignalRequest(
            ticker_a="aapl ",
            ticker_b=" msft.b",
            signal_timestamp=datetime(2024, 6, 1, 15, 30),
            z_score=2.15,
        )
        assert sig.ticker_a == "AAPL"
        assert sig.ticker_b == "MSFT-B"
        assert sig.signal_timestamp.tzinfo is not None

    def test_signal_request_requires_valid_fields(self):
        with pytest.raises(ValidationError):
            SignalRequest(
                ticker_a="",
                ticker_b="MSFT",
                signal_timestamp=datetime.now(),
                z_score=1.0,
            )

    def test_gatekeeper_response_validation(self):
        resp = GatekeeperResponse(
            decision="GO",
            confidence_score=0.95,
            risk_category="none",
            justification="No fundamental risks found in recent filings.",
            filing_ids_inspected=["chunk-1", "chunk-2"],
        )
        assert resp.decision == "GO"
        assert resp.risk_category == "none"

        # Invalid decision
        with pytest.raises(ValidationError):
            GatekeeperResponse(
                decision="MAYBE",  # type: ignore
                confidence_score=0.5,
                risk_category="none",
                justification="Testing invalid",
            )

        # Confidence out of bounds
        with pytest.raises(ValidationError):
            GatekeeperResponse(
                decision="VETO",
                confidence_score=1.5,
                risk_category="financial_distress",
                justification="Invalid score",
            )


class TestPrefilter:
    def test_prefilter_bypasses_when_no_filings(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = MagicMock(count=0)

        sig = SignalRequest(
            ticker_a="AAPL",
            ticker_b="MSFT",
            signal_timestamp=datetime(2024, 6, 1, 16, 0, tzinfo=timezone.utc),
            z_score=2.3,
        )

        bypassed, resp = evaluate_prefilter(mock_client, sig)
        assert bypassed is True
        assert resp is not None
        assert resp.decision == "GO"
        assert resp.confidence_score == 1.0
        assert resp.risk_category == "none"
        assert "Pre-Filter Bypass" in resp.justification
        assert resp.filing_ids_inspected == []

    def test_prefilter_triggers_when_recent_filings_exist(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        # First call returns 3 (ticker_a), second returns 0 (ticker_b)
        mock_client.count.side_effect = [MagicMock(count=3), MagicMock(count=0)]

        sig = SignalRequest(
            ticker_a="AAPL",
            ticker_b="MSFT",
            signal_timestamp=datetime(2024, 6, 1, 16, 0, tzinfo=timezone.utc),
            z_score=2.3,
        )

        bypassed, resp = evaluate_prefilter(mock_client, sig)
        assert bypassed is False
        assert resp is None


class TestTools:
    def test_search_sec_filings_enforces_zero_lookahead(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        as_of = datetime(2024, 6, 1, 16, 0, tzinfo=timezone.utc)

        # Mock points: one valid (prior), one invalid (lookahead)
        valid_point = MagicMock()
        valid_point.id = "p1"
        valid_point.payload = {
            "chunk_id": "c1",
            "ticker": "AAPL",
            "document_id": "doc1",
            "available_at": "2024-05-30T20:00:00+00:00",
            "section_type": "item_section",
            "text": "Routine business update",
        }

        future_point = MagicMock()
        future_point.id = "p2"
        future_point.payload = {
            "chunk_id": "c2",
            "ticker": "AAPL",
            "document_id": "doc2",
            "available_at": "2024-06-02T20:00:00+00:00",  # AFTER as_of!
            "section_type": "item_section",
            "text": "Future filing that should never be returned",
        }

        mock_client.scroll.return_value = ([valid_point, future_point], None)

        chunks = search_sec_filings(
            ticker="AAPL",
            as_of_timestamp=as_of,
            query="",
            client=mock_client,
        )

        # Only the valid prior chunk should be retained
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"
        assert datetime.fromisoformat(chunks[0].available_at) <= as_of


class TestAgentCore:
    def test_agent_evaluates_bypassed_signal_without_llm(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = MagicMock(count=0)

        agent = AgenticRiskGatekeeper(qdrant_client=mock_client)
        sig = SignalRequest(
            ticker_a="AAPL",
            ticker_b="MSFT",
            signal_timestamp=datetime(2024, 6, 1, 16, 0, tzinfo=timezone.utc),
            z_score=2.0,
        )

        response = agent.evaluate(sig)
        assert response.decision == "GO"
        assert response.risk_category == "none"

    def test_forensic_fallback_detects_item_402_restatement(self):
        mock_client = MagicMock()
        agent = AgenticRiskGatekeeper(qdrant_client=mock_client)

        sig = SignalRequest(
            ticker_a="BA",
            ticker_b="AIR",
            signal_timestamp=datetime(2024, 6, 1, 16, 0, tzinfo=timezone.utc),
            z_score=-2.8,
        )

        evidence = {
            "BA": [
                FilingChunk(
                    chunk_id="chunk-ba-402",
                    ticker="BA",
                    document_id="doc-ba",
                    item_code="4.02",
                    available_at="2024-05-30T15:00:00Z",
                    text="Company announces restatement of previously issued financial statements due to accounting error.",
                )
            ],
            "AIR": [],
        }

        fallback = agent._rule_based_forensic_fallback(sig, evidence, reason="LLM unavailable test")
        assert fallback.decision == "VETO"
        assert fallback.risk_category == "financial_distress"
        assert "Item 4.02" in fallback.justification
        assert "chunk-ba-402" in fallback.filing_ids_inspected

    def test_agent_structured_llm_invocation(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        # Prefilter triggers: 2 filings
        mock_client.count.side_effect = [MagicMock(count=2), MagicMock(count=0)]

        point = MagicMock()
        point.id = "p-item"
        point.payload = {
            "chunk_id": "c-ba-8k",
            "ticker": "BA",
            "document_id": "doc1",
            "item_code": "5.02",
            "available_at": "2024-05-30T20:00:00+00:00",
            "section_type": "item_section",
            "text": "CEO resignation announced following board review.",
        }
        mock_client.scroll.return_value = ([point], None)

        agent = AgenticRiskGatekeeper(qdrant_client=mock_client)

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = GatekeeperResponse(
            decision="VETO",
            confidence_score=0.92,
            risk_category="executive_governance",
            justification="Item 5.02 indicates sudden CEO departure. Spread divergence reflects executive leadership crisis.",
            filing_ids_inspected=["c-ba-8k"],
        )

        sample_chunks = {
            "BA": [
                FilingChunk(
                    chunk_id="c-ba-8k",
                    ticker="BA",
                    document_id="doc1",
                    item_code="5.02",
                    available_at="2024-05-30T20:00:00+00:00",
                    section_type="item_section",
                    text="CEO resignation announced following board review.",
                )
            ],
            "CAT": [],
        }

        with patch.object(agent, "_build_langchain_chain", return_value=mock_chain), \
             patch("src.agent.agent.retrieve_pair_filings", return_value=sample_chunks):
            sig = SignalRequest(
                ticker_a="BA",
                ticker_b="CAT",
                signal_timestamp=datetime(2024, 6, 1, 16, 0, tzinfo=timezone.utc),
                z_score=2.6,
            )
            resp = agent.evaluate(sig)
            assert resp.decision == "VETO"
            assert resp.risk_category == "executive_governance"
            assert "CEO departure" in resp.justification
