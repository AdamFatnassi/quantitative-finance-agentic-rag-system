from contextlib import contextmanager
from datetime import date, datetime, timezone

import pytest
from langchain_core.runnables import RunnableLambda

from src.rag.evaluation.signal_evidence import (
    SignalVerdict,
    adjudicate_context,
    evaluate_and_persist_signal,
    write_signal_evidence,
)


class FakeConnection:
    def __init__(self):
        self.executions = []

    def execute(self, statement, parameters):
        self.executions.append((str(statement), parameters))


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    @contextmanager
    def begin(self):
        yield self.connection


class FakeStructuredLLM:
    def __init__(self, response):
        self.response = response
        self.prompt_text = ""

    def with_structured_output(self, schema):
        assert schema is SignalVerdict

        def respond(prompt_value):
            self.prompt_text = prompt_value.to_string()
            return self.response

        return RunnableLambda(respond)


def test_adjudicator_returns_structured_verdict_and_restricts_evidence_ids():
    llm = FakeStructuredLLM({
        "verdict": "VETO",
        "confidence": 0.91,
        "evidence_ids": ["chunk-a"],
        "rationale": "The filing describes a material leadership change.",
    })
    signal = {
        "signal_id": "signal-1",
        "pair": ("AAA", "BBB"),
        "direction": "long_spread",
        "zscore": 2.5,
        "model_version": "stat-arb-1",
    }
    context = {
        "t_signal": "2024-05-01T20:00:00+00:00",
        "legs": {
            "AAA": [{
                "chunk_id": "chunk-a",
                "payload": {"item_code": "5.02", "text": "CEO transition"},
            }],
            "BBB": [],
        },
    }

    verdict = adjudicate_context(signal, context, llm=llm)

    assert verdict.verdict == "VETO"
    assert verdict.confidence == 0.91
    assert verdict.evidence_ids == ["chunk-a"]
    assert "permanent fundamental shift" in llm.prompt_text
    assert "5.02" in llm.prompt_text

    llm.response = {
        "verdict": "GO",
        "confidence": 0.5,
        "evidence_ids": ["not-retrieved"],
        "rationale": "Unsupported citation",
    }
    with pytest.raises(ValueError, match="not retrieved"):
        adjudicate_context(signal, context, llm=llm)


def test_no_evidence_conservatively_vetoes_without_calling_llm():
    verdict = adjudicate_context(
        {"direction": "long_spread", "zscore": 2.0, "model_version": "v1"},
        {"legs": {"AAA": [], "BBB": []}, "t_signal": "2024-01-01T00:00:00Z"},
    )

    assert verdict.verdict == "VETO"
    assert verdict.confidence == 0
    assert verdict.evidence_ids == []


def test_none_coverage_policy_is_explicit_even_if_context_has_chunks(caplog):
    signal = {
        "signal_id": "signal-none-coverage",
        "direction": "long_spread",
        "zscore": 2.0,
        "model_version": "v1",
    }
    context = {
        "legs": {
            "AAA": [{"chunk_id": "chunk-a"}],
            "BBB": [],
        },
        "t_signal": "2024-01-01T00:00:00Z",
    }

    verdict = adjudicate_context(
        signal,
        context,
        coverage={"status": "none"},
    )

    assert verdict.verdict == "VETO"
    assert "policy=veto" in verdict.rationale
    assert "applying policy veto" in caplog.text
    with pytest.raises(RuntimeError, match="policy=raise"):
        adjudicate_context(
            signal,
            context,
            coverage={"status": "none"},
            none_coverage_policy="raise",
        )


def test_evidence_writer_replaces_rows_and_persists_empty_leg_verdict():
    engine = FakeEngine()
    signal = {
        "signal_time": date(2024, 5, 1),
        "pair_id": 12,
        "model_id": 34,
    }
    context = {
        "legs": {
            "AAA": [{"chunk_id": "chunk-a", "rank": 1, "score": 0.87}],
            "BBB": [],
        },
        "coverage": {"status": "partial"},
    }
    verdict = SignalVerdict(
        verdict="VETO",
        confidence=0.9,
        evidence_ids=["chunk-a"],
        rationale="Material event disclosed by leg AAA.",
    )

    row_count = write_signal_evidence(
        engine,
        signal,
        context,
        verdict,
        model_version="gpt-4o-mini",
        retrieved_at=datetime(2024, 5, 1, 20, tzinfo=timezone.utc),
    )

    assert row_count == 2
    assert len(engine.connection.executions) == 2
    delete_sql, delete_params = engine.connection.executions[0]
    insert_sql, insert_params = engine.connection.executions[1]
    assert "DELETE FROM signal_evidence" in delete_sql
    assert delete_params == {"signal_time": date(2024, 5, 1), "pair_id": 12, "model_id": 34}
    assert "INSERT INTO signal_evidence" in insert_sql
    assert len(insert_params) == 2
    assert insert_params[0]["chunk_id"] == "chunk-a"
    assert insert_params[0]["verdict"] == "VETO"
    assert insert_params[0]["coverage_status"] == "partial"
    assert insert_params[1]["chunk_id"] is None
    assert insert_params[1]["rank"] is None


def test_evaluate_accepts_date_signal_and_applies_explicit_none_coverage(monkeypatch):
    from src.rag.evaluation import signal_evidence

    engine = FakeEngine()
    observed = {}

    def fake_retrieve(signal, client=None, config=None, encoder=None):
        observed["t_signal"] = signal["t_signal"]
        observed["client"] = client
        observed["config"] = config
        observed["encoder"] = encoder
        return {
            "t_signal": signal["t_signal"],
            "legs": {"AAA": [], "BBB": []},
        }

    monkeypatch.setattr(signal_evidence, "retrieve_context", fake_retrieve)
    result = evaluate_and_persist_signal(
        {
            "signal_time": date(2024, 1, 2),
            "pair_id": 12,
            "model_id": 34,
            "pair": ("AAA", "BBB"),
            "direction": "long_spread",
            "zscore": 2.0,
            "model_version": "stat-arb-v1",
        },
        engine,
        none_coverage_policy="veto",
    )

    assert observed["t_signal"] == "2024-01-02T21:00:00+00:00"
    assert observed["client"] is None
    assert observed["config"] is not None
    assert observed["encoder"] is None
    assert result["coverage"]["status"] == "none"
    assert result["verdict"]["verdict"] == "VETO"
