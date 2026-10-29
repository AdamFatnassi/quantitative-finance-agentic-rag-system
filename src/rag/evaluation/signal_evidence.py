"""Classify point-in-time signal context and persist its retrieved evidence."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.rag.coverage import normalize_signal_time
from src.rag.qdrant.retrieve_context import (
    QueryEncoder,
    RetrievalConfig,
    retrieve_context,
)

logger = logging.getLogger(__name__)


class SignalVerdict(BaseModel):
    verdict: Literal["GO", "VETO"]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    rationale: str = Field(min_length=1)


def adjudicate_context(
    signal: dict,
    context: dict,
    model_name: str | None = None,
    llm=None,
    coverage: dict | None = None,
    none_coverage_policy: str = "veto",
) -> SignalVerdict:
    """Adjudicate with an explicit policy for signals that lack text coverage."""
    if none_coverage_policy not in {"veto", "raise"}:
        raise ValueError("none_coverage_policy must be 'veto' or 'raise'")
    evidence = [
        chunk
        for leg_chunks in context["legs"].values()
        for chunk in leg_chunks
    ]
    if coverage is None:
        populated_legs = sum(bool(chunks) for chunks in context["legs"].values())
        coverage_status = (
            "none" if not evidence
            else "full" if populated_legs == len(context["legs"])
            else "partial"
        )
        coverage = {"status": coverage_status}
    coverage_status = coverage.get("status")
    if coverage_status not in {"full", "partial", "none"}:
        raise ValueError("coverage.status must be full, partial, or none")
    if coverage_status == "none" or not evidence:
        message = (
            "No eligible point-in-time text coverage; policy="
            f"{none_coverage_policy}."
        )
        logger.warning(
            "Signal %s has none text coverage; applying policy %s",
            signal.get("signal_id"),
            none_coverage_policy,
        )
        if none_coverage_policy == "raise":
            raise RuntimeError(message)
        return SignalVerdict(
            verdict="VETO",
            confidence=0,
            evidence_ids=[],
            rationale=message + " The signal cannot be cleared.",
        )
    if coverage_status == "partial":
        logger.warning(
            "Signal %s has partial text coverage; LLM adjudication will use only "
            "available evidence.",
            signal.get("signal_id"),
        )

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    model_name = model_name or os.getenv("RAG_VERDICT_MODEL", "gpt-4o-mini")
    llm = llm or ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(SignalVerdict)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a conservative fundamental-information gate for a statistical-arbitrage
pairs-trading signal. Decide whether recent information explains the spread divergence
as a permanent fundamental shift (VETO) or as noise that should mean-revert (GO).

VETO if either leg has a material shock, including SEC 8-K items 4.02, 5.02, or 1.03,
litigation, a restatement, or a merger/acquisition. Use only the supplied evidence
available by the signal time. Do not infer that missing evidence is favorable. Cite
only supplied chunk_ids in evidence_ids and return the required structured JSON.""",
        ),
        (
            "human",
            "Signal: {signal}\nCoverage: {coverage}\n"
            "Eligible evidence by leg: {evidence}",
        ),
    ])
    chain = prompt | structured_llm
    raw_verdict = chain.invoke({
        "signal": json.dumps({
            "signal_id": signal.get("signal_id"),
            "pair": signal.get("pair"),
            "t_signal": context["t_signal"],
            "direction": signal["direction"],
            "zscore": signal["zscore"],
            "model_version": signal["model_version"],
        }, default=str),
        "evidence": json.dumps(evidence, default=str),
        "coverage": json.dumps(coverage, default=str),
    })
    verdict = SignalVerdict.model_validate(raw_verdict)
    eligible_ids = {str(chunk["chunk_id"]) for chunk in evidence}
    unknown_ids = set(verdict.evidence_ids) - eligible_ids
    if unknown_ids:
        raise ValueError(
            f"LLM returned evidence IDs not retrieved for this signal: {sorted(unknown_ids)}"
        )
    return verdict


def _signal_identity(signal: dict) -> dict:
    missing = {"signal_time", "pair_id", "model_id"} - set(signal)
    if missing:
        raise ValueError(
            "signal evidence persistence requires existing ml_signals key fields: "
            + ", ".join(sorted(missing))
        )
    timestamp = pd.Timestamp(signal["signal_time"])
    signal_time = timestamp.date()
    pair_id = int(signal["pair_id"])
    model_id = int(signal["model_id"])
    if pair_id < 1 or model_id < 1:
        raise ValueError("pair_id and model_id must be positive")
    return {"signal_time": signal_time, "pair_id": pair_id, "model_id": model_id}


def write_signal_evidence(
    engine,
    signal: dict,
    context: dict,
    verdict: SignalVerdict,
    model_version: str,
    retrieved_at: datetime | None = None,
) -> int:
    """Replace one signal's evidence atomically, including empty-leg verdict rows."""
    identity = _signal_identity(signal)
    if not model_version.strip():
        raise ValueError("model_version must not be empty")
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")

    rows = []
    for ticker, chunks in context["legs"].items():
        if not chunks:
            rows.append({
                **identity,
                "leg_ticker": ticker,
                "chunk_id": None,
                "rank": None,
                "score": None,
            })
        for chunk in chunks:
            rows.append({
                **identity,
                "leg_ticker": ticker,
                "chunk_id": str(chunk["chunk_id"]),
                "rank": int(chunk["rank"]),
                "score": float(chunk["score"]),
            })
    if not rows:
        raise ValueError("signal context must contain both pair legs")
    for row in rows:
        row.update({
            "retrieved_at": retrieved_at.astimezone(timezone.utc),
            "model_version": model_version,
            "coverage_status": context.get("coverage", {}).get("status", "unknown"),
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "rationale": verdict.rationale,
        })

    from sqlalchemy import text

    delete_statement = text(
        "DELETE FROM signal_evidence "
        "WHERE signal_time = :signal_time AND pair_id = :pair_id AND model_id = :model_id"
    )
    insert_statement = text(
        "INSERT INTO signal_evidence ("
        "signal_time, pair_id, model_id, leg_ticker, chunk_id, rank, score, "
        "retrieved_at, model_version, coverage_status, verdict, confidence, rationale"
        ") VALUES ("
        ":signal_time, :pair_id, :model_id, :leg_ticker, :chunk_id, :rank, :score, "
        ":retrieved_at, :model_version, :coverage_status, :verdict, :confidence, :rationale"
        ")"
    )
    with engine.begin() as connection:
        connection.execute(delete_statement, identity)
        connection.execute(insert_statement, rows)
    return len(rows)


def evaluate_and_persist_signal(
    signal: dict,
    engine,
    retrieval_config: RetrievalConfig = RetrievalConfig(),
    encoder: QueryEncoder | None = None,
    client=None,
    llm=None,
    verdict_model: str | None = None,
    coverage: dict | None = None,
    none_coverage_policy: str = "veto",
) -> dict:
    """Retrieve, classify, and write one query-time signal decision and its evidence."""
    query_signal = dict(signal)
    if not query_signal.get("signal_id"):
        identity = _signal_identity(signal)
        query_signal["signal_id"] = (
            f"{identity['signal_time'].isoformat()}:{identity['pair_id']}:"
            f"{identity['model_id']}"
        )
    signal_time = query_signal.get("t_signal", query_signal.get("signal_time"))
    if signal_time is None:
        raise ValueError("signal requires t_signal or signal_time for point-in-time retrieval")
    query_signal["t_signal"] = normalize_signal_time(signal_time).isoformat()
    context = retrieve_context(
        query_signal,
        client=client,
        config=retrieval_config,
        encoder=encoder,
    )
    if coverage is None:
        populated_legs = sum(bool(chunks) for chunks in context["legs"].values())
        coverage = {
            "status": (
                "none" if populated_legs == 0
                else "full" if populated_legs == len(context["legs"])
                else "partial"
            )
        }
    context["coverage"] = coverage
    verdict = adjudicate_context(
        signal,
        context,
        model_name=verdict_model,
        llm=llm,
        coverage=coverage,
        none_coverage_policy=none_coverage_policy,
    )
    saved_rows = write_signal_evidence(
        engine,
        signal,
        context,
        verdict,
        model_version=verdict_model or os.getenv("RAG_VERDICT_MODEL", "gpt-4o-mini"),
    )
    return {
        "context": context,
        "verdict": verdict.model_dump(mode="json"),
        "coverage": coverage,
        "evidence_rows": saved_rows,
    }
