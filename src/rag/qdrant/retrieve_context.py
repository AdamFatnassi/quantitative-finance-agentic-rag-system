"""Point-in-time hybrid retrieval for statistical-arbitrage signals."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.rag.qdrant.ingest_chunks import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    FastEmbedEncoder,
    validate_collection_models,
)
from src.rag.qdrant.qdrant_lifecycle import get_client


@dataclass(frozen=True)
class RetrievalConfig:
    collection_name: str = field(
        default_factory=lambda: os.getenv("RAG_QDRANT_COLLECTION", "financial_text_chunks")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )
    sparse_model: str = field(
        default_factory=lambda: os.getenv("RAG_SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
    )
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    top_k_per_leg: int = 5
    recency_8k_days: int = 90
    recency_10q_days: int = 180
    transcripts_per_leg: int = 2
    scroll_page_size: int = 256

    def __post_init__(self) -> None:
        if self.top_k_per_leg < 1 or self.transcripts_per_leg < 1:
            raise ValueError("top_k_per_leg and transcripts_per_leg must be >= 1")
        if self.recency_8k_days < 1 or self.recency_10q_days < 1:
            raise ValueError("SEC recency windows must be >= 1 day")
        if self.scroll_page_size < 1:
            raise ValueError("scroll_page_size must be >= 1")


class QueryEncoder(Protocol):
    def embed(self, texts: list[str]) -> tuple[list, list]:
        ...


def _as_signal_time(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("signal.t_signal must be timezone-aware")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _ticker_source_filter(
    ticker: str, source_type: str, signal_time: datetime
) -> qm.Filter:
    return qm.Filter(must=[
        qm.FieldCondition(key="tickers", match=qm.MatchAny(any=[ticker])),
        qm.FieldCondition(
            key="source_type", match=qm.MatchValue(value=source_type)
        ),
        qm.FieldCondition(
            key="available_at", range=qm.DatetimeRange(lte=signal_time)
        ),
    ])


def _latest_accession(
    client: QdrantClient,
    collection_name: str,
    ticker: str,
    signal_time: datetime,
) -> str | None:
    scroll_filter = _ticker_source_filter(ticker, "sec_10k", signal_time)
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1,
        order_by=qm.OrderBy(key="available_at", direction=qm.Direction.DESC),
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    accession = points[0].payload.get("accession_number")
    return accession if isinstance(accession, str) and accession else None


def _latest_transcript_ids(
    client: QdrantClient,
    collection_name: str,
    ticker: str,
    signal_time: datetime,
    limit: int,
    page_size: int,
) -> list[str]:
    scroll_filter = _ticker_source_filter(
        ticker, "earnings_transcript", signal_time
    )
    transcript_ids: list[str] = []
    seen: set[str] = set()
    offset = None
    while len(transcript_ids) < limit:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            order_by=qm.OrderBy(key="available_at", direction=qm.Direction.DESC),
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            transcript_id = point.payload.get("transcript_id")
            if isinstance(transcript_id, str) and transcript_id not in seen:
                seen.add(transcript_id)
                transcript_ids.append(transcript_id)
                if len(transcript_ids) >= limit:
                    break
        if offset is None:
            break
    return transcript_ids


def _eligible_scope_filters(
    client: QdrantClient,
    collection_name: str,
    ticker: str,
    signal_time: datetime,
    config: RetrievalConfig,
) -> list[tuple[str, qm.Filter]]:
    scopes: list[tuple[str, qm.Filter]] = []
    for source_type, days, name in (
        ("sec_8k", config.recency_8k_days, "recent_8k"),
        ("sec_10q", config.recency_10q_days, "recent_10q"),
    ):
        lower_bound = signal_time - timedelta(days=days)
        scopes.append((name, qm.Filter(must=[
            qm.FieldCondition(key="tickers", match=qm.MatchAny(any=[ticker])),
            qm.FieldCondition(
                key="source_type", match=qm.MatchValue(value=source_type)
            ),
            qm.FieldCondition(key="available_at", range=qm.DatetimeRange(
                gte=lower_bound, lte=signal_time
            )),
        ])))

    accession = _latest_accession(
        client, collection_name, ticker, signal_time
    )
    if accession:
        scopes.append(("latest_10k", qm.Filter(must=[
            qm.FieldCondition(key="tickers", match=qm.MatchAny(any=[ticker])),
            qm.FieldCondition(
                key="source_type", match=qm.MatchValue(value="sec_10k")
            ),
            qm.FieldCondition(
                key="accession_number", match=qm.MatchValue(value=accession)
            ),
            qm.FieldCondition(
                key="available_at", range=qm.DatetimeRange(lte=signal_time)
            ),
        ])))

    transcript_ids = _latest_transcript_ids(
        client,
        collection_name,
        ticker,
        signal_time,
        config.transcripts_per_leg,
        config.scroll_page_size,
    )
    if transcript_ids:
        scopes.append(("latest_transcripts", qm.Filter(must=[
            qm.FieldCondition(key="tickers", match=qm.MatchAny(any=[ticker])),
            qm.FieldCondition(
                key="source_type",
                match=qm.MatchValue(value="earnings_transcript"),
            ),
            qm.FieldCondition(
                key="transcript_id", match=qm.MatchAny(any=transcript_ids)
            ),
            qm.FieldCondition(
                key="available_at", range=qm.DatetimeRange(lte=signal_time)
            ),
        ])))
    return scopes


def _query_scope(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    sparse_vector,
    query_filter: qm.Filter,
    limit: int,
) -> list:
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            qm.Prefetch(
                query=query_vector,
                using=DENSE_VECTOR,
                filter=query_filter,
                limit=limit,
            ),
            qm.Prefetch(
                query=qm.SparseVector(
                    indices=[int(index) for index in sparse_vector.indices],
                    values=[float(value) for value in sparse_vector.values],
                ),
                using=SPARSE_VECTOR,
                filter=query_filter,
                limit=limit,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return response.points


def retrieve_context(
    signal: dict,
    client: QdrantClient | None = None,
    config: RetrievalConfig = RetrievalConfig(),
    encoder: QueryEncoder | None = None,
) -> dict:
    """Retrieve the latest eligible evidence independently for each pair leg."""
    required = {"signal_id", "pair", "t_signal", "direction", "zscore", "model_version"}
    missing = required - set(signal)
    if missing:
        raise ValueError(f"signal is missing required fields: {sorted(missing)}")
    if not isinstance(signal["pair"], (tuple, list)) or len(signal["pair"]) != 2:
        raise ValueError("signal.pair must contain exactly two tickers")
    pair = tuple(str(ticker).strip().upper() for ticker in signal["pair"])
    if not pair[0] or not pair[1] or pair[0] == pair[1]:
        raise ValueError("signal.pair must contain two distinct non-empty tickers")
    signal_time = _as_signal_time(signal["t_signal"])
    own_client = client is None
    client = client or get_client(config.qdrant_host, config.qdrant_port)
    if not client.collection_exists(config.collection_name):
        if own_client:
            client.close()
        raise ValueError(f"Qdrant collection {config.collection_name!r} does not exist")
    validate_collection_models(
        client,
        config.collection_name,
        config.embedding_model,
        config.sparse_model,
    )
    if encoder is None:
        encoder = FastEmbedEncoder(config.embedding_model, config.sparse_model)

    results: dict[str, list[dict]] = {}
    try:
        for index, ticker in enumerate(pair):
            peer = pair[1 - index]
            query_text = (
                f"Recent fundamental information for {ticker}, relative to {peer}, "
                f"that may explain a {signal['direction']} pairs-trading spread "
                f"divergence with z-score {signal['zscore']}."
            )
            dense, sparse = encoder.embed([query_text])
            if len(dense) != 1 or len(sparse) != 1:
                raise ValueError("query embedder must return exactly one dense and sparse vector")

            merged: dict[str, dict] = {}
            scopes = _eligible_scope_filters(
                client, config.collection_name, ticker, signal_time, config
            )
            for scope_name, query_filter in scopes:
                for point in _query_scope(
                    client,
                    config.collection_name,
                    [float(value) for value in dense[0]],
                    sparse[0],
                    query_filter,
                    config.top_k_per_leg,
                ):
                    payload = point.payload or {}
                    available_at = _as_signal_time(payload.get("available_at"))
                    if available_at > signal_time:
                        raise AssertionError(
                            f"Qdrant returned future chunk {point.id} for signal "
                            f"{signal.get('signal_id')}"
                        )
                    chunk_id = str(payload.get("chunk_id") or point.id)
                    result = {
                        "chunk_id": chunk_id,
                        "point_id": str(point.id),
                        "ticker": ticker,
                        "available_at": available_at.isoformat(),
                        "score": float(point.score),
                        "payload": payload,
                        "scope": scope_name,
                    }
                    previous = merged.get(chunk_id)
                    if previous is None or result["score"] > previous["score"]:
                        merged[chunk_id] = result
            ranked = sorted(
                merged.values(), key=lambda result: result["score"], reverse=True
            )[: config.top_k_per_leg]
            for rank, result in enumerate(ranked, start=1):
                result["rank"] = rank
            results[ticker] = ranked
    finally:
        if own_client:
            client.close()

    return {
        "signal_id": signal["signal_id"],
        "pair": pair,
        "t_signal": signal_time.isoformat(),
        "direction": signal["direction"],
        "zscore": signal["zscore"],
        "model_version": signal["model_version"],
        "legs": results,
    }
