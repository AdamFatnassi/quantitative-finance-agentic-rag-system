from dataclasses import dataclass
import warnings

import pandas as pd
from qdrant_client import QdrantClient

from src.rag.qdrant.ingest_chunks import IndexerConfig, ingest_frame
from src.rag.qdrant.replay_validation import replay_signals
from src.rag.qdrant.retrieve_context import RetrievalConfig, retrieve_context


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class FakeEncoder:
    def embed(self, texts):
        dense = [[1.0, 0.5, 0.25] for _ in texts]
        sparse = [SparseVector([2, 7], [1.0, 0.5]) for _ in texts]
        return dense, sparse


def _chunk(ticker, source_type, available_at, source_id, document_id, index):
    is_transcript = source_type == "earnings_transcript"
    return {
        "ticker": ticker,
        "cik": 1000000,
        "available_at": pd.Timestamp(available_at),
        "published_at": pd.Timestamp(available_at),
        "section_type": "section",
        "form_type": (
            "10-K" if source_type == "sec_10k"
            else "10-Q" if source_type == "sec_10q"
            else "8-K" if source_type == "sec_8k"
            else None
        ),
        "item_code": "1A" if source_type == "sec_10k" else None,
        "accession_number": None if is_transcript else source_id,
        "transcript_id": source_id if is_transcript else None,
        "availability_estimated": is_transcript,
        "source_type": source_type,
        "chunk_id": f"{document_id}#{index:03d}",
        "document_id": document_id,
        "chunk_index": index,
        "text": f"{source_type} evidence for {ticker}",
        "embedding_text": f"{source_type} evidence for {ticker}",
    }


def test_retrieve_context_filters_each_leg_by_point_in_time_and_recency():
    client = QdrantClient(":memory:")
    config = IndexerConfig(collection_name="pit-context")
    chunks = pd.DataFrame([
        _chunk("AAA", "sec_8k", "2024-12-01T20:00:00Z", "aaa-8k-recent", "aaa-8k-recent", 0),
        _chunk("AAA", "sec_8k", "2024-07-01T20:00:00Z", "aaa-8k-old", "aaa-8k-old", 0),
        _chunk("AAA", "sec_8k", "2025-01-01T15:00:00Z", "aaa-8k-future", "aaa-8k-future", 0),
        _chunk("AAA", "sec_10k", "2023-02-01T20:00:00Z", "aaa-10k-old", "aaa-10k-old", 0),
        _chunk("AAA", "sec_10k", "2024-02-01T20:00:00Z", "aaa-10k-latest", "aaa-10k-latest", 0),
        _chunk("AAA", "earnings_transcript", "2024-12-15T14:30:00Z", "aaa-call-3", "aaa-call-3", 0),
        _chunk("AAA", "earnings_transcript", "2024-11-15T14:30:00Z", "aaa-call-2", "aaa-call-2", 0),
        _chunk("AAA", "earnings_transcript", "2024-10-01T14:30:00Z", "aaa-call-1", "aaa-call-1", 0),
        _chunk("AAA", "earnings_transcript", "2025-01-01T14:30:00Z", "aaa-call-future", "aaa-call-future", 0),
        _chunk("BBB", "sec_8k", "2024-12-02T20:00:00Z", "bbb-8k-recent", "bbb-8k-recent", 0),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ingest_frame(chunks, client, config, FakeEncoder())

    signal = {
        "signal_id": "signal-1",
        "pair": ("AAA", "BBB"),
        "t_signal": "2024-12-31T21:00:00Z",
        "direction": "long_spread",
        "zscore": 2.4,
        "model_version": "walk-forward-v1",
    }
    context = retrieve_context(
        signal,
        client=client,
        config=RetrievalConfig(
            collection_name="pit-context",
            top_k_per_leg=10,
            recency_8k_days=90,
        ),
        encoder=FakeEncoder(),
    )

    aaa = context["legs"]["AAA"]
    ids = {chunk["payload"]["chunk_id"] for chunk in aaa}
    assert "aaa-8k-recent#000" in ids
    assert "aaa-8k-old#000" not in ids
    assert "aaa-8k-future#000" not in ids
    assert "aaa-10k-latest#000" in ids
    assert "aaa-10k-old#000" not in ids
    assert "aaa-call-3#000" in ids
    assert "aaa-call-2#000" in ids
    assert "aaa-call-1#000" not in ids
    assert "aaa-call-future#000" not in ids
    assert all(chunk["ticker"] == "AAA" for chunk in aaa)
    assert all(
        pd.Timestamp(chunk["available_at"]) <= pd.Timestamp(signal["t_signal"])
        for chunk in aaa
    )
    assert context["legs"]["BBB"]
    client.close()


def test_retrieval_excludes_filing_just_before_its_acceptance_time():
    client = QdrantClient(":memory:")
    config = IndexerConfig(collection_name="acceptance-cutoff")
    chunk = _chunk(
        "AAA",
        "sec_8k",
        "2024-05-01T19:30:00Z",
        "aaa-known-filing",
        "aaa-known-filing",
        0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ingest_frame(pd.DataFrame([chunk]), client, config, FakeEncoder())

    signal = {
        "signal_id": "before-filing",
        "pair": ("AAA", "BBB"),
        "t_signal": "2024-05-01T19:29:59Z",
        "direction": "long_spread",
        "zscore": 2.0,
        "model_version": "walk-forward-v1",
    }
    context = retrieve_context(
        signal,
        client=client,
        config=RetrievalConfig(collection_name="acceptance-cutoff"),
        encoder=FakeEncoder(),
    )

    assert context["legs"]["AAA"] == []
    client.close()


def test_chronological_signal_replay_reports_zero_lookahead_violations():
    client = QdrantClient(":memory:")
    index_config = IndexerConfig(collection_name="chronological-replay")
    chunks = pd.DataFrame([
        _chunk(
            "AAA", "sec_8k", "2024-05-01T19:30:00Z",
            "aaa-known-filing", "aaa-known-filing", 0,
        ),
        _chunk(
            "AAA", "earnings_transcript", "2024-05-15T13:30:00Z",
            "aaa-call-1", "aaa-call-1", 0,
        ),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ingest_frame(chunks, client, index_config, FakeEncoder())

    def signal(signal_id, t_signal):
        return {
            "signal_id": signal_id,
            "pair": ("AAA", "BBB"),
            "t_signal": t_signal,
            "direction": "long_spread",
            "zscore": 2.0,
            "model_version": "walk-forward-v1",
        }

    report = replay_signals(
        [
            signal("after", "2024-05-01T19:30:00Z"),
            signal("before", "2024-05-01T19:29:59Z"),
            signal("later", "2024-05-30T20:00:00Z"),
        ],
        client,
        RetrievalConfig(collection_name="chronological-replay"),
        FakeEncoder(),
    )

    assert report["signals_tested"] == 3
    assert report["retrievals"] == 3
    assert report["violations"] == 0
    assert report["estimated_retrieval_share"] == 1 / 3
    client.close()
