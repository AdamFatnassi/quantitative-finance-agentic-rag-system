from dataclasses import dataclass

import pandas as pd
import pytest
from qdrant_client import QdrantClient

from src.rag.qdrant.ingest_chunks import (
    CHUNK_FILES,
    IndexerConfig,
    ingest_frame,
    load_chunk_frames,
    persist_ingestion_status,
)


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        dense = [[float(index + 1), 0.5, 0.25] for index, _ in enumerate(texts)]
        sparse = [SparseVector([1, 5], [1.0, 0.25]) for _ in texts]
        return dense, sparse


def _chunk(ticker, document_id, *, accession_number=None, transcript_id=None):
    return {
        "ticker": ticker,
        "cik": 1234567,
        "available_at": pd.Timestamp("2024-05-01T20:00:00Z"),
        "published_at": pd.Timestamp("2024-05-01T20:00:00Z"),
        "section_type": "section",
        "form_type": "10-K",
        "item_code": "1A",
        "accession_number": accession_number,
        "transcript_id": transcript_id,
        "availability_estimated": False,
        "source_type": "sec_10k",
        "chunk_id": f"{document_id}#000",
        "document_id": document_id,
        "chunk_index": 0,
        "text": f"Evidence for {ticker}.",
        "embedding_text": f"Evidence for {ticker}.",
    }


def test_upsert_is_idempotent_and_skips_existing_point_ids(monkeypatch):
    client = QdrantClient(":memory:")
    encoder = FakeEncoder()
    config = IndexerConfig(collection_name="test-hybrid")
    indexed_fields = []
    create_payload_index = client.create_payload_index

    def record_payload_index(*args, **kwargs):
        indexed_fields.append(kwargs["field_name"])
        return create_payload_index(*args, **kwargs)

    monkeypatch.setattr(client, "create_payload_index", record_payload_index)
    frame = pd.DataFrame([
        _chunk("AAA", "AAA:accession:section", accession_number="AAA-accession"),
        _chunk("BBB", "BBB:accession:section", accession_number="BBB-accession"),
    ])

    with pytest.warns(UserWarning, match="Payload indexes have no effect"):
        first = ingest_frame(frame, client, config, encoder)
    second = ingest_frame(frame, client, config, encoder)

    assert first["indexed"] == 2
    assert first["point_count"] == 2
    assert second["indexed"] == 0
    assert second["skipped"] == 2
    assert second["point_count"] == 2
    assert encoder.calls == 1
    assert set(indexed_fields) == {"ticker", "tickers", "available_at", "section_type"}
    with pytest.raises(ValueError, match="different embedding models"):
        ingest_frame(
            frame,
            client,
            IndexerConfig(
                collection_name="test-hybrid",
                embedding_model="another-dense-model",
            ),
            encoder,
        )
    client.close()


def test_ticker_filter_indexes_only_new_universe_member(tmp_path):
    client = QdrantClient(":memory:")
    encoder = FakeEncoder()
    config = IndexerConfig(collection_name="ticker-scope")
    sec_path = tmp_path / "processed" / "8k-sec" / "chunks.parquet"
    transcript_path = (
        tmp_path / "processed" / "earnings-transcripts" / "chunks.parquet"
    )
    sec_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    sec_chunk = _chunk("AAA", "AAA:accession:section", accession_number="AAA-accession")
    sec_chunk.pop("transcript_id")
    transcript_chunk = _chunk(
        "BBB", "BBB:transcript:turn", transcript_id="BBB:2024-Q1"
    )
    transcript_chunk.pop("accession_number")
    pd.DataFrame([sec_chunk]).to_parquet(sec_path, index=False)
    pd.DataFrame([transcript_chunk]).to_parquet(transcript_path, index=False)

    initial = load_chunk_frames(tmp_path, ["AAA"])
    assert initial["ticker"].tolist() == ["AAA"]
    assert ingest_frame(initial, client, config, encoder)["indexed"] == 1

    added = load_chunk_frames(tmp_path, ["BBB"])
    assert added["ticker"].tolist() == ["BBB"]
    result = ingest_frame(added, client, config, encoder)

    assert result["indexed"] == 1
    assert result["point_count"] == 2
    points, _ = client.scroll(config.collection_name, limit=10, with_vectors=False)
    assert {point.payload["ticker"] for point in points} == {"AAA", "BBB"}
    client.close()


def test_default_chunk_loading_is_limited_to_filtered_universe(tmp_path):
    lake_root = tmp_path / "data" / "lake"
    universe_path = lake_root.parent / "universe.parquet"
    universe_path.parent.mkdir(parents=True)
    pd.DataFrame([{
        "ticker": "AAA",
        "cik": "0001234567",
        "sector": "Technology",
        "first_date": "2020-01-01",
        "last_date": "2024-12-31",
    }]).to_parquet(universe_path, index=False)

    sec_path = lake_root / CHUNK_FILES[0]
    transcript_path = lake_root / CHUNK_FILES[1]
    sec_path.parent.mkdir(parents=True)
    transcript_path.parent.mkdir(parents=True)
    pd.DataFrame([
        _chunk("AAA", "AAA:accession:section", accession_number="AAA-accession"),
        _chunk("OLD", "OLD:accession:section", accession_number="OLD-accession"),
    ]).to_parquet(sec_path, index=False)
    pd.DataFrame([
        _chunk("OLD", "OLD:transcript:turn", transcript_id="OLD:2024-Q1")
    ]).to_parquet(transcript_path, index=False)

    chunks = load_chunk_frames(lake_root)

    assert chunks["ticker"].tolist() == ["AAA"]


def test_ingestion_status_writer_upserts_source_ranges():
    class Connection:
        def execute(self, statement, rows):
            self.statement = str(statement)
            self.rows = rows

    class Engine:
        def __init__(self):
            self.connection = Connection()

        def begin(self):
            engine = self

            class Transaction:
                def __enter__(self):
                    return engine.connection

                def __exit__(self, *args):
                    return False

            return Transaction()

    engine = Engine()
    status = pd.DataFrame([{
        "ticker": "AAA",
        "source": "sec",
        "earliest_available_at": pd.Timestamp("2023-01-01T00:00:00Z"),
        "latest_available_at": pd.Timestamp("2024-01-01T00:00:00Z"),
        "chunk_count": 12,
    }])

    assert persist_ingestion_status(engine, status) == 1
    assert "ON CONFLICT (ticker, source) DO UPDATE" in engine.connection.statement
    assert engine.connection.rows[0]["chunk_count"] == 12
