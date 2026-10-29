"""Unit tests for RAG metadata schema and chunk payload validation."""

from datetime import date, datetime, timezone
import pytest
from pydantic import ValidationError

from src.rag.chunking.meta_data_schema import (
    BaseChunkMetadata,
    ChunkPayload,
    SEC8KMetadata,
    SEC10KMetadata,
    SEC10QMetadata,
    TranscriptMetadata,
)


def test_base_chunk_metadata_normalizes_to_utc():
    ny_time = datetime.fromisoformat("2024-05-01T15:00:00-04:00")
    meta = BaseChunkMetadata(
        source_type="sec_8k",
        ticker="AAPL",
        published_at=ny_time,
        available_at=ny_time,
    )
    assert meta.published_at.tzinfo == timezone.utc
    assert meta.published_at.isoformat() == "2024-05-01T19:00:00+00:00"


def test_sec_8k_metadata_validation():
    meta = SEC8KMetadata(
        ticker="MSFT",
        company_name="Microsoft Corp",
        cik=789019,
        accession_number="0000789019-24-000010",
        form="8-K",
        filing_date=date(2024, 4, 25),
        published_at=datetime(2024, 4, 25, 20, 30, tzinfo=timezone.utc),
        available_at=datetime(2024, 4, 25, 20, 30, tzinfo=timezone.utc),
        item_code="2.02",
        item_title="Results of Operations",
    )
    assert meta.source_type == "sec_8k"
    assert meta.form == "8-K"
    assert meta.cik == 789019
    assert meta.item_code == "2.02"


def test_sec_10q_metadata_fiscal_period():
    meta = SEC10QMetadata(
        ticker="GOOGL",
        company_name="Alphabet Inc",
        cik=1652044,
        accession_number="0001652044-24-000020",
        form="10-Q",
        filing_date=date(2024, 7, 23),
        published_at=datetime(2024, 7, 23, 21, 0, tzinfo=timezone.utc),
        available_at=datetime(2024, 7, 23, 21, 0, tzinfo=timezone.utc),
        fiscal_year=2024,
        fiscal_period="Q2",
    )
    assert meta.source_type == "sec_10q"
    assert meta.fiscal_period == "Q2"


def test_transcript_metadata_validation():
    published = datetime(2024, 7, 30, 21, 30, tzinfo=timezone.utc)
    meta = TranscriptMetadata(
        ticker="AAPL",
        cik=320193,
        transcript_id="AAPL:2024-Q3",
        fiscal_quarter="2024-Q3",
        published_at=published,
        available_at=published,
        availability_estimated=False,
        availability_basis="provider_published_at",
        section="prepared_remarks",
        speaker_name="Tim Cook",
        speaker_role="CEO",
        turn_index=0,
    )
    assert meta.source_type == "earnings_transcript"
    assert meta.ticker == "AAPL"
    assert meta.speaker_role == "CEO"
    assert meta.turn_index == 0


def test_chunk_payload_valid():
    meta = SEC8KMetadata(
        ticker="NVDA",
        company_name="Nvidia Corp",
        cik=1045810,
        accession_number="0001045810-24-000005",
        form="8-K",
        filing_date=date(2024, 8, 28),
        published_at=datetime(2024, 8, 28, 20, 20, tzinfo=timezone.utc),
        available_at=datetime(2024, 8, 28, 20, 20, tzinfo=timezone.utc),
    )
    payload = ChunkPayload(
        chunk_id="NVDA:0001#000",
        document_id="NVDA:0001",
        text="Sample filing chunk text with financial information.",
        word_count=7,
        chunk_index=0,
        total_chunks=1,
        metadata=meta,
    )
    dumped = payload.model_dump(mode="json")
    assert dumped["chunk_id"] == "NVDA:0001#000"
    assert dumped["metadata"]["source_type"] == "sec_8k"


def test_chunk_payload_invalid_constraints():
    meta = SEC8KMetadata(
        ticker="NVDA",
        company_name="Nvidia Corp",
        cik=1045810,
        accession_number="0001045810-24-000005",
        form="8-K",
        filing_date=date(2024, 8, 28),
        published_at=datetime(2024, 8, 28, 20, 20, tzinfo=timezone.utc),
        available_at=datetime(2024, 8, 28, 20, 20, tzinfo=timezone.utc),
    )
    # total_chunks must be > 0
    with pytest.raises(ValidationError):
        ChunkPayload(
            chunk_id="NVDA:0001#000",
            document_id="NVDA:0001",
            text="text",
            word_count=1,
            chunk_index=0,
            total_chunks=0,
            metadata=meta,
        )

    # word_count must be >= 0
    with pytest.raises(ValidationError):
        ChunkPayload(
            chunk_id="NVDA:0001#000",
            document_id="NVDA:0001",
            text="text",
            word_count=-5,
            chunk_index=0,
            total_chunks=1,
            metadata=meta,
        )
