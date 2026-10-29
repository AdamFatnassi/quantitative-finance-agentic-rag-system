from datetime import datetime, timezone

import pandas as pd

from src.rag.chunking.earnings_transcript_chunker import EarningsTranscriptChunker
from src.rag.chunking.sec_chunker import ChunkerConfig


def test_transcript_turns_are_chunked_with_valid_metadata(tmp_path):
    published_at = datetime(2026, 10, 1, 12, tzinfo=timezone.utc).isoformat()
    metadata = {
        "source_type": "earnings_transcript",
        "ticker": "ACME",
        "cik": 1234567,
        "transcript_id": "ACME:2026-Q3",
        "related_tickers": [],
        "company_name": "Acme Corp",
        "published_at": published_at,
        "retrieved_at": published_at,
        "available_at": published_at,
        "availability_estimated": False,
        "section_type": "prepared_remarks",
        "form_type": None,
        "availability_basis": "provider_published_at",
        "source_url": "https://example.com/transcript",
        "raw_path": "raw/earnings-transcripts/ACME/2026Q3.json",
        "fiscal_quarter": "2026-Q3",
        "section": "prepared_remarks",
        "speaker_name": "Jordan Lee",
        "speaker_role": "CEO",
        "turn_index": 0,
        "sentiment": "Bullish",
    }
    frame = pd.DataFrame([{
        "document_id": "ACME:2026-Q3",
        "turn_id": "ACME:2026-Q3:turn-0000",
        "turn_index": 0,
        "ticker": "ACME",
        "cik": 1234567,
        "transcript_id": "ACME:2026-Q3",
        "fiscal_quarter": "2026-Q3",
        "section": "prepared_remarks",
        "speaker_name": "Jordan Lee",
        "speaker_role": "CEO",
        "available_at": published_at,
        "text": "Acme revenue grew in every region.",
        "metadata": metadata,
    }])
    chunker = EarningsTranscriptChunker(
        ChunkerConfig(
            lake_root=tmp_path,
            max_tokens=100,
            overlap_tokens=10,
            min_chunk_tokens=0,
        )
    )

    chunks = chunker.chunk_frame(frame)

    assert len(chunks) == 1
    chunk = chunks.iloc[0]
    assert chunk["chunk_id"] == "ACME:2026-Q3:turn-0000#000"
    assert chunk["document_id"] == "ACME:2026-Q3:turn-0000"
    assert chunk["metadata"]["source_type"] == "earnings_transcript"
    assert chunk["metadata"]["available_at"] == published_at.replace("+00:00", "Z")
    assert chunk["metadata"]["availability_estimated"] is False
    assert chunk["metadata"]["section_type"] == "prepared_remarks"
    assert chunk["metadata"]["form_type"] is None
    assert chunk["metadata"]["speaker_role"] == "CEO"
    assert "2026-Q3" in chunk["context_header"]
    assert chunk["embedding_text"].endswith(chunk["text"])
    assert chunk["point_id"] == chunker.chunk_frame(frame).iloc[0]["point_id"]


def test_chunker_writes_parquet_output(tmp_path):
    timestamp = "2026-10-01T12:00:00+00:00"
    metadata = {
        "source_type": "earnings_transcript",
        "ticker": "ACME",
        "cik": 1234567,
        "transcript_id": "ACME:2026-Q3",
        "related_tickers": [],
        "company_name": "Acme Corp",
        "published_at": timestamp,
        "retrieved_at": timestamp,
        "available_at": timestamp,
        "availability_estimated": False,
        "section_type": "prepared_remarks",
        "form_type": None,
        "availability_basis": "provider_published_at",
        "source_url": None,
        "raw_path": "raw/earnings-transcripts/ACME/2026Q3.json",
        "fiscal_quarter": "2026-Q3",
        "section": "prepared_remarks",
        "speaker_name": "Jordan Lee",
        "speaker_role": "CEO",
        "turn_index": 0,
        "sentiment": None,
    }
    chunker = EarningsTranscriptChunker(
        ChunkerConfig(
            lake_root=tmp_path,
            max_tokens=100,
            overlap_tokens=10,
            min_chunk_tokens=0,
        )
    )
    chunker.in_path.parent.mkdir(parents=True)
    pd.DataFrame([{
        "document_id": "ACME:2026-Q3",
        "turn_id": "ACME:2026-Q3:turn-0000",
        "turn_index": 0,
        "ticker": "ACME",
        "fiscal_quarter": "2026-Q3",
        "section": "prepared_remarks",
        "speaker_name": "Jordan Lee",
        "speaker_role": "CEO",
        "available_at": timestamp,
        "text": "Acme revenue grew in every region.",
        "metadata": metadata,
    }]).to_parquet(chunker.in_path, index=False)

    chunks = chunker.run()

    restored = pd.read_parquet(chunker.out_path)
    assert len(chunks) == len(restored) == 1
    assert restored.iloc[0]["metadata"]["source_type"] == "earnings_transcript"
