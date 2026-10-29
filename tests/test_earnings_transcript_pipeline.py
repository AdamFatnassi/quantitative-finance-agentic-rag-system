import json
from pathlib import Path

import pandas as pd

from src.rag.cleaning.earnings_transcript_cleaner import (
    CleanerConfig,
    EarningsTranscriptCleaner,
    clean_transcript,
)
from src.rag.cleaning.transcript_availability import (
    next_trading_open_after_call,
    quarter_fallback_open,
)
from src.rag.scraping.earnings_transcript_fetcher import normalize_quarter, normalize_ticker


def _record() -> dict:
    return {
        "ticker": "ACME",
        "cik": 1234567,
        "quarter": "2026Q3",
        "status": "ok",
        "source_url": "https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=ACME&quarter=2026Q3",
        "raw_path": "raw/earnings-transcripts/ACME/2026Q3.json",
        "fetched_at_utc": "2026-10-01T12:00:00+00:00",
    }


def _payload() -> dict:
    return {
        "symbol": "ACME",
        "quarter": "2026Q3",
        "transcript": [
            {
                "speaker": "Jordan Lee",
                "title": "Chief Executive Officer",
                "content": "  Revenue grew.\r\n\r\n  Costs were stable. ",
                "sentiment": "Bullish",
            },
            {
                "speaker": "Taylor Kim",
                "title": "Analyst, Example Research",
                "content": "What drove the growth?",
                "sentiment": "Neutral",
            },
        ],
    }


def test_transcript_without_provider_time_uses_conservative_quarter_fallback() -> None:
    turns = clean_transcript(_record(), _payload(), CleanerConfig())

    assert len(turns) == 2
    assert turns[0]["fiscal_quarter"] == "2026-Q3"
    assert turns[0]["section"] == "prepared_remarks"
    assert turns[1]["section"] == "q_and_a"
    assert turns[0]["speaker_role"] == "CEO"
    assert turns[1]["speaker_role"] == "analyst"
    assert turns[0]["text"] == "Revenue grew.\n\nCosts were stable."
    assert turns[0]["published_at"] is None
    expected = quarter_fallback_open(2026, 3).isoformat()
    assert turns[0]["available_at"] == expected
    assert turns[0]["availability_estimated"] is True
    assert turns[0]["availability_basis"] == "quarter_end_plus_90_days"
    assert turns[0]["metadata"]["source_type"] == "earnings_transcript"
    assert turns[0]["metadata"]["retrieved_at"] == "2026-10-01T12:00:00Z"
    assert turns[0]["metadata"]["available_at"] == expected.replace("+00:00", "Z")


def test_provider_publication_time_is_used_when_present() -> None:
    payload = {**_payload(), "published_at": "2026-09-28T16:30:00-04:00"}

    turns = clean_transcript(_record(), payload, CleanerConfig())

    assert turns[0]["available_at"] == "2026-09-28T20:30:00+00:00"
    assert turns[0]["availability_basis"] == "provider_published_at"
    assert turns[0]["availability_estimated"] is False


def test_explicit_call_datetime_uses_next_xnys_open() -> None:
    payload = {**_payload(), "call_datetime": "2026-09-28T16:30:00-04:00"}

    turns = clean_transcript(_record(), payload, CleanerConfig())

    assert turns[0]["available_at"] == next_trading_open_after_call(
        "2026-09-28T16:30:00-04:00"
    ).isoformat()
    assert turns[0]["availability_estimated"] is True
    assert turns[0]["availability_basis"] == "call_datetime_lag"


def test_cleaner_reads_only_successful_manifest_entries_and_writes_parquet(tmp_path) -> None:
    lake_root = Path(tmp_path)
    raw_file = lake_root / "raw/earnings-transcripts/ACME/2026Q3.json"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text(json.dumps(_payload()), encoding="utf-8")
    raw_path = "raw/earnings-transcripts/ACME/2026Q3.json"
    manifest = lake_root / "raw/earnings-transcripts/manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    successful = {**_record(), "raw_path": raw_path}
    empty = {**successful, "quarter": "2026Q2", "status": "empty"}
    manifest.write_text(
        "\n".join(json.dumps(entry) for entry in (successful, empty)) + "\n",
        encoding="utf-8",
    )

    cleaner = EarningsTranscriptCleaner(CleanerConfig(lake_root=lake_root))
    frame = cleaner.run(write=True)

    assert len(frame) == 2
    assert frame["document_id"].nunique() == 1
    assert cleaner.out_path.is_file()
    restored = pd.read_parquet(cleaner.out_path)
    assert restored.iloc[1]["metadata"]["speaker_role"] == "analyst"


def test_scraper_input_normalization_rejects_invalid_path_components() -> None:
    assert normalize_ticker("brk.b") == "BRK-B"
    assert normalize_quarter("2026q3") == "2026Q3"
    for invalid in ("../AAPL", ""):
        try:
            normalize_ticker(invalid)
        except ValueError:
            continue
        raise AssertionError(f"invalid ticker accepted: {invalid!r}")
    try:
        normalize_quarter("2026Q5")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid quarter accepted")


def test_retrieval_time_is_never_used_as_transcript_availability() -> None:
    turns = clean_transcript(_record(), _payload(), CleanerConfig())

    assert turns[0]["available_at"] != _record()["fetched_at_utc"]
    assert turns[0]["available_at"] == quarter_fallback_open(2026, 3).isoformat()
