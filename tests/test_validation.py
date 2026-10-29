"""Unit tests for src.rag.validation.

Tests cover:
  - check_timestamps: pass on matching available_at, fail when available_at < acceptance
  - check_identity: pass with unique keys, fail with duplicate (cik, acc, doc_canonical, chunk_index)
  - check_ingestion_status: pass/fail/disabled logic
  - check_retrieval_assertion: skips gracefully when collection missing
  - _strip_ticker_prefix: canonical document_id logic
  - _parse_raw_acceptance: DST boundary (same as existing test, but within this module)
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from src.rag.validation import (
    CheckResult,
    ValidationConfig,
    ValidationError,
    _offset_histogram,
    _parse_raw_acceptance,
    _strip_ticker_prefix,
    check_identity,
    check_ingestion_status,
    check_retrieval_assertion,
    check_timestamps,
    run_validation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sec_chunk(
    *,
    ticker: str = "AAPL",
    accession_number: str = "0000320193-21-000063",
    cik: int = 320193,
    document_id: str = "AAPL:0000320193-21-000063:item_section",
    chunk_index: int = 0,
    available_at: str = "2021-07-27T20:35:44+00:00",
    source_type: str = "sec_8k",
    chunk_id: str | None = None,
    content_hash: str = "abc123",
) -> dict:
    return {
        "ticker": ticker,
        "accession_number": accession_number,
        "cik": cik,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "available_at": available_at,
        "source_type": source_type,
        "chunk_id": chunk_id or f"{ticker}:{accession_number}:item_section#{chunk_index:03d}",
        "content_hash": content_hash,
        "text": "sample text",
    }


# ---------------------------------------------------------------------------
# _strip_ticker_prefix
# ---------------------------------------------------------------------------

class TestStripTickerPrefix:
    def test_strips_ticker_prefix(self):
        doc_id = "AAPL:0000320193-21-000063:item_section"
        acc = "0000320193-21-000063"
        canonical = _strip_ticker_prefix(doc_id, acc)
        assert canonical == "0000320193-21-000063:item_section"

    def test_googl_strips_to_same_as_goog(self):
        acc = "0001652044-24-000022"
        goog_doc = f"GOOG:{acc}:Item-1A"
        googl_doc = f"GOOGL:{acc}:Item-1A"
        assert _strip_ticker_prefix(goog_doc, acc) == _strip_ticker_prefix(googl_doc, acc)

    def test_no_ticker_prefix_unchanged(self):
        doc_id = "0000320193-21-000063:item_section"
        acc = "0000320193-21-000063"
        assert _strip_ticker_prefix(doc_id, acc) == doc_id

    def test_empty_string(self):
        assert _strip_ticker_prefix("", "0000320193-21-000063") == ""


# ---------------------------------------------------------------------------
# _parse_raw_acceptance
# ---------------------------------------------------------------------------

class TestParseRawAcceptance:
    def test_parses_eastern_to_utc_summer(self, tmp_path):
        """20210727163544 = 2021-07-27 16:35:44 ET (EDT = UTC-4) → 20:35:44 UTC."""
        raw = tmp_path / "filing.txt"
        raw.write_text("<ACCEPTANCE-DATETIME>20210727163544\n<FORM-TYPE>8-K\n")
        result = _parse_raw_acceptance(raw)
        assert result is not None
        assert result == datetime(2021, 7, 27, 20, 35, 44, tzinfo=timezone.utc)

    def test_parses_eastern_to_utc_winter(self, tmp_path):
        """20220115143000 = 2022-01-15 14:30:00 ET (EST = UTC-5) → 19:30:00 UTC."""
        raw = tmp_path / "filing.txt"
        raw.write_text("<ACCEPTANCE-DATETIME>20220115143000\n<FORM-TYPE>10-K\n")
        result = _parse_raw_acceptance(raw)
        assert result is not None
        assert result == datetime(2022, 1, 15, 19, 30, 0, tzinfo=timezone.utc)

    def test_dst_boundary_spring_forward(self, tmp_path):
        """2024-03-10 03:00 does not exist in ET (spring forward); ZoneInfo fold=0."""
        # 2024-03-10 02:30 ET does not exist. 02:30 is in the gap.
        # We use 03:00 AM on 2024-03-10 (already EST->EDT boundary).
        # The 'before' time: 01:59 ET = 06:59 UTC (EST, UTC-5).
        # The 'after' time:  03:01 ET = 07:01 UTC (EDT, UTC-4).
        raw = tmp_path / "spring.txt"
        raw.write_text("<ACCEPTANCE-DATETIME>20240310010000\n")  # 01:00 ET = 06:00 UTC (EST)
        result = _parse_raw_acceptance(raw)
        assert result == datetime(2024, 3, 10, 6, 0, 0, tzinfo=timezone.utc)

    def test_returns_none_for_missing_file(self, tmp_path):
        result = _parse_raw_acceptance(tmp_path / "nonexistent.txt")
        assert result is None

    def test_returns_none_when_tag_absent(self, tmp_path):
        raw = tmp_path / "filing.txt"
        raw.write_text("No ACCEPTANCE-DATETIME tag here.\n")
        result = _parse_raw_acceptance(raw)
        assert result is None


# ---------------------------------------------------------------------------
# _offset_histogram
# ---------------------------------------------------------------------------

class TestOffsetHistogram:
    def test_buckets_sum_to_total(self):
        offsets = [-1.0, 0.0, 30.0, 120.0, 3600.0, 50000.0, 100000.0]
        hist = _offset_histogram(offsets)
        assert sum(hist.values()) == len(offsets)

    def test_violation_bucket(self):
        hist = _offset_histogram([-5.0, -0.001])
        assert hist["< 0 s  [VIOLATION]"] == 2


# ---------------------------------------------------------------------------
# check_timestamps
# ---------------------------------------------------------------------------

class TestCheckTimestamps:
    def _make_manifest(self, tmp_path: Path, accession: str, raw_rel: str, acc_dt: str) -> None:
        manifest_dir = tmp_path / "raw" / "8k-sec"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "accession_number": accession,
            "raw_path": raw_rel,
            "acceptance_datetime": acc_dt,
        }
        (manifest_dir / "manifest.jsonl").write_text(json.dumps(record) + "\n")

    def _make_raw_file(self, tmp_path: Path, cik: str, accession: str, raw_dt: str) -> str:
        """Create a raw SGML file and return the path relative to lake_root."""
        cik_padded = cik.zfill(10)
        filing_dir = tmp_path / "raw" / "8k-sec" / "filings" / cik_padded
        filing_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{accession}.txt"
        (filing_dir / fname).write_text(f"<ACCEPTANCE-DATETIME>{raw_dt}\n<FORM-TYPE>8-K\n")
        return f"raw/8k-sec/filings/{cik_padded}/{fname}"

    def test_pass_when_available_at_matches(self, tmp_path):
        raw_rel = self._make_raw_file(tmp_path, "320193", "0000320193-21-000063", "20210727163544")
        self._make_manifest(tmp_path, "0000320193-21-000063", raw_rel, "2021-07-27T20:35:44+00:00")
        chunk = _make_sec_chunk(available_at="2021-07-27T20:35:44+00:00")
        df = pd.DataFrame([chunk])
        result = check_timestamps(df, tmp_path, mismatch_threshold=0.0)
        assert result.passed is True
        assert result.counts["violations"] == 0

    def test_fail_when_available_at_before_acceptance(self, tmp_path):
        raw_rel = self._make_raw_file(tmp_path, "320193", "0000320193-21-000063", "20210727163544")
        self._make_manifest(tmp_path, "0000320193-21-000063", raw_rel, "2021-07-27T20:35:44+00:00")
        # available_at is 30 min BEFORE acceptance (2021-07-27 20:05 UTC)
        chunk = _make_sec_chunk(available_at="2021-07-27T20:05:00+00:00")
        df = pd.DataFrame([chunk])
        result = check_timestamps(df, tmp_path, mismatch_threshold=0.0)
        assert result.passed is False
        assert result.counts["violations"] == 1
        v = result.violations[0]
        assert v["corrected_available_at"] == "2021-07-27T20:35:44+00:00"

    def test_pass_within_threshold(self, tmp_path):
        """One violation out of 100 chunks, threshold=2% → PASS."""
        raw_rel = self._make_raw_file(tmp_path, "320193", "0000320193-21-000063", "20210727163544")
        self._make_manifest(tmp_path, "0000320193-21-000063", raw_rel, "2021-07-27T20:35:44+00:00")
        good = _make_sec_chunk(available_at="2021-07-27T20:35:44+00:00")
        bad = _make_sec_chunk(available_at="2021-07-27T20:05:00+00:00")
        rows = [good] * 99 + [bad]
        df = pd.DataFrame(rows)
        result = check_timestamps(df, tmp_path, mismatch_threshold=0.02)
        assert result.passed is True   # 1/100 = 1% < 2% threshold

    def test_no_sec_chunks(self, tmp_path):
        df = pd.DataFrame([{"source_type": "earnings_transcript", "available_at": "2021-01-01T00:00:00Z"}])
        result = check_timestamps(df, tmp_path)
        assert result.passed is True
        assert "No SEC chunks" in result.notes


# ---------------------------------------------------------------------------
# check_identity
# ---------------------------------------------------------------------------

class TestCheckIdentity:
    def test_pass_unique_keys(self):
        rows = [
            _make_sec_chunk(ticker="AAPL", chunk_index=0),
            _make_sec_chunk(ticker="AAPL", chunk_index=1),
            _make_sec_chunk(ticker="AAPL", chunk_index=2),
        ]
        df = pd.DataFrame(rows)
        result = check_identity(df)
        assert result.passed is True
        assert result.counts["duplicate_key_groups"] == 0

    def test_fail_dual_class_duplicate(self):
        """GOOG and GOOGL indexing same filing → same canonical doc_id → duplicate."""
        acc = "0001652044-24-000022"
        cik = 1652044
        goog = _make_sec_chunk(
            ticker="GOOG", accession_number=acc, cik=cik, chunk_index=0,
            document_id=f"GOOG:{acc}:Item-1A", content_hash="hash_a",
        )
        googl = _make_sec_chunk(
            ticker="GOOGL", accession_number=acc, cik=cik, chunk_index=0,
            document_id=f"GOOGL:{acc}:Item-1A", content_hash="hash_a",
        )
        df = pd.DataFrame([goog, googl])
        result = check_identity(df)
        assert result.passed is False
        assert result.counts["duplicate_key_groups"] == 1
        v = result.violations[0]
        assert "GOOG" in v["tickers"] and "GOOGL" in v["tickers"]
        assert v["same_content"] is True

    def test_different_chunk_indices_not_duplicate(self):
        acc = "0001652044-24-000022"
        goog = _make_sec_chunk(ticker="GOOG", accession_number=acc, chunk_index=0,
                               document_id=f"GOOG:{acc}:Item-1A")
        googl = _make_sec_chunk(ticker="GOOGL", accession_number=acc, chunk_index=1,
                                document_id=f"GOOGL:{acc}:Item-1A")
        df = pd.DataFrame([goog, googl])
        result = check_identity(df)
        # Different chunk_index → not a duplicate even if same doc prefix
        assert result.counts["duplicate_key_groups"] == 0

    def test_empty_frame(self):
        df = pd.DataFrame()
        result = check_identity(df)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_ingestion_status
# ---------------------------------------------------------------------------

class TestCheckIngestionStatus:
    def _make_universe(self, tickers: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"ticker": tickers})

    def _make_status_parquet(self, tmp_path: Path, rows: list[dict]) -> Path:
        path = tmp_path / "ingestion_status.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def test_pass_all_tickers_fetched(self, tmp_path):
        universe = self._make_universe(["AAPL", "MSFT"])
        parquet = self._make_status_parquet(tmp_path, [
            {"ticker": "AAPL", "source": "sec", "chunk_count": 50, "status_updated_at": "2021-01-01"},
            {"ticker": "MSFT", "source": "sec", "chunk_count": 30, "status_updated_at": "2021-01-01"},
        ])
        result = check_ingestion_status(universe, status_parquet=parquet)
        assert result.passed is True
        assert result.counts["sec_fetched"] == 2
        assert result.counts["sec_missing"] == 0

    def test_fail_missing_ticker(self, tmp_path):
        universe = self._make_universe(["AAPL", "HOOD"])
        parquet = self._make_status_parquet(tmp_path, [
            {"ticker": "AAPL", "source": "sec", "chunk_count": 50, "status_updated_at": "2021-01-01"},
            # HOOD missing
        ])
        result = check_ingestion_status(universe, status_parquet=parquet)
        assert result.passed is False
        assert result.counts["sec_missing"] == 1
        assert result.violations[0]["ticker"] == "HOOD"

    def test_fail_zero_chunk_count(self, tmp_path):
        universe = self._make_universe(["AAPL"])
        parquet = self._make_status_parquet(tmp_path, [
            {"ticker": "AAPL", "source": "sec", "chunk_count": 0, "status_updated_at": "2021-01-01"},
        ])
        result = check_ingestion_status(universe, status_parquet=parquet)
        assert result.passed is False
        assert "chunk_count = 0" in result.violations[0]["reason"]

    def test_transcripts_always_disabled(self, tmp_path):
        universe = self._make_universe(["AAPL"])
        parquet = self._make_status_parquet(tmp_path, [
            {"ticker": "AAPL", "source": "sec", "chunk_count": 10, "status_updated_at": "2021-01-01"},
            {"ticker": "AAPL", "source": "transcripts", "chunk_count": 0, "status_updated_at": "2021-01-01"},
        ])
        result = check_ingestion_status(universe, status_parquet=parquet)
        assert result.passed is True
        assert result.counts["transcripts_disabled"] == 1

    def test_empty_universe(self, tmp_path):
        result = check_ingestion_status(pd.DataFrame(columns=["ticker"]))
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_retrieval_assertion
# ---------------------------------------------------------------------------

class TestCheckRetrievalAssertion:
    def test_skips_when_collection_missing(self):
        """Should return PASS (not fail) when the collection doesn't exist."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False

        with patch("src.rag.validation.QdrantClient", return_value=mock_client):
            result = check_retrieval_assertion("nonexistent_collection", sample_size=1)
        assert result.passed is True
        assert "does not exist" in result.notes

    def test_skips_empty_collection(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = MagicMock(count=0)

        with patch("src.rag.validation.QdrantClient", return_value=mock_client):
            result = check_retrieval_assertion("empty_collection", sample_size=1)
        assert result.passed is True
        assert "empty" in result.notes

    def test_pass_no_pre_filing_hits(self):
        """Before available_at → 0 hits; after → 1 hit → PASS."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.count.side_effect = [
            MagicMock(count=10),   # total
            MagicMock(count=0),    # before available_at
            MagicMock(count=5),    # after available_at
        ]
        point = MagicMock()
        point.id = "point-1"
        point.payload = {
            "available_at": "2021-07-27T20:35:44+00:00",
            "accession_number": "0000320193-21-000063",
        }
        mock_client.scroll.return_value = ([point], None)

        with patch("src.rag.validation.QdrantClient", return_value=mock_client):
            result = check_retrieval_assertion("test_collection", sample_size=1)
        assert result.passed is True
        assert result.counts["pre_filing_violations"] == 0

    def test_fail_pre_filing_hit(self):
        """Before available_at → 1 hit → FAIL (future leakage)."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.count.side_effect = [
            MagicMock(count=10),   # total
            MagicMock(count=1),    # before available_at → violation
            MagicMock(count=5),    # after available_at
        ]
        point = MagicMock()
        point.id = "point-1"
        point.payload = {
            "available_at": "2021-07-27T20:35:44+00:00",
            "accession_number": "0000320193-21-000063",
        }
        mock_client.scroll.return_value = ([point], None)

        with patch("src.rag.validation.QdrantClient", return_value=mock_client):
            result = check_retrieval_assertion("test_collection", sample_size=1)
        assert result.passed is False
        assert result.counts["pre_filing_violations"] == 1
        assert "leakage" in result.violations[0]["error"].lower()


# ---------------------------------------------------------------------------
# run_validation (integration, fully mocked I/O)
# ---------------------------------------------------------------------------

class TestRunValidation:
    def _make_config(self, tmp_path: Path) -> ValidationConfig:
        return ValidationConfig(
            lake_root=tmp_path / "lake",
            collection_name="test_chunks",
            fail_on_violation=True,
        )

    def test_all_pass_returns_passed_true(self, tmp_path):
        """With no chunks and no universe, all checks trivially pass."""
        config = self._make_config(tmp_path)
        # No parquet files → checks skip gracefully
        with patch("src.rag.validation.check_retrieval_assertion") as mock_ret:
            mock_ret.return_value = CheckResult(name="retrieval_assertion", passed=True, counts={})
            report = run_validation(config)
        assert report["passed"] is True

    def test_raises_validation_error_on_failure(self, tmp_path):
        """A failed check raises ValidationError when fail_on_violation=True."""
        config = self._make_config(tmp_path)
        with patch("src.rag.validation.check_timestamps") as mock_ts, \
             patch("src.rag.validation.check_identity") as mock_id, \
             patch("src.rag.validation.check_ingestion_status") as mock_st, \
             patch("src.rag.validation.check_retrieval_assertion") as mock_ret:

            mock_ts.return_value = CheckResult(
                name="timestamp", passed=False,
                counts={"violations": 5, "sec_chunks_evaluated": 10,
                        "mismatch_rate": 0.5, "threshold": 0.02,
                        "parse_failures": 0, "offset_histogram": {}},
                violations=[{"accession_number": "0000320193-21-000063"}],
            )
            mock_id.return_value = CheckResult(name="identity", passed=True, counts={})
            mock_st.return_value = CheckResult(name="ingestion_status", passed=True, counts={})
            mock_ret.return_value = CheckResult(name="retrieval_assertion", passed=True, counts={})

            with pytest.raises(ValidationError, match="timestamp"):
                run_validation(config)

    def test_no_fail_returns_report_with_failures(self, tmp_path):
        """fail_on_violation=False returns report without raising."""
        config = ValidationConfig(
            lake_root=tmp_path / "lake",
            collection_name="test_chunks",
            fail_on_violation=False,
        )
        with patch("src.rag.validation.check_timestamps") as mock_ts, \
             patch("src.rag.validation.check_identity") as mock_id, \
             patch("src.rag.validation.check_ingestion_status") as mock_st, \
             patch("src.rag.validation.check_retrieval_assertion") as mock_ret:

            mock_ts.return_value = CheckResult(name="timestamp", passed=False, counts={})
            mock_id.return_value = CheckResult(name="identity", passed=True, counts={})
            mock_st.return_value = CheckResult(name="ingestion_status", passed=True, counts={})
            mock_ret.return_value = CheckResult(name="retrieval_assertion", passed=True, counts={})

            report = run_validation(config)

        assert report["passed"] is False   # overall fail
        assert report["checks"]["timestamp"]["passed"] is False
        assert report["checks"]["identity"]["passed"] is True


# ---------------------------------------------------------------------------
# build_ingestion_status status column (coverage.py)
# ---------------------------------------------------------------------------

class TestBuildIngestionStatusStatusColumn:
    """Smoke test that build_ingestion_status now returns a 'status' column."""

    def test_status_fetched_when_chunks_present(self):
        from src.rag.coverage import build_ingestion_status
        chunks = pd.DataFrame([
            {"ticker": "AAPL", "source_type": "sec_8k", "available_at": "2021-01-01T00:00:00Z"},
        ])
        result = build_ingestion_status(chunks, tickers=["AAPL"])
        aapl_sec = result[(result["ticker"] == "AAPL") & (result["source"] == "sec")]
        assert aapl_sec["status"].iloc[0] == "fetched"

    def test_status_disabled_for_transcripts(self):
        from src.rag.coverage import build_ingestion_status
        chunks = pd.DataFrame([
            {"ticker": "AAPL", "source_type": "sec_8k", "available_at": "2021-01-01T00:00:00Z"},
        ])
        result = build_ingestion_status(chunks, tickers=["AAPL"])
        aapl_trans = result[(result["ticker"] == "AAPL") & (result["source"] == "transcripts")]
        assert aapl_trans["status"].iloc[0] == "disabled"

    def test_status_missing_for_no_data(self):
        from src.rag.coverage import build_ingestion_status
        result = build_ingestion_status(pd.DataFrame(), tickers=["HOOD"])
        hood_sec = result[(result["ticker"] == "HOOD") & (result["source"] == "sec")]
        assert hood_sec["status"].iloc[0] == "missing"
