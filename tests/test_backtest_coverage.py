from datetime import date

import pandas as pd
import pytest

from src.rag.coverage import (
    BacktestCoverageConfig,
    build_ingestion_status,
    coverage_for_signal,
    coverage_report,
    quarter_range,
)
from src.rag.qdrant.replay_validation import (
    check_replay_coverage,
    signals_from_frame,
)
from src.rag.coverage import normalize_signal_time


def test_backtest_ingestion_window_uses_longest_retrieval_scope():
    config = BacktestCoverageConfig(
        backtest_start=date(2022, 1, 1),
        backtest_end=date(2024, 12, 31),
        latest_10k_days=500,
        transcripts_per_leg=3,
        transcript_quarter_days=92,
    )

    assert config.max_retrieval_window_days == 500
    assert config.fetch_start == date(2020, 8, 19)


def test_quarter_range_includes_each_calendar_quarter_inclusive():
    assert quarter_range(date(2023, 11, 15), date(2024, 2, 1)) == [
        "2023Q4", "2024Q1",
    ]


def test_date_only_signal_uses_market_close_for_point_in_time_coverage():
    assert normalize_signal_time("2024-01-02") == pd.Timestamp("2024-01-02T21:00:00Z")


def test_signals_from_frame_keeps_active_signals_and_builds_unique_ids():
    signals = signals_from_frame(pd.DataFrame([
        {"time": "2024-01-02", "stock_a": "AAA", "stock_b": "BBB", "signal": 1},
        {"time": "2024-01-02", "stock_a": "CCC", "stock_b": "DDD", "signal": -1},
        {"time": "2024-01-02", "stock_a": "EEE", "stock_b": "FFF", "signal": 0},
    ]))

    assert len(signals) == 2
    assert signals[0]["signal_id"] != signals[1]["signal_id"]
    assert signals[0]["pair"] == ("AAA", "BBB")
    assert signals[0]["t_signal"] == "2024-01-02T21:00:00+00:00"


def test_transcript_backfill_defers_at_quarter_and_resumes_from_cache(tmp_path, monkeypatch):
    from src.rag.scraping.earnings_transcript_fetcher import (
        QuotaExhausted,
        TranscriptFetcher,
        FetcherConfig,
    )

    first = TranscriptFetcher(FetcherConfig(lake_root=tmp_path, api_key="test"))
    completed = []

    def exhaust_after_two(ticker, quarter):
        if len(completed) == 2:
            raise QuotaExhausted("daily budget")
        record = {
            "ticker": ticker,
            "quarter": quarter,
            "status": "empty",
        }
        first._manifest[(ticker, quarter)] = record
        completed.append(quarter)
        return record

    monkeypatch.setattr(first, "_get_or_fetch", exhaust_after_two)
    first.fetch(["AAA"], date(2024, 3, 31), start_date=date(2023, 7, 1))

    assert [quarter for _, quarter in first.deferred_work] == ["2024Q1"]
    assert first.history_depth_by_ticker["AAA"]["quarters_completed"] == 2
    assert first.history_depth_by_ticker["AAA"]["complete"] is False

    second = TranscriptFetcher(FetcherConfig(lake_root=tmp_path, api_key="test"))
    second._manifest.update(first._manifest)
    resumed = []

    def finish_remaining(ticker, quarter):
        cached = second._manifest.get((ticker, quarter))
        if cached:
            return cached
        resumed.append(quarter)
        record = {"ticker": ticker, "quarter": quarter, "status": "empty"}
        second._manifest[(ticker, quarter)] = record
        return record

    monkeypatch.setattr(second, "_get_or_fetch", finish_remaining)
    second.fetch(["AAA"], date(2024, 3, 31), start_date=date(2023, 7, 1))

    assert resumed == ["2024Q1"]
    assert second.history_depth_by_ticker["AAA"]["quarters_completed"] == 3
    assert second.history_depth_by_ticker["AAA"]["complete"] is True


def test_old_empty_transcript_quarters_do_not_starve_daily_resume(tmp_path, monkeypatch):
    from src.rag.scraping.earnings_transcript_fetcher import (
        FetcherConfig,
        TranscriptFetcher,
    )

    fetcher = TranscriptFetcher(FetcherConfig(lake_root=tmp_path, api_key="test"))
    cached = {
        "ticker": "AAA",
        "quarter": "2020Q1",
        "status": "empty",
        "fetched_at_utc": "2020-04-01T00:00:00+00:00",
    }
    fetcher._manifest[("AAA", "2020Q1")] = cached
    monkeypatch.setattr(
        fetcher,
        "_fetch_one",
        lambda *_: pytest.fail("finalized historical empty quarter was refetched"),
    )

    assert fetcher._get_or_fetch("AAA", "2020Q1") == cached


def test_coverage_report_produces_full_partial_none_and_enforces_gate():
    chunks = pd.DataFrame([
        {"ticker": ticker, "source_type": source, "available_at": timestamp,
         "transcript_id": doc_id, "document_id": doc_id}
        for ticker, source, timestamp, doc_id in (
            ("AAA", "sec_8k", "2024-12-01T20:00:00Z", "a-8k"),
            ("AAA", "sec_10q", "2024-11-01T20:00:00Z", "a-10q"),
            ("AAA", "sec_10k", "2024-02-01T20:00:00Z", "a-10k"),
            ("AAA", "earnings_transcript", "2024-11-01T20:00:00Z", "a-call-1"),
            ("AAA", "earnings_transcript", "2024-08-01T20:00:00Z", "a-call-2"),
            ("BBB", "sec_10k", "2024-02-01T20:00:00Z", "b-10k"),
            ("CCC", "sec_8k", "2025-01-01T20:00:00Z", "c-future"),
        )
    ])
    signals = [{
        "signal_id": "s1",
        "pair": ("AAA", "BBB"),
        "t_signal": "2024-12-31T21:00:00Z",
    }, {
        "signal_id": "s2",
        "pair": ("CCC", "DDD"),
        "t_signal": "2024-12-31T21:00:00Z",
    }]

    report = coverage_report(
        signals,
        chunks,
        max_none_coverage_share=0.5,
        failure_mode="warn",
    )

    assert report["distribution"] == {"full": 1, "partial": 1, "none": 2}
    assert report["none_share"] == 0.5
    assert [row["coverage"] for row in report["by_leg"]] == [
        "full", "partial", "none", "none",
    ]
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        coverage_report(
            signals,
            chunks,
            max_none_coverage_share=0.25,
            failure_mode="fail",
        )
    assert coverage_for_signal(report, "s1") == {
        "status": "partial",
        "legs": {
            "AAA": report["by_leg"][0],
            "BBB": report["by_leg"][1],
        },
        "none_legs": [],
    }


def test_ingestion_status_records_timestamp_bounds_per_ticker_and_source():
    status = build_ingestion_status(pd.DataFrame([
        {"ticker": "aaa", "source_type": "sec_8k", "available_at": "2023-01-01T00:00:00Z"},
        {"ticker": "AAA", "source_type": "sec_10k", "available_at": "2024-01-01T00:00:00Z"},
        {"ticker": "AAA", "source_type": "sec_8k", "available_at": "2024-05-01T00:00:00Z"},
        {"ticker": "AAA", "source_type": "earnings_transcript", "available_at": "2024-04-01T00:00:00Z"},
    ]))

    sec_8k = status[status["source"] == "sec"].iloc[0]
    transcript = status[status["source"] == "transcripts"].iloc[0]
    assert sec_8k["earliest_available_at"] == pd.Timestamp("2023-01-01T00:00:00Z")
    assert sec_8k["latest_available_at"] == pd.Timestamp("2024-05-01T00:00:00Z")
    assert transcript["chunk_count"] == 1


def test_ingestion_status_includes_universe_sources_with_no_chunks():
    status = build_ingestion_status(
        pd.DataFrame(columns=["ticker", "source_type", "available_at"]),
        ["AAA"],
    )

    assert status[["ticker", "source", "chunk_count"]].to_dict("records") == [
        {"ticker": "AAA", "source": "sec", "chunk_count": 0},
        {"ticker": "AAA", "source": "transcripts", "chunk_count": 0},
    ]


def test_coverage_gate_report_includes_derived_config():
    config = BacktestCoverageConfig(
        backtest_start=date(2024, 1, 1),
        backtest_end=date(2024, 12, 31),
        coverage_failure_mode="warn",
    )
    signal = {"signal_id": "s", "pair": ("AAA", "BBB"), "t_signal": "2024-06-01T00:00:00Z"}
    report = check_replay_coverage(
        [signal],
        pd.DataFrame(columns=["ticker", "source_type", "available_at"]),
        config,
    )

    assert report["config"]["max_retrieval_window_days"] == 550
    assert report["distribution"]["none"] == 2
    assert report["within_threshold"] is False
