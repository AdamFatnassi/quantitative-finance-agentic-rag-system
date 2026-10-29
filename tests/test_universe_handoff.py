import json
import sys
from pathlib import Path
from datetime import date

import pandas as pd
import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.data.data_processor import CleanMarketData
from src.rag.scraping.universe import ingestion_tickers, load_universe, universe_path


def _clean_market_data() -> CleanMarketData:
    index = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"])
    prices = pd.DataFrame({
        "AAA": [10.0, 11.0, 12.0],
        "BBB": [None, 20.0, 21.0],
    }, index=index)
    return CleanMarketData(
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        prices=prices,
        volumes=prices.fillna(0),
        sectors=pd.DataFrame([
            {"Symbol": "AAA", "GICS Sector": "Tech", "CIK": "123"},
            {"Symbol": "BBB", "GICS Sector": "Finance", "CIK": "0000000456"},
        ]),
        log_transformed=False,
    )


def test_universe_frame_has_filtered_tickers_and_observed_date_ranges():
    universe = _clean_market_data().universe_frame()

    assert universe.columns.tolist() == [
        "ticker", "cik", "sector", "first_date", "last_date",
    ]
    assert universe.to_dict("records") == [
        {
            "ticker": "AAA",
            "cik": "0000000123",
            "sector": "Tech",
            "first_date": pd.Timestamp("2023-01-02"),
            "last_date": pd.Timestamp("2023-01-04"),
        },
        {
            "ticker": "BBB",
            "cik": "0000000456",
            "sector": "Finance",
            "first_date": pd.Timestamp("2023-01-03"),
            "last_date": pd.Timestamp("2023-01-04"),
        },
    ]


def test_ingestion_defaults_to_all_universe_tickers_and_explicit_test_override(tmp_path):
    lake_root = tmp_path / "lake"
    universe_file = universe_path(lake_root)
    universe_file.parent.mkdir(parents=True, exist_ok=True)
    _clean_market_data().universe_frame().to_parquet(universe_file, index=False)

    assert load_universe(lake_root)["cik"].tolist() == ["0000000123", "0000000456"]
    assert ingestion_tickers(lake_root) == ["AAA", "BBB"]
    assert ingestion_tickers(lake_root, ["bbb", "bbb"]) == ["BBB"]
    assert ingestion_tickers(tmp_path / "missing", ["XYZ"]) == ["XYZ"]


def test_universe_reader_fails_loudly_for_missing_or_invalid_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="universe.parquet"):
        load_universe(tmp_path / "lake")

    universe_file = universe_path(tmp_path / "lake")
    universe_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "ticker": "AAA",
        "cik": "not-a-cik",
        "sector": "Tech",
        "first_date": "2023-01-02",
        "last_date": "2023-01-04",
    }]).to_parquet(universe_file, index=False)
    with pytest.raises(ValueError, match="invalid CIK"):
        load_universe(tmp_path / "lake")


def test_text_pipeline_requests_full_universe_without_pair_inputs(tmp_path, monkeypatch):
    from src.scripts import run_text_pipeline as pipeline

    lake_root = tmp_path / "lake"
    universe_file = universe_path(lake_root)
    universe_file.parent.mkdir(parents=True, exist_ok=True)
    _clean_market_data().universe_frame().to_parquet(universe_file, index=False)
    requested: list[str] = []

    class FakeScraper:
        def __init__(self, config):
            pass

        def scrape(self, tickers, as_of, forms=None, **kwargs):
            requested.extend(tickers)
            assert kwargs["start_date"] < kwargs["end_date"]
            return []

    class FakeCleaner:
        def __init__(self, config):
            pass

        def run(self, tickers, **kwargs):
            assert tickers == ["AAA", "BBB"]
            assert kwargs["start_date"] < kwargs["end_date"]
            return pd.DataFrame(columns=["ticker"])

    class FakeChunker:
        def __init__(self, config):
            self.out_path = lake_root / "processed" / "8k-sec" / "chunks.parquet"

        def chunk_frame(self, docs):
            return pd.DataFrame({"chunk_id": []})

    monkeypatch.setattr(pipeline, "SecFilingScraper", FakeScraper)
    monkeypatch.setattr(pipeline, "SecFilingCleaner", FakeCleaner)
    monkeypatch.setattr(pipeline, "SecFilingChunker", FakeChunker)

    counts = pipeline.run_text_pipeline(
        lake_root=lake_root,
        sources=("sec",),
    )

    assert requested == ["AAA", "BBB"]
    assert counts["sec_tickers_requested"] == 2
    assert counts["sec_tickers_with_filings"] == 0
    config = json.loads(
        (lake_root / "processed" / "backtest_ingestion_config.json").read_text()
    )
    assert config["backtest_start"] == "2023-01-02"
    assert config["max_retrieval_window_days"] == 550
    assert config["fetch_start"] == "2021-07-01"


def test_text_pipeline_backfills_transcripts_and_writes_provider_depth(
    tmp_path, monkeypatch
):
    from src.scripts import run_text_pipeline as pipeline

    lake_root = tmp_path / "lake"
    universe_file = universe_path(lake_root)
    universe_file.parent.mkdir(parents=True, exist_ok=True)
    _clean_market_data().universe_frame().to_parquet(universe_file, index=False)
    observed = {}

    class FakeFetcher:
        def __init__(self, config):
            observed["daily_limit"] = config.daily_limit
            self.history_depth_by_ticker = {
                "AAA": {
                    "requested_start": "2021-07-01",
                    "requested_end": date.today().isoformat(),
                    "quarters_requested": 23,
                    "quarters_completed": 23,
                    "transcripts_found": 8,
                    "earliest_transcript_quarter": "2021Q3",
                    "latest_transcript_quarter": "2024Q4",
                    "complete": True,
                }
            }
            self.deferred_work = []
            self.deferred = []
            self.notice = None

        def fetch(self, tickers, as_of, quarters, *, start_date):
            observed["tickers"] = tickers
            observed["start_date"] = start_date
            observed["quarters"] = quarters
            return []

    class FakeCleaner:
        def __init__(self, config):
            pass

        def run(self, tickers):
            return pd.DataFrame(columns=["ticker"])

    class FakeChunker:
        def __init__(self, config):
            self.out_path = lake_root / "processed" / "earnings-transcripts" / "chunks.parquet"

        def chunk_frame(self, docs):
            return pd.DataFrame({"chunk_id": []})

    monkeypatch.setattr(pipeline, "TranscriptFetcher", FakeFetcher)
    monkeypatch.setattr(pipeline, "EarningsTranscriptCleaner", FakeCleaner)
    monkeypatch.setattr(pipeline, "EarningsTranscriptChunker", FakeChunker)

    counts = pipeline.run_text_pipeline(
        lake_root=lake_root,
        sources=("transcripts",),
        backtest_start=date(2023, 1, 2),
        backtest_end=date(2024, 12, 31),
        transcripts_per_leg=3,
    )

    assert observed == {
        "daily_limit": 25,
        "tickers": ["AAA", "BBB"],
        "start_date": date(2021, 7, 1),
        "quarters": None,
    }
    assert counts["transcript_quarter_requests_deferred"] == 0
    history = json.loads(
        (lake_root / "processed" / "transcript_history_depth.json").read_text()
    )
    assert history["AAA"]["earliest_transcript_quarter"] == "2021Q3"


def test_pair_candidate_writer_keeps_window_model_and_validity_separate():
    from src.data.market_data_io import write_pair_candidates

    class Connection:
        def execute(self, statement, records):
            self.statement = str(statement)
            self.records = records

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

    candidates = pd.DataFrame([{
        "window_id": 4,
        "t0": pd.Timestamp("2024-01-31"),
        "stock_a": "AAA",
        "stock_b": "BBB",
        "sector": "Tech",
        "ssd": 0.25,
        "ssd_rank": 1,
        "is_valid": True,
    }])
    engine = Engine()

    count = write_pair_candidates(
        engine,
        candidates,
        {"AAA": 11, "BBB": 22},
        "stat-arb-test-version",
    )

    assert count == 1
    assert "INSERT INTO pair_candidates" in engine.connection.statement
    assert "INSERT INTO pairs" not in engine.connection.statement
    assert engine.connection.records[0]["window_id"] == 4
    assert engine.connection.records[0]["model_version"] == "stat-arb-test-version"
    assert engine.connection.records[0]["is_valid"] is True
