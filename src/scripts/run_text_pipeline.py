"""Fetch, clean, and chunk SEC filings and earnings transcripts."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from src.rag.chunking.earnings_transcript_chunker import EarningsTranscriptChunker
from src.rag.chunking.sec_chunker import ChunkerConfig, SecFilingChunker
from src.rag.cleaning.earnings_transcript_cleaner import (
    CleanerConfig as TranscriptCleanerConfig,
    EarningsTranscriptCleaner,
)
from src.rag.cleaning.sec_cleaner import (
    SUPPORTED_FORMS,
    CleanerConfig as SecCleanerConfig,
    SecFilingCleaner,
)
from src.rag.scraping.earnings_transcript_fetcher import (
    FetcherConfig,
    TranscriptFetcher,
)
from src.rag.scraping.sec_scraper import ScraperConfig, SecFilingScraper
from src.rag.scraping.universe import ingestion_tickers
from src.rag.coverage import BacktestCoverageConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run_text_pipeline(
    tickers: list[str] | None = None,
    lake_root: Path = Path("data/lake"),
    sources: tuple[str, ...] = ("sec", "transcripts"),
    as_of: date | None = None,
    backtest_start: date | None = None,
    backtest_end: date | None = None,
    forms: tuple[str, ...] = SUPPORTED_FORMS,
    quarters: list[str] | None = None,
    transcripts_per_leg: int = 2,
    recency_8k_days: int = 90,
    recency_10q_days: int = 180,
    latest_10k_days: int = 550,
    daily_limit: int = 25,
    max_tokens: int = 350,
    overlap_tokens: int = 50,
    min_chunk_tokens: int = 60,
) -> dict[str, int]:
    """Run requested text-source pipelines and return per-stage row counts."""
    requested_tickers = ingestion_tickers(lake_root, tickers)
    if backtest_start is None:
        universe_file = lake_root.parent / "universe.parquet"
        if not universe_file.is_file():
            raise FileNotFoundError(
                f"{universe_file} is required; pass backtest_start or run the quantitative pipeline."
            )
        universe_frame = pd.read_parquet(universe_file)
        backtest_start = pd.to_datetime(universe_frame["first_date"]).min().date()
    backtest_end = backtest_end or as_of or date.today()
    config = BacktestCoverageConfig(
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        recency_8k_days=recency_8k_days,
        recency_10q_days=recency_10q_days,
        latest_10k_days=latest_10k_days,
        transcripts_per_leg=transcripts_per_leg,
    )
    run_date = date.today()
    selected_sources = tuple(dict.fromkeys(sources))
    invalid_sources = set(selected_sources) - {"sec", "transcripts"}
    if not selected_sources or invalid_sources:
        raise ValueError(f"unsupported text sources: {sorted(invalid_sources)}")
    chunk_config = ChunkerConfig(
        lake_root=lake_root,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
    )
    counts: dict[str, int] = {}
    config_path = lake_root / "processed" / "backtest_ingestion_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")

    if "sec" in selected_sources:
        scraper = SecFilingScraper(ScraperConfig(lake_root=lake_root))
        filings = scraper.scrape(
            requested_tickers,
            run_date,
            forms=list(forms),
            start_date=config.fetch_start,
            end_date=run_date,
        )
        sec_docs = SecFilingCleaner(
            SecCleanerConfig(lake_root=lake_root, forms=forms)
        ).run(
            requested_tickers,
            start_date=config.fetch_start,
            end_date=run_date,
        )
        sec_chunker = SecFilingChunker(chunk_config)
        sec_chunks = sec_chunker.chunk_frame(sec_docs)
        sec_chunker.out_path.parent.mkdir(parents=True, exist_ok=True)
        sec_chunks.to_parquet(sec_chunker.out_path, index=False)
        counts.update({
            "sec_filings_fetched": len(filings),
            "sec_tickers_requested": len(requested_tickers),
            "sec_tickers_with_filings": (
                int(sec_docs["ticker"].nunique()) if not sec_docs.empty else 0
            ),
            "sec_documents_cleaned": len(sec_docs),
            "sec_chunks": len(sec_chunks),
        })

    if "transcripts" in selected_sources:
        fetcher = TranscriptFetcher(FetcherConfig(
            lake_root=lake_root,
            daily_limit=daily_limit,
        ))
        transcripts = fetcher.fetch(
            requested_tickers,
            run_date,
            quarters,
            start_date=None if quarters is not None else config.fetch_start,
        )
        history_path = lake_root / "processed" / "transcript_history_depth.json"
        history_path.write_text(
            json.dumps(fetcher.history_depth_by_ticker, indent=2),
            encoding="utf-8",
        )
        deferred_path = lake_root / "processed" / "transcript_deferred_work.json"
        deferred_path.write_text(
            json.dumps([
                {"ticker": ticker, "quarter": quarter}
                for ticker, quarter in fetcher.deferred_work
            ], indent=2),
            encoding="utf-8",
        )
        transcript_docs = EarningsTranscriptCleaner(
            TranscriptCleanerConfig(lake_root=lake_root)
        ).run(tickers=requested_tickers)
        transcript_chunker = EarningsTranscriptChunker(chunk_config)
        transcript_chunks = transcript_chunker.chunk_frame(transcript_docs)
        transcript_chunker.out_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_chunks.to_parquet(transcript_chunker.out_path, index=False)
        counts.update({
            "transcripts_fetched": len(transcripts),
            "transcript_tickers_requested": len(requested_tickers),
            "transcript_tickers_with_turns": (
                int(transcript_docs["ticker"].nunique()) if not transcript_docs.empty else 0
            ),
            "transcript_tickers_deferred": len(fetcher.deferred),
            "transcript_quarter_requests_deferred": len(fetcher.deferred_work),
            "transcript_turns_cleaned": len(transcript_docs),
            "transcript_chunks": len(transcript_chunks),
        })
        if fetcher.notice:
            logging.warning("Transcript fetching stopped by provider notice: %s", fetcher.notice)
        if fetcher.deferred:
            logging.warning("Transcript fetch deferred tickers: %s", ", ".join(fetcher.deferred))

    return counts


def main() -> None:
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(
        description="Fetch, clean, and chunk SEC filings and earnings transcripts."
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="test override; default: all tickers in data/universe.parquet")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("sec", "transcripts"),
        default=("sec", "transcripts"),
        help="text sources to process (default: both)",
    )
    parser.add_argument("--backtest-start", type=date.fromisoformat, required=True)
    parser.add_argument("--backtest-end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--forms",
        nargs="+",
        choices=SUPPORTED_FORMS,
        default=SUPPORTED_FORMS,
        help="SEC filing forms (default: all supported forms)",
    )
    parser.add_argument("--quarters", nargs="+", default=None, help="transcript quarters, e.g. 2026Q2")
    parser.add_argument("--transcripts-per-leg", type=int, default=2)
    parser.add_argument("--recency-8k-days", type=int, default=90)
    parser.add_argument("--recency-10q-days", type=int, default=180)
    parser.add_argument("--latest-10k-days", type=int, default=550)
    parser.add_argument("--daily-limit", type=int, default=25)
    parser.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--min-chunk-tokens", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts = run_text_pipeline(
        tickers=args.tickers,
        lake_root=args.lake_root,
        sources=tuple(args.sources),
        backtest_start=args.backtest_start,
        backtest_end=args.backtest_end,
        forms=tuple(args.forms),
        quarters=args.quarters,
        transcripts_per_leg=args.transcripts_per_leg,
        recency_8k_days=args.recency_8k_days,
        recency_10q_days=args.recency_10q_days,
        latest_10k_days=args.latest_10k_days,
        daily_limit=args.daily_limit,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
    )
    print("\nText pipeline complete")
    for stage, count in counts.items():
        print(f"  {stage}: {count}")


if __name__ == "__main__":
    main()
