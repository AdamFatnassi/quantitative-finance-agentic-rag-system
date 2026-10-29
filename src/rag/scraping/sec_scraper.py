"""
Download raw SEC filings for a set of tickers into the data lake.

EdgarTools handles ticker resolution, filing discovery, SEC identity and request
throttling. This module stores the complete submission text and a manifest; it does
not select sections, clean, chunk, or embed filings.

By default this keeps the original 8-K behavior. Pass `forms` to ScraperConfig or
--forms to download 10-K / 10-Q filings and amendments as well. Complete submission
text is stored so 8-K exhibits remain available to sec_8k_cleaner.py.

Data lake layout (lake_root defaults to data/lake)
--------------------------------------------------
    <lake_root>/raw/8k-sec/
        manifest.jsonl                        one JSON record per (ticker, filing)
        filings/<cik10>/<accession>.txt       complete raw submission text

Set SEC_USER_AGENT in the repository's .env file to your name and contact email:
    SEC_USER_AGENT=Your Name your.email@example.com

Usage:
    python -m src.rag.scraping.sec_scraper --tickers AAPL MSFT
    python -m src.rag.scraping.sec_scraper --tickers AAPL --forms 10-K 10-K/A 10-Q 10-Q/A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from edgar import Company, set_identity
from edgar.entity.filings import EntityFiling

from src.rag.scraping.universe import ingestion_tickers

logger = logging.getLogger(__name__)

# Load the repository-level .env without overriding explicitly configured environment.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

RAW_DIRNAME = "8k-sec"
MANIFEST_NAME = "manifest.jsonl"
DEFAULT_FORMS = ("8-K", "8-K/A")
SUPPORTED_FORMS = ("8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A")


EASTERN_TZ = ZoneInfo("America/New_York")


def normalize_acceptance_datetime(acceptance_datetime: datetime | str | None) -> str:
    """Normalize SEC acceptance datetime to UTC ISO string.

    If naive, EDGAR convention represents Eastern Time (America/New_York),
    which is localized and converted to UTC handling EST/EDT daylight saving boundaries.
    """
    if not acceptance_datetime:
        return ""
    if isinstance(acceptance_datetime, datetime):
        if acceptance_datetime.tzinfo is None:
            acceptance_datetime = acceptance_datetime.replace(tzinfo=EASTERN_TZ)
        return acceptance_datetime.astimezone(timezone.utc).isoformat()
    timestamp = pd.Timestamp(acceptance_datetime)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(EASTERN_TZ)
    return timestamp.tz_convert(timezone.utc).isoformat()


@dataclass(frozen=True)
class ScraperConfig:
    lake_root: Path = Path("data/lake")
    user_agent: Optional[str] = None
    forms: tuple[str, ...] = DEFAULT_FORMS


class SecFilingScraper:
    """Acquire complete SEC submissions using EdgarTools."""

    def __init__(self, config: ScraperConfig = ScraperConfig()):
        self.cfg = config
        user_agent = config.user_agent or os.environ.get("SEC_USER_AGENT", "")
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires an identifying email in its request identity. Set it first, e.g.\n"
                "  SEC_USER_AGENT=Your Name your.email@example.com in the repository .env file"
            )
        set_identity(user_agent)

        self.raw_dir = config.lake_root / "raw" / RAW_DIRNAME
        self.manifest_path = self.raw_dir / MANIFEST_NAME
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self._load_manifest()

    def scrape(
        self,
        tickers: Optional[list[str]],
        as_of: date,
        lookback_days: int = 90,
        forms: Optional[list[str]] = None,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Download requested forms filed in the inclusive filing-date window."""
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        if start_date is not None and end_date is not None:
            if end_date < start_date:
                raise ValueError("end_date must be on or after start_date")
            start, end = start_date, end_date
        else:
            if lookback_days < 1:
                raise ValueError("lookback_days must be >= 1")
            start, end = as_of - timedelta(days=lookback_days), as_of
        requested_forms = self.cfg.forms if forms is None else forms
        selected_forms = tuple(dict.fromkeys(
            form.strip().upper() for form in requested_forms
        ))
        invalid_forms = sorted(set(selected_forms) - set(SUPPORTED_FORMS))
        if not selected_forms or invalid_forms:
            raise ValueError(
                f"forms must be selected from {', '.join(SUPPORTED_FORMS)}"
                + (f"; unsupported: {', '.join(invalid_forms)}" if invalid_forms else "")
            )

        wanted = list(dict.fromkeys(
            self._normalize_ticker(t) for t in ingestion_tickers(self.cfg.lake_root, tickers)
        ))
        records: list[dict] = []

        for ticker in wanted:
            company = Company(ticker)
            filings = company.get_filings(
                form=list(selected_forms),
                filing_date=f"{start.isoformat()}:{end.isoformat()}",
                amendments=True,
            )
            logger.info(
                "%s: %d %s filings in %s .. %s",
                ticker, len(filings), ", ".join(selected_forms), start, end,
            )
            for filing in filings:
                records.append(self._store_filing(ticker, filing))
        return records

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        return ticker.strip().upper().replace(".", "-")

    def _store_filing(self, ticker: str, filing: EntityFiling) -> dict:
        accession = filing.accession_no
        key = (ticker, accession)
        cached = self._manifest.get(key)
        if cached and (self.cfg.lake_root / cached["raw_path"]).exists():
            return cached

        relative = (
            Path("raw") / RAW_DIRNAME / "filings" / f"{filing.cik:010d}" / f"{accession}.txt"
        )
        target = self.cfg.lake_root / relative

        if target.exists():
            payload = target.read_bytes()
        else:
            submission = filing.full_text_submission()
            payload = submission.encode("utf-8")
            self._atomic_write(target, payload)

        primary_document = filing.primary_document or ""
        source_url = (
            f"{filing.base_dir}/{primary_document}" if primary_document else filing.url
        )
        acceptance_datetime = normalize_acceptance_datetime(filing.acceptance_datetime)

        record = {
            "ticker": ticker,
            "cik": int(filing.cik),
            "company_name": filing.company,
            "accession_number": accession,
            "form": filing.form,
            "items": filing.items or "",
            "filing_date": str(filing.filing_date),
            "report_date": filing.report_date or "",
            "acceptance_datetime": str(acceptance_datetime),
            "primary_document": primary_document,
            "source_url": source_url,
            "raw_path": relative.as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self._manifest[key] = record
        logger.info("stored %s %s %s [%s]", ticker, accession, filing.form, record["items"])
        return record

    def _load_manifest(self) -> dict[tuple[str, str], dict]:
        if not self.manifest_path.exists():
            return {}
        manifest = {}
        with self.manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    manifest[(record["ticker"], record["accession_number"])] = record
        return manifest

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)


# Keep the original import name working for existing 8-K-only callers.
Sec8KScraper = SecFilingScraper


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC filings into the data lake.")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="test override; default: all tickers in data/universe.parquet")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument(
        "--forms",
        nargs="+",
        choices=SUPPORTED_FORMS,
        default=None,
        help="SEC forms to download (default: 8-K 8-K/A)",
    )
    parser.add_argument("--lake-root", default="data/lake")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    scraper = SecFilingScraper(ScraperConfig(lake_root=Path(args.lake_root)))
    tickers = ingestion_tickers(Path(args.lake_root), args.tickers)
    records = scraper.scrape(tickers, as_of, args.lookback_days, args.forms)

    print(f"\n{len(records)} filings in window (as of {as_of}, lookback {args.lookback_days}d)")
    for record in sorted(records, key=lambda row: (row["ticker"], row["filing_date"])):
        print(
            f"  {record['ticker']:<6} {record['filing_date']}  {record['form']:<6} "
            f"items=[{record['items']}]  {record['raw_path']}"
        )


if __name__ == "__main__":
    main()
