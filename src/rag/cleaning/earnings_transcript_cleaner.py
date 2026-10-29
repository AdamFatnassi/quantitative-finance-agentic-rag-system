"""Normalize cached Alpha Vantage earnings transcripts into speaker-turn records."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.rag.chunking.meta_data_schema import TranscriptMetadata
from src.rag.scraping.earnings_transcript_fetcher import (
    MANIFEST_NAME,
    RAW_DIRNAME,
    normalize_quarter,
    normalize_ticker,
)
from src.rag.scraping.universe import load_universe
from src.rag.cleaning.transcript_availability import (
    next_trading_open_after_call,
    quarter_fallback_open,
)

logger = logging.getLogger(__name__)

_ANALYST_TITLE = re.compile(r"\banalyst\b", re.IGNORECASE)
_ROLE_RULES = (
    (re.compile(r"\bchief executive officer\b|\bCEO\b", re.IGNORECASE), "CEO"),
    (re.compile(r"\bchief financial officer\b|\bCFO\b", re.IGNORECASE), "CFO"),
    (re.compile(r"\boperator\b", re.IGNORECASE), "operator"),
    (_ANALYST_TITLE, "analyst"),
    (re.compile(r"\binvestor relations\b", re.IGNORECASE), "investor_relations"),
)


@dataclass(frozen=True)
class CleanerConfig:
    lake_root: Path = Path("data/lake")
    min_chars: int = 1

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be >= 1")


def fiscal_quarter_label(value: str) -> str:
    """Convert provider label YYYYQn to the stable metadata label YYYY-Qn."""
    quarter = normalize_quarter(value)
    return f"{quarter[:4]}-{quarter[4:]}"


def clean_text(value: str) -> str:
    """Normalize provider text while retaining paragraph boundaries."""
    text = unicodedata.normalize("NFKC", value).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_speaker_role(title: object) -> Optional[str]:
    if not isinstance(title, str) or not title.strip():
        return None
    normalized = title.strip()
    for pattern, role in _ROLE_RULES:
        if pattern.search(normalized):
            return role
    return normalized


def _timestamp(value: object) -> Optional[str]:
    """Accept only explicit timezone-aware timestamps; dates and naive values are unknown."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def clean_transcript(record: dict, payload: dict, config: CleanerConfig) -> list[dict]:
    """Create one normalized row per speaker turn from a validated raw response."""
    ticker = normalize_ticker(str(record["ticker"]))
    quarter_value = normalize_quarter(str(record["quarter"]))
    payload_quarter = normalize_quarter(str(payload.get("quarter") or quarter_value))
    if payload_quarter != quarter_value:
        raise ValueError(f"raw transcript quarter does not match manifest for {ticker}")
    quarter = fiscal_quarter_label(quarter_value)
    response_ticker = payload.get("symbol")
    if response_ticker and normalize_ticker(str(response_ticker)) != ticker:
        raise ValueError(f"raw transcript symbol does not match manifest ticker {ticker}")

    turns = payload.get("transcript")
    if not isinstance(turns, list):
        raise ValueError(f"transcript is not a list for {ticker} {quarter}")

    first_analyst = next(
        (
            index for index, turn in enumerate(turns)
            if isinstance(turn, dict) and _ANALYST_TITLE.search(str(turn.get("title", "")))
        ),
        None,
    )
    published_at = _timestamp(payload.get("published_at") or record.get("published_at"))
    retrieved_at = _timestamp(record.get("fetched_at_utc"))
    call_datetime = payload.get("call_datetime") or record.get("call_datetime")
    trusted_call_date = (
        payload.get("trusted_call_datetime")
        or payload.get("trusted_call_date")
        or record.get("trusted_call_datetime")
        or record.get("trusted_call_date")
    )
    if published_at:
        available_at = published_at
        availability_basis = "provider_published_at"
        availability_estimated = False
    elif call_datetime:
        call_open = next_trading_open_after_call(call_datetime)
        if call_open is None:
            raise ValueError(f"invalid call datetime for {ticker} {quarter}")
        available_at = call_open.isoformat()
        availability_basis = "call_datetime_lag"
        availability_estimated = True
    elif trusted_call_date:
        call_open = next_trading_open_after_call(trusted_call_date)
        if call_open is None:
            raise ValueError(f"invalid trusted call date for {ticker} {quarter}")
        available_at = call_open.isoformat()
        availability_basis = "trusted_call_date_lag"
        availability_estimated = True
    else:
        year, quarter_number = int(quarter_value[:4]), int(quarter_value[-1])
        available_at = quarter_fallback_open(year, quarter_number).isoformat()
        availability_basis = "quarter_end_plus_90_days"
        availability_estimated = True

    cik_value = record.get("cik")
    if cik_value is None or not str(cik_value).strip().isdigit():
        raise ValueError(f"no SEC CIK is available for transcript {ticker} {quarter}")
    cik = int(cik_value)
    source_url = record.get("source_url")
    raw_path = record.get("raw_path")
    document_id = f"{ticker}:{quarter}"
    transcript_id = document_id
    output: list[dict] = []

    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"invalid turn {index} for {ticker} {quarter}")
        content = turn.get("content")
        if not isinstance(content, str):
            raise ValueError(f"turn {index} has no text for {ticker} {quarter}")
        text = clean_text(content)
        if len(text) < config.min_chars:
            continue

        title = turn.get("title")
        speaker_name = turn.get("speaker")
        speaker_name = speaker_name.strip() if isinstance(speaker_name, str) else None
        role = normalize_speaker_role(title)
        section = (
            "prepared_remarks" if first_analyst is not None and index < first_analyst
            else "q_and_a" if first_analyst is not None
            else None
        )
        sentiment = turn.get("sentiment")
        sentiment = sentiment.strip() if isinstance(sentiment, str) and sentiment.strip() else None
        metadata = TranscriptMetadata.model_validate({
            "source_type": "earnings_transcript",
            "ticker": ticker,
            "cik": cik,
            "transcript_id": transcript_id,
            "accession_number": None,
            "item_code": None,
            "related_tickers": [],
            "company_name": None,
            "published_at": published_at,
            "retrieved_at": retrieved_at,
            "available_at": available_at,
            "availability_estimated": availability_estimated,
            "availability_basis": availability_basis,
            "section_type": section,
            "form_type": None,
            "source_url": source_url,
            "raw_path": raw_path,
            "fiscal_quarter": quarter,
            "section": section,
            "speaker_name": speaker_name,
            "speaker_role": role,
            "turn_index": index,
            "sentiment": sentiment,
        }).model_dump(mode="json")
        turn_id = f"{document_id}:turn-{index:04d}"
        output.append({
            "document_id": document_id,
            "turn_id": turn_id,
            "turn_index": index,
            "ticker": ticker,
            "cik": cik,
            "transcript_id": transcript_id,
            "accession_number": None,
            "fiscal_quarter": quarter,
            "item_code": None,
            "section": section,
            "section_type": section,
            "form_type": None,
            "speaker_name": speaker_name,
            "speaker_role": role,
            "sentiment": sentiment,
            "published_at": published_at,
            "retrieved_at": retrieved_at,
            "available_at": available_at,
            "availability_estimated": availability_estimated,
            "availability_basis": availability_basis,
            "source_url": source_url,
            "raw_path": raw_path,
            "text": text,
            "word_count": len(text.split()),
            "n_chars": len(text),
            "metadata": metadata,
        })
    return output


class EarningsTranscriptCleaner:
    def __init__(self, config: CleanerConfig = CleanerConfig()):
        self.cfg = config
        self.raw_dir = config.lake_root / "raw" / RAW_DIRNAME
        self.manifest_path = self.raw_dir / MANIFEST_NAME
        self.out_path = (
            config.lake_root / "processed" / RAW_DIRNAME / "transcripts_clean.parquet"
        )

    def run(
        self,
        tickers: Optional[list[str]] = None,
        write: bool = True,
    ) -> pd.DataFrame:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.manifest_path} not found; run earnings_transcript_fetcher first"
            )

        wanted = {normalize_ticker(ticker) for ticker in tickers} if tickers else None
        records: dict[tuple[str, str], dict] = {}
        with self.manifest_path.open(encoding="utf-8") as manifest_file:
            for line in manifest_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                ticker = normalize_ticker(str(record["ticker"]))
                quarter = normalize_quarter(str(record["quarter"]))
                records[(ticker, quarter)] = record

        missing_cik_tickers = {
            ticker for (ticker, _), record in records.items()
            if record.get("status") == "ok" and not record.get("cik")
        }
        if missing_cik_tickers:
            universe = load_universe(self.cfg.lake_root)
            cik_by_ticker = dict(zip(universe["ticker"], universe["cik"]))
            absent = sorted(missing_cik_tickers - cik_by_ticker.keys())
            if absent:
                raise ValueError(
                    "no CIK mapping in universe.parquet for cached transcript ticker(s): "
                    + ", ".join(absent)
                )
            for (ticker, _), record in records.items():
                if ticker in missing_cik_tickers:
                    record["cik"] = cik_by_ticker[ticker]

        rows: list[dict] = []
        for (ticker, quarter), record in sorted(records.items()):
            if record.get("status") != "ok" or (wanted is not None and ticker not in wanted):
                continue
            relative_path = Path(record["raw_path"])
            raw_path = relative_path if relative_path.is_absolute() else self.cfg.lake_root / relative_path
            if not raw_path.is_file():
                logger.warning("raw transcript missing; skipped: %s", raw_path)
                continue
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            rows.extend(clean_transcript(record, payload, self.cfg))

        columns = [
            "document_id", "turn_id", "turn_index", "ticker", "cik", "transcript_id",
            "accession_number", "item_code", "fiscal_quarter", "section",
            "section_type", "form_type",
            "speaker_name", "speaker_role", "sentiment", "published_at",
            "retrieved_at", "available_at", "availability_estimated",
            "availability_basis", "source_url",
            "raw_path", "text", "word_count", "n_chars", "metadata",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if not frame.empty:
            frame = frame.sort_values(
                ["ticker", "fiscal_quarter", "turn_index"], kind="stable"
            ).reset_index(drop=True)
        if write:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self.out_path, index=False)
            logger.info("wrote %d transcript turns -> %s", len(frame), self.out_path)
        return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean cached Alpha Vantage earnings transcripts."
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--lake-root", default="data/lake")
    parser.add_argument("--min-chars", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cleaner = EarningsTranscriptCleaner(
        CleanerConfig(lake_root=Path(args.lake_root), min_chars=args.min_chars)
    )
    turns = cleaner.run(tickers=args.tickers)
    print(f"{len(turns)} speaker turns from {turns['document_id'].nunique() if not turns.empty else 0} transcripts")
    if not turns.empty:
        print(turns.groupby(["ticker", "fiscal_quarter", "section"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
