"""
earnings_transcript_fetcher.py - acquire earnings call transcripts from Alpha Vantage
and dump them, untouched, into the data lake.

Responsibility: ACQUISITION ONLY (quota, caching, raw dump). Normalisation, point-in-
time dating and row building live in earnings_transcript_cleaner.py.

Why it is built around caching and a quota
------------------------------------------
Alpha Vantage's free key is small (about 25 requests/day, 5/minute; verify your own
key's limit), and one request returns one transcript (one ticker, one quarter). So:
  * only the MOST RECENT call per ticker is fetched by default - for a signal on pair
    (A, B) that is 2 requests;
  * every response is stored, so a transcript is never downloaded twice;
  * "no transcript yet" answers are stored too (re-checked after `empty_ttl_days`), so
    probing a quarter that has not been reported does not burn quota repeatedly;
  * a persistent daily counter stops the run before the quota is exceeded; tickers not
    reached are reported as `deferred` and picked up by the next run (cache makes
    re-runs cheap);
  * if Alpha Vantage answers with a quota / premium-plan notice, the run stops at once
    and that message is NOT cached as data.

Quarter labels
--------------
Alpha Vantage addresses a call as YYYYQn (e.g. 2026Q2). Candidates are generated from
CALENDAR quarters, newest first. If a company's fiscal quarters differ from calendar
quarters, the label of its latest call may differ - check one real response with the
printed summary, and pass `--quarters` explicitly if needed.

Data lake layout (lake_root defaults to data/lake)
--------------------------------------------------
    <lake_root>/raw/earnings-transcripts/
        manifest.jsonl                       one record per fetch (last record per key wins)
        quota.json                           requests used today (UTC)
        <TICKER>/<YYYYQn>.json               raw API response, byte-for-byte

Setup and usage
---------------
    Add ALPHAVANTAGE_API_KEY=your-free-key to .env.
    python -m src.rag.scraping.earnings_transcript_fetcher --tickers AAPL MSFT
    python -m src.rag.cleaning.earnings_transcript_cleaner --tickers AAPL MSFT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import requests

from src.rag.coverage import quarter_range

load_dotenv()
from src.rag.scraping.universe import ingestion_tickers, load_universe

logger = logging.getLogger(__name__)

AV_URL = "https://www.alphavantage.co/query"
RAW_DIRNAME = "earnings-transcripts"
MANIFEST_NAME = "manifest.jsonl"
QUOTA_NAME = "quota.json"


class AlphaVantageNotice(RuntimeError):
    """Alpha Vantage returned a quota / plan message instead of data."""


class QuotaExhausted(RuntimeError):
    """Local daily request budget used up."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FetcherConfig:
    lake_root: Path = Path("data/lake")
    api_key: Optional[str] = None                 # falls back to env ALPHAVANTAGE_API_KEY
    daily_limit: int = 25
    min_request_interval: float = 12.5            # seconds; stays under 5 requests/minute
    timeout: float = 30.0
    max_attempts: int = 2
    probe_quarters: int = 2                       # how many quarters to try per ticker
    min_days_after_quarter_end: int = 7           # a call cannot exist before this
    empty_ttl_days: int = 7                       # re-check "no transcript yet" after this long
    empty_finality_days: int = 365                # older empty quarters remain cached

    def __post_init__(self) -> None:
        if self.daily_limit < 1 or self.max_attempts < 1 or self.probe_quarters < 1:
            raise ValueError("daily_limit, max_attempts, and probe_quarters must be >= 1")
        if self.min_request_interval < 0 or self.timeout <= 0:
            raise ValueError("min_request_interval must be >= 0 and timeout must be > 0")
        if (
            self.min_days_after_quarter_end < 0
            or self.empty_ttl_days < 0
            or self.empty_finality_days < 0
        ):
            raise ValueError("quarter delay and empty cache TTL must be >= 0")


# ---------------------------------------------------------------------------
# Small helpers (also used by the cleaner)
# ---------------------------------------------------------------------------

def quarter_end(year: int, quarter: int) -> date:
    month = 3 * quarter
    return date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)


def candidate_quarters(as_of: date, n: int, min_days_after_end: int = 7) -> list[str]:
    """Calendar quarters newest-first whose call could already have happened by `as_of`."""
    year, quarter = as_of.year, (as_of.month - 1) // 3 + 1
    out: list[str] = []
    while len(out) < n:
        if quarter_end(year, quarter) + timedelta(days=min_days_after_end) <= as_of:
            out.append(f"{year}Q{quarter}")
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
    return out


def load_manifest(lake_root: Path) -> dict[tuple[str, str], dict]:
    """{(ticker, quarter): record}; later records overwrite earlier ones."""
    path = lake_root / "raw" / RAW_DIRNAME / MANIFEST_NAME
    manifest: dict[tuple[str, str], dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    manifest[(record["ticker"], record["quarter"])] = record
    return manifest


def describe_payload(payload: dict) -> str:
    """One-line structural summary, to verify the response format on a real key."""
    turns = payload.get("transcript") or []
    first = turns[0] if turns and isinstance(turns[0], dict) else {}
    return (f"top-level keys={sorted(payload.keys())} | turns={len(turns)} | "
            f"turn keys={sorted(first.keys())}")


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper().replace(".", "-")
    if not normalized or not re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)*", normalized):
        raise ValueError(f"invalid ticker: {ticker!r}")
    return normalized


def normalize_quarter(quarter: str) -> str:
    normalized = quarter.strip().upper()
    if not re.fullmatch(r"\d{4}Q[1-4]", normalized):
        raise ValueError(f"invalid quarter {quarter!r}; expected YYYYQ1 through YYYYQ4")
    return normalized


class _DailyQuota:
    """Persistent per-UTC-day request counter."""

    def __init__(self, path: Path, limit: int):
        self.path, self.limit = path, limit
        today = datetime.now(timezone.utc).date().isoformat()
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.state = state if state.get("date") == today else {"date": today, "used": 0}

    @property
    def remaining(self) -> int:
        return max(self.limit - self.state["used"], 0)

    def consume(self) -> None:
        if self.remaining <= 0:
            raise QuotaExhausted(f"daily limit of {self.limit} requests reached")
        self.state["used"] += 1
        self.path.write_text(json.dumps(self.state), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class TranscriptFetcher:
    def __init__(self, config: FetcherConfig = FetcherConfig()):
        self.cfg = config
        self._api_key = config.api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError('Set your key first: $env:ALPHAVANTAGE_API_KEY = "your-free-key"')

        self.raw_dir = config.lake_root / "raw" / RAW_DIRNAME
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.raw_dir / MANIFEST_NAME
        self._manifest = load_manifest(config.lake_root)
        self._quota = _DailyQuota(self.raw_dir / QUOTA_NAME, config.daily_limit)
        self._session = requests.Session()
        self._last_request = 0.0
        self._cik_by_ticker: dict[str, int] = {}

        self.deferred: list[str] = []          # tickers not reached (quota / notice)
        self.deferred_work: list[tuple[str, str]] = []
        self.notice: Optional[str] = None      # Alpha Vantage message that stopped the run
        self.history_depth_by_ticker: dict[str, dict] = {}

    # ----- public API ------------------------------------------------------

    def fetch(
        self,
        tickers: Optional[list[str]],
        as_of: date,
        quarters: Optional[list[str]] = None,
        *,
        start_date: date | None = None,
    ) -> list[dict]:
        """
        Fetch every quarter intersecting start_date..as_of when start_date is set.
        Successful and empty quarter responses are cached so quota deferrals resume
        without repeating completed requests.
        """
        self.deferred, self.deferred_work, self.notice = [], [], None
        self.history_depth_by_ticker = {}
        try:
            universe = load_universe(self.cfg.lake_root)
        except FileNotFoundError:
            if tickers is None:
                raise
            self._cik_by_ticker = {}
        else:
            self._cik_by_ticker = dict(zip(universe["ticker"], universe["cik"]))
        wanted = list(dict.fromkeys(
            normalize_ticker(t) for t in ingestion_tickers(self.cfg.lake_root, tickers)
        ))
        if quarters is not None:
            quarters = list(dict.fromkeys(normalize_quarter(q) for q in quarters))
        elif start_date is not None:
            if as_of < start_date:
                raise ValueError("as_of must be on or after start_date")
            quarters = quarter_range(start_date, as_of)
        if start_date is not None:
            return self._fetch_range(wanted, quarters or [], start_date, as_of)
        found: list[dict] = []

        for index, ticker in enumerate(wanted):
            requested_quarters = quarters or candidate_quarters(
                as_of, self.cfg.probe_quarters, self.cfg.min_days_after_quarter_end
            )
            quarter_index = 0
            try:
                if start_date is None and quarters is None:
                    found += self._latest(ticker, as_of)
                    requested_quarters = candidate_quarters(
                        as_of, self.cfg.probe_quarters, self.cfg.min_days_after_quarter_end
                    )
                else:
                    for quarter_index, quarter in enumerate(requested_quarters):
                        record = self._get_or_fetch(ticker, quarter)
                        if record and record["status"] == "ok":
                            found.append(record)
            except QuotaExhausted as exc:
                remainder = requested_quarters[quarter_index:]
                self.history_depth_by_ticker[ticker] = self._history_depth(
                    ticker, requested_quarters, start_date, as_of
                )
                self.deferred_work.extend((ticker, value) for value in remainder)
                self.deferred_work.extend(
                    (later_ticker, value)
                    for later_ticker in wanted[index + 1 :]
                    for value in (quarters or candidate_quarters(
                        as_of, self.cfg.probe_quarters, self.cfg.min_days_after_quarter_end
                    ))
                )
                self.deferred = list(dict.fromkeys(t for t, _ in self.deferred_work))
                for later_ticker in wanted[index + 1 :]:
                    later_quarters = quarters or candidate_quarters(
                        as_of, self.cfg.probe_quarters, self.cfg.min_days_after_quarter_end
                    )
                    self.history_depth_by_ticker[later_ticker] = self._history_depth(
                        later_ticker, later_quarters, start_date, as_of
                    )
                logger.warning(
                    "%s - deferring %d ticker-quarter request(s)",
                    exc, len(self.deferred_work),
                )
                break
            except AlphaVantageNotice as exc:
                self.notice = str(exc)
                logger.error("Alpha Vantage notice, stopping: %s", exc)
                self.history_depth_by_ticker[ticker] = self._history_depth(
                    ticker, requested_quarters, start_date, as_of
                )
                self.deferred_work.extend(
                    (ticker, value) for value in requested_quarters[quarter_index:]
                )
                self.deferred_work.extend(
                    (later_ticker, value)
                    for later_ticker in wanted[index + 1 :]
                    for value in (quarters or requested_quarters)
                )
                self.deferred = list(dict.fromkeys(t for t, _ in self.deferred_work))
                for later_ticker in wanted[index + 1 :]:
                    later_quarters = quarters or requested_quarters
                    self.history_depth_by_ticker[later_ticker] = self._history_depth(
                        later_ticker, later_quarters, start_date, as_of
                    )
                break
            self.history_depth_by_ticker[ticker] = self._history_depth(
                ticker, requested_quarters, start_date, as_of
            )
        return found

    def _fetch_range(
        self,
        tickers: list[str],
        quarters: list[str],
        start_date: date,
        as_of: date,
    ) -> list[dict]:
        """Resume the exact unfinished ticker-quarter queue before new work."""
        requested = [(ticker, quarter) for ticker in tickers for quarter in quarters]
        allowed = set(requested)
        queue_path = self.raw_dir / "deferred_work.json"
        pending: list[tuple[str, str]] = []
        if queue_path.is_file():
            saved = json.loads(queue_path.read_text(encoding="utf-8"))
            if not isinstance(saved, list):
                raise ValueError(f"invalid deferred-work queue: {queue_path}")
            pending = [
                (normalize_ticker(item["ticker"]), normalize_quarter(item["quarter"]))
                for item in saved
                if isinstance(item, dict)
                and isinstance(item.get("ticker"), str)
                and isinstance(item.get("quarter"), str)
            ]
        pending = [task for task in dict.fromkeys(pending) if task in allowed]
        pending_set = set(pending)
        tasks = pending + [task for task in requested if task not in pending_set]

        found: list[dict] = []
        self.deferred_work = []
        for index, (ticker, quarter) in enumerate(tasks):
            try:
                record = self._get_or_fetch(ticker, quarter)
            except QuotaExhausted as exc:
                self.deferred_work = tasks[index:]
                logger.warning(
                    "%s - deferring %d ticker-quarter request(s)",
                    exc,
                    len(self.deferred_work),
                )
                break
            except AlphaVantageNotice as exc:
                self.notice = str(exc)
                logger.error("Alpha Vantage notice, stopping: %s", exc)
                self.deferred_work = tasks[index:]
                break
            if record and record["status"] == "ok":
                found.append(record)

        self.deferred = list(dict.fromkeys(ticker for ticker, _ in self.deferred_work))
        queue_path.write_text(
            json.dumps([
                {"ticker": ticker, "quarter": quarter}
                for ticker, quarter in self.deferred_work
            ], indent=2),
            encoding="utf-8",
        )
        for ticker in tickers:
            self.history_depth_by_ticker[ticker] = self._history_depth(
                ticker, quarters, start_date, as_of
            )
        return found

    def _history_depth(
        self,
        ticker: str,
        requested_quarters: list[str],
        start_date: date | None,
        as_of: date,
    ) -> dict:
        records = [
            self._manifest[(ticker, quarter)]
            for quarter in requested_quarters
            if (ticker, quarter) in self._manifest
            and self._manifest[(ticker, quarter)].get("status") == "ok"
        ]
        completed = all(
            (ticker, quarter) in self._manifest
            and self._manifest[(ticker, quarter)].get("status") in {"ok", "empty"}
            for quarter in requested_quarters
        )
        labels = [record["quarter"] for record in records]
        return {
            "requested_start": start_date.isoformat() if start_date else None,
            "requested_end": as_of.isoformat(),
            "quarters_requested": len(requested_quarters),
            "quarters_completed": sum(
                (ticker, quarter) in self._manifest
                and self._manifest[(ticker, quarter)].get("status") in {"ok", "empty"}
                for quarter in requested_quarters
            ),
            "transcripts_found": len(labels),
            "earliest_transcript_quarter": min(labels) if labels else None,
            "latest_transcript_quarter": max(labels) if labels else None,
            "complete": completed,
        }

    @property
    def quota_remaining(self) -> int:
        return self._quota.remaining

    # ----- per-ticker logic ------------------------------------------------

    def _latest(self, ticker: str, as_of: date) -> list[dict]:
        for quarter in candidate_quarters(as_of, self.cfg.probe_quarters, self.cfg.min_days_after_quarter_end):
            record = self._get_or_fetch(ticker, quarter)
            if record and record["status"] == "ok":
                return [record]
        logger.info("%s: no transcript found in the probed quarters", ticker)
        return []

    def _get_or_fetch(self, ticker: str, quarter: str) -> Optional[dict]:
        cached = self._manifest.get((ticker, quarter))
        if cached and cached["status"] == "ok":
            return cached
        if cached and cached["status"] == "empty":
            year, quarter_number = int(quarter[:4]), int(quarter[-1])
            finalized = (
                quarter_end(year, quarter_number)
                + timedelta(days=self.cfg.empty_finality_days)
                < date.today()
            )
            if finalized or self._is_fresh(cached):
                return cached
        return self._fetch_one(ticker, quarter)

    def _is_fresh(self, record: dict) -> bool:
        fetched = datetime.fromisoformat(record["fetched_at_utc"])
        return datetime.now(timezone.utc) - fetched < timedelta(days=self.cfg.empty_ttl_days)

    # ----- HTTP + storage --------------------------------------------------

    def _fetch_one(self, ticker: str, quarter: str) -> dict:
        params = {"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": ticker,
                  "quarter": quarter, "apikey": self._api_key}
        response = None
        for attempt in range(1, self.cfg.max_attempts + 1):
            pause = self.cfg.min_request_interval - (time.monotonic() - self._last_request)
            if pause > 0:
                time.sleep(pause)
            self._quota.consume()
            self._last_request = time.monotonic()
            try:
                response = self._session.get(AV_URL, params=params, timeout=self.cfg.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                logger.warning("%s %s attempt %d failed: %s", ticker, quarter, attempt, type(exc).__name__)
                response = None
                time.sleep(5)
        if response is None:
            raise RuntimeError(f"could not fetch {ticker} {quarter}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(f"Alpha Vantage returned invalid JSON for {ticker} {quarter}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Alpha Vantage returned an invalid payload for {ticker} {quarter}")
        self._raise_on_notice(payload, ticker, quarter)

        turns = payload.get("transcript")
        if not isinstance(turns, list) or any(not isinstance(turn, dict) for turn in turns):
            raise ValueError(
                f"Alpha Vantage returned an invalid transcript structure for {ticker} {quarter}"
            )
        if any(not isinstance(turn.get("content"), str) for turn in turns):
            raise ValueError(
                f"Alpha Vantage returned a turn without text for {ticker} {quarter}"
            )
        status = "ok" if turns else "empty"
        relative = Path("raw") / RAW_DIRNAME / ticker / f"{quarter}.json"
        self._atomic_write(self.cfg.lake_root / relative, response.content)

        record = {
            "ticker": ticker,
            "cik": self._cik_by_ticker.get(ticker),
            "quarter": quarter,
            "status": status,
            "n_turns": len(turns),
            "source_url": f"{AV_URL}?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={quarter}",
            "raw_path": relative.as_posix(),
            "bytes": len(response.content),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self._manifest[(ticker, quarter)] = record
        logger.info("%s %s -> %s (%d turns); quota left today: %d",
                    ticker, quarter, status, len(turns), self._quota.remaining)
        return record

    @staticmethod
    def _raise_on_notice(payload: dict, ticker: str, quarter: str) -> None:
        if "Error Message" in payload:
            raise ValueError(f"Alpha Vantage rejected {ticker} {quarter}: {payload['Error Message']}")
        for key in ("Note", "Information"):
            if key in payload:                                    # quota / premium message: never cache as data
                raise AlphaVantageNotice(str(payload[key]))

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch earnings call transcripts into the data lake.")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="test override; default: all tickers in data/universe.parquet")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--quarters", nargs="+", default=None, help="explicit labels, e.g. 2026Q2 2026Q1")
    parser.add_argument("--probe-quarters", type=int, default=2)
    parser.add_argument("--daily-limit", type=int, default=25)
    parser.add_argument("--lake-root", default="data/lake")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    fetcher = TranscriptFetcher(FetcherConfig(lake_root=Path(args.lake_root), daily_limit=args.daily_limit,
                                              probe_quarters=args.probe_quarters))
    tickers = ingestion_tickers(Path(args.lake_root), args.tickers)
    records = fetcher.fetch(tickers, as_of, args.quarters)

    print(f"\n{len(records)} transcript(s) available; quota left today: {fetcher.quota_remaining}")
    for rec in records:
        print(f"  {rec['ticker']:<6} {rec['quarter']}  turns={rec['n_turns']}  {rec['raw_path']}")
    if records:                                                   # verify the real response format once
        raw = json.loads((fetcher.cfg.lake_root / records[0]["raw_path"]).read_text(encoding="utf-8"))
        print("format check:", describe_payload(raw))
    if fetcher.deferred:
        print(f"deferred (run again tomorrow / after quota resets): {', '.join(fetcher.deferred)}")
    if fetcher.notice:
        print(f"Alpha Vantage said: {fetcher.notice}")


if __name__ == "__main__":
    main()
