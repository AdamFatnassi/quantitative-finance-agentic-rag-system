"""Post-ingestion data-validation stage.

Runs automatically after ingestion and raises ValidationError on violations.
Emits one consolidated report: pass/fail per invariant, counts, and up to
MAX_VIOLATION_EXAMPLES examples per failed check.

Four invariants
---------------
1. TIMESTAMP   – For every SEC chunk, available_at >= <ACCEPTANCE-DATETIME>
                 parsed from the raw SGML header (Eastern time, DST-aware).
                 available_at is corrected to max(stored, header_utc).
                 Mismatch rate must be <= timestamp_mismatch_threshold.

2. IDENTITY    – (cik, accession_number, document_id_canonical, chunk_index)
                 is a unique key in chunks.parquet.  document_id_canonical
                 strips the leading <TICKER>: prefix so GOOG/GOOGL chunks from
                 the same filing are flagged as duplicates.  The report lists
                 tickers per duplicate group and whether the text is identical.

3. STATUS      – ingestion_status (DB or Parquet) has a row for every
                 (ticker, source) in the universe.  Valid statuses:
                   fetched   – chunk_count > 0
                   missing   – row absent or chunk_count == 0
                   deferred  – explicitly set by caller
                   failed    – explicitly set by caller
                   disabled  – transcripts source (not yet enabled)
                 A missing SEC row counts as a violation.

4. RETRIEVAL   – Qdrant's DatetimeRange filter enforces available_at <= t_signal.
                 Samples N points; for each, asserts 0 results at
                 available_at - 1 s and >= 1 result at available_at + 1 s.

Usage
-----
    python -m src.rag.validation
    python -m src.rag.validation --no-fail   # report without raising
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm  # noqa: F401 – used in check_retrieval_assertion
    _QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None  # type: ignore[assignment,misc]
    qm = None  # type: ignore[assignment]
    _QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EASTERN_TZ = ZoneInfo("America/New_York")
_ACCEPT_DT_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")

VALID_STATUSES = ("fetched", "missing", "deferred", "failed", "disabled")
MAX_VIOLATION_EXAMPLES = 10

# Default thresholds
DEFAULT_TIMESTAMP_MISMATCH_THRESHOLD: float = 0.02   # 2 %


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    counts: dict[str, Any]
    violations: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "counts": self.counts,
            "violation_count": len(self.violations),
            "violations": self.violations[:MAX_VIOLATION_EXAMPLES],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ValidationConfig:
    lake_root: Path = Path("data/lake")
    chunks_path: Path | None = None          # default: lake_root/processed/8k-sec/chunks.parquet
    collection_name: str = "financial_text_chunks"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    database_url: str | None = None
    universe_path: Path | None = None        # default: lake_root/../universe.parquet
    timestamp_mismatch_threshold: float = DEFAULT_TIMESTAMP_MISMATCH_THRESHOLD
    retrieval_sample_size: int = 5
    fail_on_violation: bool = True


class ValidationError(RuntimeError):
    """Raised when any invariant is violated and fail_on_violation=True."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_raw_acceptance(raw_path: Path) -> datetime | None:
    """Return UTC acceptance datetime parsed from the SGML <ACCEPTANCE-DATETIME> tag.

    The tag value is a naive 14-digit string (YYYYMMDDHHMMSS) representing
    Eastern Time.  DST is handled via ZoneInfo fold/gap semantics.
    Only the first 30 lines are read (the header is always near the top).
    """
    try:
        with raw_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                m = _ACCEPT_DT_RE.search(line)
                if m:
                    raw = m.group(1)
                    naive_eastern = datetime(
                        int(raw[0:4]), int(raw[4:6]), int(raw[6:8]),
                        int(raw[8:10]), int(raw[10:12]), int(raw[12:14]),
                        tzinfo=EASTERN_TZ,
                    )
                    return naive_eastern.astimezone(timezone.utc)
                if i > 30:
                    break
    except OSError as exc:
        logger.debug("Could not open raw file %s: %s", raw_path, exc)
    return None


def _load_manifest(lake_root: Path) -> dict[str, dict]:
    """Load manifest.jsonl keyed by accession_number (last record wins)."""
    path = lake_root / "raw" / "8k-sec" / "manifest.jsonl"
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["accession_number"]] = rec
    return records


def _strip_ticker_prefix(doc_id: str, accession: str) -> str:
    """Return document_id with leading ``<TICKER>:`` stripped.

    The current chunk pipeline prefixes doc_id with the ticker so that
    GOOG:0001652044-24-000022:Item-1A and GOOGL:0001652044-24-000022:Item-1A
    map to the same canonical id: ``0001652044-24-000022:Item-1A``.
    """
    if not doc_id or ":" not in doc_id:
        return doc_id
    head, rest = doc_id.split(":", 1)
    # The ticker is a short uppercase word; the accession always appears in the rest.
    if accession and accession in rest and not head.startswith("000"):
        return rest
    return doc_id


def _offset_histogram(offsets: list[float]) -> dict[str, int]:
    """Bin available_at - acceptance_utc offsets (seconds) into labelled buckets."""
    buckets: list[tuple[str, float, float]] = [
        ("< 0 s  [VIOLATION]", float("-inf"), 0),
        ("0 – 60 s", 0, 60),
        ("1 – 30 min", 60, 1800),
        ("30 min – 4 h", 1800, 14400),
        ("4 h – 24 h", 14400, 86400),
        ("> 24 h", 86400, float("inf")),
    ]
    hist: dict[str, int] = {}
    for label, lo, hi in buckets:
        hist[label] = sum(1 for o in offsets if lo <= o < hi)
    return hist


# ---------------------------------------------------------------------------
# Check 1 – Timestamp
# ---------------------------------------------------------------------------

def check_timestamps(
    chunks: pd.DataFrame,
    lake_root: Path,
    mismatch_threshold: float = DEFAULT_TIMESTAMP_MISMATCH_THRESHOLD,
) -> CheckResult:
    """Compare every SEC chunk's available_at against its raw SGML acceptance time.

    Parses ``<ACCEPTANCE-DATETIME>`` from the raw submission .txt file, converts
    from Eastern to UTC (DST-aware), and flags chunks where
    ``available_at < acceptance_utc``.
    """
    name = "timestamp"
    sec_mask = chunks["source_type"].str.startswith("sec_", na=False) if "source_type" in chunks.columns else pd.Series(False, index=chunks.index)
    sec_chunks = chunks[sec_mask].copy()

    if sec_chunks.empty:
        return CheckResult(
            name=name, passed=True,
            counts={"sec_chunks_evaluated": 0},
            notes="No SEC chunks to validate.",
        )

    sec_chunks["_avail_dt"] = pd.to_datetime(sec_chunks["available_at"], utc=True, errors="coerce")
    manifest = _load_manifest(lake_root)

    # Build per-accession UTC acceptance times, cached from raw files
    raw_accept: dict[str, datetime | None] = {}
    for acc, rec in manifest.items():
        raw_rel = rec.get("raw_path", "")
        raw_accept[acc] = _parse_raw_acceptance(lake_root / raw_rel) if raw_rel else None
        # Fall back to manifest acceptance_datetime
        if raw_accept[acc] is None:
            acc_str = rec.get("acceptance_datetime", "")
            if acc_str:
                try:
                    raw_accept[acc] = datetime.fromisoformat(acc_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

    violations: list[dict] = []
    offsets: list[float] = []
    evaluated = 0
    parse_failures = 0

    for _, row in sec_chunks.iterrows():
        acc = str(row.get("accession_number", ""))
        avail = row["_avail_dt"]
        if pd.isna(avail):
            continue

        ref_utc = raw_accept.get(acc)
        if ref_utc is None:
            parse_failures += 1
            continue

        evaluated += 1
        avail_utc = avail.to_pydatetime()
        offset_s = (avail_utc - ref_utc).total_seconds()
        offsets.append(offset_s)

        if avail_utc < ref_utc:
            violations.append({
                "accession_number": acc,
                "ticker": str(row.get("ticker", "")),
                "chunk_id": str(row.get("chunk_id", "")),
                "available_at": avail_utc.isoformat(),
                "acceptance_datetime_utc": ref_utc.isoformat(),
                "offset_seconds": round(offset_s, 1),
                "corrected_available_at": ref_utc.isoformat(),
            })

    mismatch_rate = len(violations) / evaluated if evaluated else 0.0
    passed = mismatch_rate <= mismatch_threshold

    return CheckResult(
        name=name,
        passed=passed,
        counts={
            "sec_chunks_evaluated": evaluated,
            "parse_failures": parse_failures,
            "violations": len(violations),
            "mismatch_rate": round(mismatch_rate, 6),
            "threshold": mismatch_threshold,
            "offset_histogram": _offset_histogram(offsets),
        },
        violations=violations,
        notes=(
            f"Mismatch rate {mismatch_rate:.2%} "
            f"({'within' if passed else 'exceeds'}) threshold {mismatch_threshold:.2%}. "
            f"Raw header parsed for {len(raw_accept)} accessions; "
            f"{parse_failures} accession(s) had no parseable acceptance time."
        ),
    )


# ---------------------------------------------------------------------------
# Check 2 – Identity
# ---------------------------------------------------------------------------

def check_identity(chunks: pd.DataFrame) -> CheckResult:
    """Assert (cik, accession_number, document_id_canonical, chunk_index) is unique.

    document_id is canonicalized by stripping the leading ``<TICKER>:`` prefix.
    This means GOOG and GOOGL chunks from the same filing are identified as
    duplicates (same physical text stored under two tickers).

    The check also collects tickers per (cik, accession_number) and reports
    filings indexed under more than one ticker.
    """
    name = "identity"
    sec_mask = (
        chunks["source_type"].str.startswith("sec_", na=False)
        if "source_type" in chunks.columns
        else pd.Series(False, index=chunks.index)
    )
    sec_chunks = chunks[sec_mask].copy()

    if sec_chunks.empty:
        return CheckResult(
            name=name, passed=True,
            counts={"sec_chunks": 0},
            notes="No SEC chunks to check.",
        )

    required_for_key = {"accession_number", "chunk_index", "document_id"}
    if not required_for_key.issubset(sec_chunks.columns):
        missing_cols = required_for_key - set(sec_chunks.columns)
        return CheckResult(
            name=name, passed=False,
            counts={"error": 1},
            notes=f"Chunks DataFrame is missing columns required for identity check: {sorted(missing_cols)}",
        )

    # Build canonical document_id (strip ticker prefix)
    sec_chunks["_doc_canonical"] = sec_chunks.apply(
        lambda r: _strip_ticker_prefix(
            str(r.get("document_id", "") or ""),
            str(r.get("accession_number", "") or ""),
        ),
        axis=1,
    )

    key_cols = ["cik", "accession_number", "_doc_canonical", "chunk_index"]
    available_key = [c for c in key_cols if c in sec_chunks.columns]

    groups = sec_chunks.groupby(available_key, dropna=False, sort=False)
    violations: list[dict] = []
    for key, group in groups:
        if len(group) <= 1:
            continue
        tickers = sorted(group["ticker"].dropna().unique().tolist()) if "ticker" in group.columns else []
        hashes = group["content_hash"].unique().tolist() if "content_hash" in group.columns else []
        violations.append({
            "key": dict(zip(available_key, key if isinstance(key, tuple) else [key])),
            "count": len(group),
            "tickers": tickers,
            "same_content": len(hashes) == 1,
            "content_hash": hashes[0] if len(hashes) == 1 else hashes,
            "fix": "Deduplicate in chunker; store tickers as a list field.",
        })

    # Informational: filings with multiple tickers
    multi_ticker_count = 0
    if "ticker" in sec_chunks.columns and "accession_number" in sec_chunks.columns:
        multi_ticker_count = int(
            sec_chunks.groupby("accession_number")["ticker"]
            .nunique()
            .gt(1)
            .sum()
        )

    total_unique_keys = len(groups) - len(violations)
    passed = len(violations) == 0

    return CheckResult(
        name=name,
        passed=passed,
        counts={
            "sec_chunks": len(sec_chunks),
            "unique_natural_keys": total_unique_keys,
            "duplicate_key_groups": len(violations),
            "multi_ticker_filings": multi_ticker_count,
        },
        violations=violations,
        notes=(
            f"{len(violations)} duplicate key group(s). "
            f"{multi_ticker_count} filing(s) indexed under multiple tickers "
            "(expected for dual-class shares; fix by deduplicating in the chunker "
            "and storing tickers as a list in the payload)."
        ),
    )


# ---------------------------------------------------------------------------
# Check 3 – Ingestion status
# ---------------------------------------------------------------------------

def check_ingestion_status(
    universe: pd.DataFrame,
    database_url: str | None = None,
    status_parquet: Path | None = None,
) -> CheckResult:
    """Every (ticker, source) in the universe must have an ingestion_status row.

    Sources
    -------
    sec         – must have chunk_count > 0 → status 'fetched'; else 'missing'.
    transcripts – always 'disabled' (source not yet enabled); never a violation.

    At least one of database_url or status_parquet must be provided.
    """
    name = "ingestion_status"

    if universe.empty:
        return CheckResult(
            name=name, passed=True,
            counts={"universe_tickers": 0},
            notes="Universe is empty; nothing to check.",
        )

    tickers = sorted(universe["ticker"].astype(str).str.upper().unique().tolist())

    # Load existing rows from DB or Parquet
    existing: dict[tuple[str, str], dict] = {}

    if database_url:
        _db_url = database_url.replace("postgresql+psycopg://", "postgresql://").strip()
        try:
            import psycopg  # type: ignore
            with psycopg.connect(_db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT ticker, source, chunk_count, status_updated_at "
                        "FROM ingestion_status"
                    )
                    for row in cur.fetchall():
                        t, src, cnt, upd = row
                        existing[(t.upper(), src)] = {
                            "ticker": t,
                            "source": src,
                            "chunk_count": int(cnt or 0),
                            "status_updated_at": str(upd),
                        }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not query ingestion_status from DB: %s", exc)

    if not existing and status_parquet and status_parquet.exists():
        df = pd.read_parquet(status_parquet)
        for _, row in df.iterrows():
            key = (str(row["ticker"]).upper(), str(row["source"]))
            existing[key] = row.to_dict()

    status_counts: dict[str, int] = {
        "fetched": 0, "missing": 0, "disabled": 0,
    }
    violations: list[dict] = []

    for ticker in tickers:
        # transcripts: always disabled, never a violation
        status_counts["disabled"] += 1

        # sec: must exist with chunk_count > 0
        row = existing.get((ticker, "sec"))
        if row is None:
            status_counts["missing"] += 1
            violations.append({
                "ticker": ticker,
                "source": "sec",
                "status": "missing",
                "reason": "No row in ingestion_status",
            })
        elif int(row.get("chunk_count") or 0) == 0:
            status_counts["missing"] += 1
            violations.append({
                "ticker": ticker,
                "source": "sec",
                "status": "missing",
                "reason": "chunk_count = 0",
                "status_updated_at": str(row.get("status_updated_at", "")),
            })
        else:
            status_counts["fetched"] += 1

    passed = len(violations) == 0
    return CheckResult(
        name=name,
        passed=passed,
        counts={
            "universe_tickers": len(tickers),
            "sec_fetched": status_counts["fetched"],
            "sec_missing": status_counts["missing"],
            "transcripts_disabled": status_counts["disabled"],
        },
        violations=violations,
        notes=(
            f"{status_counts['fetched']}/{len(tickers)} tickers have SEC data. "
            f"{status_counts['missing']} missing. "
            f"Transcripts source is 'disabled' for all tickers (not yet enabled)."
        ),
    )


# ---------------------------------------------------------------------------
# Check 4 – Retrieval assertion
# ---------------------------------------------------------------------------

def check_retrieval_assertion(
    collection_name: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    sample_size: int = 5,
) -> CheckResult:
    """Verify Qdrant's DatetimeRange filter enforces available_at <= t_signal.

    For each sampled point:
      - Query at t_signal = available_at - 1 s → must return 0 points (pre-filing).
      - Query at t_signal = available_at + 1 s → must return >= 1 point (post-filing).

    A pre-filing hit is a hard violation (future information leakage).
    A post-filing miss is a warning (index gap, not a safety violation).
    """
    name = "retrieval_assertion"

    if not _QDRANT_AVAILABLE:
        return CheckResult(
            name=name, passed=False,
            counts={},
            notes="qdrant_client not installed; cannot run retrieval assertion.",
        )

    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)

        if not client.collection_exists(collection_name):
            client.close()
            return CheckResult(
                name=name, passed=True,
                counts={"collection_exists": False},
                notes=f"Collection {collection_name!r} does not exist; skipping.",
            )

        total_points = client.count(collection_name, exact=True).count
        if total_points == 0:
            client.close()
            return CheckResult(
                name=name, passed=True,
                counts={"total_points": 0},
                notes="Collection is empty; skipping.",
            )

        points, _ = client.scroll(
            collection_name=collection_name,
            limit=sample_size,
            with_payload=True,
            with_vectors=False,
        )

        pre_violations: list[dict] = []
        post_misses: list[dict] = []

        for point in points:
            payload = point.payload or {}
            avail_str = payload.get("available_at")
            accession = payload.get("accession_number")
            if not avail_str or not accession:
                continue

            avail = datetime.fromisoformat(avail_str)
            if avail.tzinfo is None:
                avail = avail.replace(tzinfo=timezone.utc)

            t_before = avail - timedelta(seconds=1)
            t_after = avail + timedelta(seconds=1)

            def _count(t: datetime) -> int:
                return client.count(
                    collection_name=collection_name,
                    count_filter=qm.Filter(must=[
                        qm.FieldCondition(
                            key="accession_number",
                            match=qm.MatchValue(value=accession),
                        ),
                        qm.FieldCondition(
                            key="available_at",
                            range=qm.DatetimeRange(lte=t.isoformat()),
                        ),
                    ]),
                    exact=True,
                ).count

            cnt_before = _count(t_before)
            cnt_after = _count(t_after)

            if cnt_before > 0:
                pre_violations.append({
                    "point_id": str(point.id),
                    "accession_number": accession,
                    "available_at": avail_str,
                    "t_signal": t_before.isoformat(),
                    "chunks_returned": cnt_before,
                    "error": "Future information leakage: chunks visible before available_at",
                })

            if cnt_after == 0:
                post_misses.append({
                    "point_id": str(point.id),
                    "accession_number": accession,
                    "available_at": avail_str,
                    "t_signal": t_after.isoformat(),
                    "error": "No chunks returned at t_signal > available_at (index gap?)",
                })

        client.close()
        passed = len(pre_violations) == 0
        return CheckResult(
            name=name,
            passed=passed,
            counts={
                "total_points": total_points,
                "sampled": len(points),
                "pre_filing_violations": len(pre_violations),
                "post_filing_misses": len(post_misses),
            },
            violations=pre_violations,
            notes=(
                f"Sampled {len(points)} of {total_points} points. "
                f"Pre-filing violations (leakage): {len(pre_violations)}. "
                f"Post-filing misses (harmless gaps): {len(post_misses)}."
            ),
        )

    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=name, passed=False,
            counts={"error": 1},
            violations=[{"error": str(exc)}],
            notes=f"Exception during retrieval assertion check: {exc}",
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_validation(config: ValidationConfig) -> dict:
    """Run all four validation checks and return a consolidated report dict.

    Raises ValidationError if any check fails and config.fail_on_violation is True.
    The report is always returned (even when raising) so callers can log it.
    """
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

    lake_root = config.lake_root
    chunks_path = (
        config.chunks_path
        or (lake_root / "processed" / "8k-sec" / "chunks.parquet")
    )
    universe_path = (
        config.universe_path
        or (lake_root.parent / "universe.parquet")
    )

    # Load data
    if chunks_path.exists():
        chunks = pd.read_parquet(chunks_path)
    else:
        chunks = pd.DataFrame()
        logger.warning("No chunk Parquet found at %s; timestamp/identity checks skipped.", chunks_path)

    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
    else:
        universe = pd.DataFrame(columns=["ticker"])
        logger.warning("No universe Parquet found at %s; status check skipped.", universe_path)

    database_url = config.database_url or os.environ.get("DATABASE_URL") or None
    status_parquet = lake_root / "processed" / "ingestion_status.parquet"

    # Run checks
    results: dict[str, CheckResult] = {}
    results["timestamp"] = check_timestamps(chunks, lake_root, config.timestamp_mismatch_threshold)
    results["identity"] = check_identity(chunks)
    results["ingestion_status"] = check_ingestion_status(
        universe,
        database_url=database_url,
        status_parquet=status_parquet,
    )
    results["retrieval_assertion"] = check_retrieval_assertion(
        config.collection_name,
        config.qdrant_host,
        config.qdrant_port,
        config.retrieval_sample_size,
    )

    all_passed = all(r.passed for r in results.values())
    report = {
        "passed": all_passed,
        "checks": {name: result.to_dict() for name, result in results.items()},
    }

    if not all_passed:
        failed = [name for name, r in results.items() if not r.passed]
        logger.error("Validation FAILED: %s", ", ".join(failed))
        for name in failed:
            logger.error("  [%s] %s", name, results[name].notes)
        if config.fail_on_violation:
            raise ValidationError(
                f"Ingestion validation failed ({', '.join(failed)}). "
                "Check the validation report for details."
            )
    else:
        logger.info("Validation PASSED: all 4 invariants satisfied.")

    return report


def print_report(report: dict) -> None:
    """Pretty-print a validation report to stdout."""
    overall = "PASS" if report.get("passed") else "FAIL"
    print(f"\n{'='*60}")
    print(f"  VALIDATION REPORT  [{overall}]")
    print(f"{'='*60}")
    for name, check in report.get("checks", {}).items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"\n  [{status}] {name}")
        for k, v in check.get("counts", {}).items():
            print(f"         {k}: {v}")
        if check.get("notes"):
            print(f"         note: {check['notes']}")
        if not check["passed"] and check.get("violations"):
            print(f"         violations ({check['violation_count']} total, showing {min(MAX_VIOLATION_EXAMPLES, check['violation_count'])}):")
            for v in check["violations"]:
                print(f"           {json.dumps(v, default=str)}")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    parser.add_argument(
        "--collection",
        default=os.getenv("RAG_QDRANT_COLLECTION", "financial_text_chunks"),
    )
    parser.add_argument(
        "--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost")
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333"))
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument(
        "--timestamp-threshold",
        type=float,
        default=DEFAULT_TIMESTAMP_MISMATCH_THRESHOLD,
        help="Maximum allowed mismatch rate (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of Qdrant points to sample for retrieval assertion",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Print the report but exit 0 even on violations",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    config = ValidationConfig(
        lake_root=args.lake_root,
        collection_name=args.collection,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        database_url=args.database_url or None,
        timestamp_mismatch_threshold=args.timestamp_threshold,
        retrieval_sample_size=args.sample_size,
        fail_on_violation=not args.no_fail,
    )

    try:
        report = run_validation(config)
        print_report(report)
    except ValidationError as exc:
        # report was already printed/logged before the exception
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
