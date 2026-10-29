"""Backtest-derived acquisition ranges and point-in-time text coverage reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")


def normalize_signal_time(value: object) -> pd.Timestamp:
    """Normalize timestamps; date-only/naive signal times mean New York market close."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        if timestamp.time() == time.min:
            timestamp = timestamp.normalize() + pd.Timedelta(hours=16)
        timestamp = timestamp.tz_localize(MARKET_TZ)
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class BacktestCoverageConfig:
    backtest_start: date
    backtest_end: date
    recency_8k_days: int = 90
    recency_10q_days: int = 180
    latest_10k_days: int = 550
    transcripts_per_leg: int = 2
    transcript_quarter_days: int = 92
    max_none_coverage_share: float = 0.05
    coverage_failure_mode: Literal["fail", "warn"] = "fail"

    def __post_init__(self) -> None:
        if self.backtest_end < self.backtest_start:
            raise ValueError("backtest_end must be on or after backtest_start")
        if self.backtest_end > date.today():
            raise ValueError("backtest_end cannot be in the future")
        if min(
            self.recency_8k_days,
            self.recency_10q_days,
            self.latest_10k_days,
            self.transcripts_per_leg,
            self.transcript_quarter_days,
        ) < 1:
            raise ValueError("retrieval windows and transcript count must be positive")
        if not 0 <= self.max_none_coverage_share <= 1:
            raise ValueError("max_none_coverage_share must be between 0 and 1")

    @property
    def max_retrieval_window_days(self) -> int:
        return max(
            self.recency_8k_days,
            self.recency_10q_days,
            self.latest_10k_days,
            self.transcripts_per_leg * self.transcript_quarter_days,
        )

    @property
    def fetch_start(self) -> date:
        return self.backtest_start - timedelta(days=self.max_retrieval_window_days)

    def as_dict(self) -> dict:
        return {
            "backtest_start": self.backtest_start.isoformat(),
            "backtest_end": self.backtest_end.isoformat(),
            "recency_8k_days": self.recency_8k_days,
            "recency_10q_days": self.recency_10q_days,
            "latest_10k_days": self.latest_10k_days,
            "transcripts_per_leg": self.transcripts_per_leg,
            "transcript_quarter_days": self.transcript_quarter_days,
            "max_retrieval_window_days": self.max_retrieval_window_days,
            "fetch_start": self.fetch_start.isoformat(),
            "fetch_end": date.today().isoformat(),
            "max_none_coverage_share": self.max_none_coverage_share,
            "coverage_failure_mode": self.coverage_failure_mode,
        }


def quarter_range(start: date, end: date) -> list[str]:
    """Return every calendar quarter intersecting the inclusive date range."""
    if end < start:
        raise ValueError("quarter range end must be on or after start")
    year, quarter = start.year, (start.month - 1) // 3 + 1
    final_year, final_quarter = end.year, (end.month - 1) // 3 + 1
    labels = []
    while (year, quarter) <= (final_year, final_quarter):
        labels.append(f"{year}Q{quarter}")
        quarter += 1
        if quarter == 5:
            year, quarter = year + 1, 1
    return labels


def coverage_report(
    signals: list[dict],
    chunks: pd.DataFrame,
    *,
    recency_8k_days: int = 90,
    recency_10q_days: int = 180,
    latest_10k_days: int = 550,
    transcripts_per_leg: int = 2,
    max_none_coverage_share: float = 0.05,
    failure_mode: Literal["fail", "warn"] = "fail",
) -> dict:
    """Classify each leg by eligible source coverage at its signal timestamp.

    Full means recent 8-K evidence, an available 10-K, and the requested number
    of distinct transcripts are all represented. Partial means some eligible
    chunks exist but one or more of those source scopes is absent; none means
    there are no eligible chunks for that leg.
    """
    if not 0 <= max_none_coverage_share <= 1:
        raise ValueError("max_none_coverage_share must be between 0 and 1")
    if failure_mode not in {"fail", "warn"}:
        raise ValueError("failure_mode must be 'fail' or 'warn'")
    required = {"ticker", "available_at", "source_type"}
    missing = required - set(chunks.columns)
    if missing:
        raise ValueError(f"coverage chunks are missing columns: {sorted(missing)}")

    frame = chunks.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    if frame["available_at"].isna().any():
        raise ValueError("coverage chunks contain invalid available_at timestamps")
    rows = []
    ordered_signals = sorted(signals, key=lambda item: pd.Timestamp(item["t_signal"]))
    for signal_index, signal in enumerate(ordered_signals):
        signal_time = normalize_signal_time(signal["t_signal"])
        signal_id = signal.get("signal_id")
        if signal_id is None and {"signal_time", "pair_id", "model_id"} <= set(signal):
            signal_id = (
                f"{pd.Timestamp(signal['signal_time']).date().isoformat()}:"
                f"{signal['pair_id']}:{signal['model_id']}"
            )
        if signal_id is None:
            signal_id = f"signal-{signal_index + 1}"
        for ticker in signal["pair"]:
            eligible = frame[
                (frame["ticker"] == str(ticker).upper())
                & (frame["available_at"] <= signal_time)
            ]
            if eligible.empty:
                status = "none"
                has_recent_8k = has_recent_10q = has_10k = False
                transcript_count = 0
            else:
                source_types = eligible["source_type"].astype(str)
                has_recent_8k = bool((
                    source_types.eq("sec_8k")
                    & (eligible["available_at"] >= signal_time - pd.Timedelta(days=recency_8k_days))
                ).any())
                has_recent_10q = bool((
                    source_types.eq("sec_10q")
                    & (eligible["available_at"] >= signal_time - pd.Timedelta(days=recency_10q_days))
                ).any())
                has_10k = bool((
                    source_types.eq("sec_10k")
                    & (eligible["available_at"] >= signal_time - pd.Timedelta(days=latest_10k_days))
                ).any())
                transcripts = eligible[source_types.eq("earnings_transcript")]
                transcript_key = (
                    "transcript_id" if "transcript_id" in transcripts
                    else "document_id" if "document_id" in transcripts
                    else None
                )
                transcript_count = (
                    int(transcripts[transcript_key].dropna().nunique())
                    if transcript_key else len(transcripts)
                )
                status = (
                    "full"
                    if (
                        has_recent_8k and has_recent_10q and has_10k
                        and transcript_count >= transcripts_per_leg
                    )
                    else "partial"
                )
            rows.append({
                "signal_id": str(signal_id),
                "t_signal": signal_time.isoformat(),
                "ticker": str(ticker).upper(),
                "coverage": status,
                "eligible_chunks": int(len(eligible)),
                "recent_8k": has_recent_8k,
                "recent_10q": has_recent_10q,
                "has_10k": has_10k,
                "transcripts_available": transcript_count,
            })

    report_frame = pd.DataFrame(rows, columns=[
        "signal_id", "t_signal", "ticker", "coverage", "eligible_chunks",
        "recent_8k", "recent_10q", "has_10k", "transcripts_available",
    ])
    total = len(report_frame)
    distribution = {
        label: int((report_frame["coverage"] == label).sum()) if total else 0
        for label in ("full", "partial", "none")
    }
    none_share = distribution["none"] / total if total else 0.0
    report = {
        "signals_tested": len(signals),
        "legs_tested": total,
        "distribution": distribution,
        "none_share": none_share,
        "max_none_coverage_share": max_none_coverage_share,
        "coverage_failure_mode": failure_mode,
        "by_leg": rows,
        "within_threshold": none_share <= max_none_coverage_share,
    }
    if not report["within_threshold"] and failure_mode == "fail":
        raise ValueError(
            f"no-evidence coverage share {none_share:.3f} exceeds configured "
            f"maximum {max_none_coverage_share:.3f}"
        )
    return report


def build_ingestion_status(
    chunks: pd.DataFrame,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize indexed chunk timestamp bounds for each ticker and source.

    Returns columns:
        ticker, source, earliest_available_at, latest_available_at,
        chunk_count, status

    Status values
    -------------
    fetched   – chunk_count > 0
    missing   – row is implied by the universe but chunk_count == 0
    disabled  – transcripts source (not yet enabled)
    """
    columns = [
        "ticker", "source", "earliest_available_at", "latest_available_at",
        "chunk_count", "status",
    ]
    if chunks.empty:
        if tickers:
            rows = []
            for ticker in dict.fromkeys(tickers):
                t = ticker.strip().upper()
                rows.append({
                    "ticker": t, "source": "sec",
                    "earliest_available_at": pd.NaT, "latest_available_at": pd.NaT,
                    "chunk_count": 0, "status": "missing",
                })
                rows.append({
                    "ticker": t, "source": "transcripts",
                    "earliest_available_at": pd.NaT, "latest_available_at": pd.NaT,
                    "chunk_count": 0, "status": "disabled",
                })
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame(columns=columns)
    frame = chunks.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    source_map = {
        "sec_8k": "sec",
        "sec_10k": "sec",
        "sec_10q": "sec",
        "earnings_transcript": "transcripts",
    }
    unknown_sources = set(frame["source_type"].dropna()) - set(source_map)
    if unknown_sources:
        raise ValueError(f"unsupported chunk source types: {sorted(unknown_sources)}")
    frame["source"] = frame["source_type"].map(
        source_map
    )
    status = (
        frame.groupby(["ticker", "source"], as_index=False)
        .agg(
            earliest_available_at=("available_at", "min"),
            latest_available_at=("available_at", "max"),
            chunk_count=("available_at", "size"),
        )
    )
    if tickers:
        universe = pd.MultiIndex.from_product(
            [
                list(dict.fromkeys(t.strip().upper() for t in tickers)),
                ("sec", "transcripts"),
            ],
            names=("ticker", "source"),
        ).to_frame(index=False)
        status = universe.merge(status, on=["ticker", "source"], how="left")
        status["chunk_count"] = status["chunk_count"].fillna(0).astype(int)

    def _derive_status(row: "pd.Series") -> str:
        if row["source"] == "transcripts":
            return "disabled"
        return "fetched" if int(row["chunk_count"] or 0) > 0 else "missing"

    status["status"] = status.apply(_derive_status, axis=1)
    return status[columns]



def coverage_for_signal(report: dict, signal_id: str) -> dict:
    """Reduce the pre-replay leg rows to coverage metadata for one signal."""
    rows = [row for row in report["by_leg"] if str(row["signal_id"]) == str(signal_id)]
    if not rows:
        raise KeyError(f"coverage report has no signal {signal_id!r}")
    labels = [row["coverage"] for row in rows]
    status = (
        "none" if all(label == "none" for label in labels)
        else "full" if all(label == "full" for label in labels)
        else "partial"
    )
    return {
        "status": status,
        "legs": {row["ticker"]: row for row in rows},
        "none_legs": [row["ticker"] for row in rows if row["coverage"] == "none"],
    }
