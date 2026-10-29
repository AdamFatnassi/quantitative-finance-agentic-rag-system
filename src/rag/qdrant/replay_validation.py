"""Chronological point-in-time replay metrics for signal-context retrieval."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient

from src.rag.coverage import (
    BacktestCoverageConfig,
    coverage_report,
    normalize_signal_time,
)
from src.rag.qdrant.retrieve_context import (
    QueryEncoder,
    RetrievalConfig,
    retrieve_context,
)
from src.rag.qdrant.qdrant_lifecycle import get_client

logger = logging.getLogger(__name__)


def replay_signals(
    signals: list[dict],
    client: QdrantClient,
    config: RetrievalConfig = RetrievalConfig(),
    encoder: QueryEncoder | None = None,
) -> dict[str, int | float]:
    """Replay signals by event time and report PIT violations and estimated hits."""
    normalized_signals = [
        {**signal, "t_signal": normalize_signal_time(signal["t_signal"]).isoformat()}
        for signal in signals
    ]
    ordered_signals = sorted(
        normalized_signals, key=lambda signal: pd.Timestamp(signal["t_signal"])
    )
    violations = 0
    retrieval_count = 0
    estimated_count = 0
    for signal in ordered_signals:
        context = retrieve_context(signal, client, config, encoder)
        signal_time = normalize_signal_time(signal["t_signal"])
        for leg_chunks in context["legs"].values():
            for chunk in leg_chunks:
                retrieval_count += 1
                available_at = pd.Timestamp(chunk["available_at"])
                if available_at > signal_time:
                    violations += 1
                if chunk["payload"].get("availability_estimated") is True:
                    estimated_count += 1
    if violations:
        raise AssertionError(
            f"point-in-time replay found {violations} future chunk(s)"
        )
    return {
        "signals_tested": len(ordered_signals),
        "retrievals": retrieval_count,
        "violations": violations,
        "estimated_retrieval_share": (
            estimated_count / retrieval_count if retrieval_count else 0.0
        ),
    }


def load_indexed_chunks(
    client: QdrantClient,
    collection_name: str,
    page_size: int = 512,
) -> pd.DataFrame:
    """Read indexed metadata from Qdrant for an exact pre-replay coverage audit."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    columns = [
        "ticker", "available_at", "source_type", "transcript_id", "document_id",
    ]
    rows = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=columns,
            with_vectors=False,
        )
        rows.extend(point.payload or {} for point in points)
        if offset is None:
            break
    return pd.DataFrame(rows, columns=columns)


def check_replay_coverage(
    signals: list[dict],
    chunks: pd.DataFrame,
    config: BacktestCoverageConfig,
) -> dict:
    """Generate the leg-level coverage report and enforce its configured gate."""
    report = coverage_report(
        signals,
        chunks,
        recency_8k_days=config.recency_8k_days,
        recency_10q_days=config.recency_10q_days,
        latest_10k_days=config.latest_10k_days,
        transcripts_per_leg=config.transcripts_per_leg,
        max_none_coverage_share=config.max_none_coverage_share,
        failure_mode=config.coverage_failure_mode,
    )
    report["config"] = config.as_dict()
    return report


def signals_from_frame(frame: pd.DataFrame) -> list[dict]:
    """Convert the quantitative OOS signal artifact to active replay signals."""
    if "signal" in frame:
        frame = frame[frame["signal"].fillna(0).ne(0)]
    signals = []
    for row in frame.to_dict(orient="records"):
        pair = row.get("pair", (row.get("stock_a"), row.get("stock_b")))
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(pd.isna(v) for v in pair):
            raise ValueError("signal records must contain pair or stock_a/stock_b")
        signal_time = row.get("t_signal", row.get("time"))
        if signal_time is None or pd.isna(signal_time):
            raise ValueError("signal records must include t_signal or time")
        signal_id = row.get("signal_id")
        if signal_id is None or pd.isna(signal_id):
            signal_id = f"{signal_time}:{pair[0]}:{pair[1]}:{len(signals)}"
        signals.append({
            **row,
            "signal_id": str(signal_id),
            "pair": pair,
            "t_signal": normalize_signal_time(signal_time).isoformat(),
        })
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check point-in-time text coverage before replaying signals."
    )
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--backtest-start", type=date.fromisoformat, required=True)
    parser.add_argument("--backtest-end", type=date.fromisoformat, required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--recency-8k-days", type=int, default=90)
    parser.add_argument("--recency-10q-days", type=int, default=180)
    parser.add_argument("--latest-10k-days", type=int, default=550)
    parser.add_argument("--transcripts-per-leg", type=int, default=2)
    parser.add_argument("--max-none-share", type=float, default=0.05)
    parser.add_argument("--on-exceed", choices=("fail", "warn"), default="fail")
    parser.add_argument("--report", type=Path, default=Path("data/lake/processed/coverage_report.json"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.signals)
    signals = signals_from_frame(frame)
    config = BacktestCoverageConfig(
        backtest_start=args.backtest_start,
        backtest_end=args.backtest_end,
        recency_8k_days=args.recency_8k_days,
        recency_10q_days=args.recency_10q_days,
        latest_10k_days=args.latest_10k_days,
        transcripts_per_leg=args.transcripts_per_leg,
        max_none_coverage_share=args.max_none_share,
        coverage_failure_mode=args.on_exceed,
    )
    retrieval_config = RetrievalConfig()
    collection = args.collection or retrieval_config.collection_name
    client = get_client(retrieval_config.qdrant_host, retrieval_config.qdrant_port)
    try:
        chunks = load_indexed_chunks(client, collection)
    finally:
        client.close()
    report_config = (
        replace(config, coverage_failure_mode="warn")
        if config.coverage_failure_mode == "fail"
        else config
    )
    report = check_replay_coverage(signals, chunks, report_config)
    report["coverage_failure_mode"] = config.coverage_failure_mode
    report["config"]["coverage_failure_mode"] = config.coverage_failure_mode
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "signals_tested": report["signals_tested"],
        "legs_tested": report["legs_tested"],
        "distribution": report["distribution"],
        "none_share": report["none_share"],
        "within_threshold": report["within_threshold"],
        "report": str(args.report),
    }, indent=2))
    if not report["within_threshold"] and args.on_exceed == "warn":
        logger.warning(
            "No-evidence coverage share %.3f exceeds configured maximum %.3f",
            report["none_share"],
            args.max_none_share,
        )
    elif not report["within_threshold"]:
        raise SystemExit(
            f"coverage report saved to {args.report}; none share "
            f"{report['none_share']:.3f} exceeds {args.max_none_share:.3f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
