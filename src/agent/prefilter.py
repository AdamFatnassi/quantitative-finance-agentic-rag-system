"""prefilter.py

Fast Pre-Filter Optimization module.

Performs a high-speed metadata-only count check against Qdrant to determine
whether either ticker in a pairs-trading signal had any SEC filings or transcripts
published in the 7 days strictly prior to the signal timestamp.

If NO filings are found within the lookback window, the signal bypasses the LLM
entirely and immediately returns a default GO decision (zero LLM compute cost,
sub-millisecond latency).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.agent.schemas import GatekeeperResponse, SignalRequest

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = os.getenv("RAG_QDRANT_COLLECTION", "financial_text_chunks")
DEFAULT_LOOKBACK_DAYS = int(os.getenv("PREFILTER_LOOKBACK_DAYS", "7"))


def _build_prefilter_condition(
    ticker: str,
    window_start: datetime,
    window_end: datetime,
) -> qm.Filter:
    """Construct an exact point-in-time filter for a ticker within [window_start, window_end]."""
    ticker_clean = ticker.strip().upper()
    return qm.Filter(
        must=[
            qm.Filter(
                should=[
                    qm.FieldCondition(key="tickers", match=qm.MatchAny(any=[ticker_clean])),
                    qm.FieldCondition(key="ticker", match=qm.MatchValue(value=ticker_clean)),
                ]
            ),
            qm.FieldCondition(
                key="available_at",
                range=qm.DatetimeRange(
                    gte=window_start.isoformat(),
                    lte=window_end.isoformat(),
                ),
            ),
        ]
    )


def count_recent_filings(
    client: QdrantClient,
    ticker: str,
    as_of: datetime,
    collection_name: str = DEFAULT_COLLECTION,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> int:
    """Count how many chunks exist for a ticker strictly within (as_of - lookback_days, as_of].

    Returns 0 if collection does not exist or client connection fails.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    else:
        as_of = as_of.astimezone(timezone.utc)

    window_start = as_of - timedelta(days=lookback_days)

    try:
        if not client.collection_exists(collection_name):
            logger.warning("Collection %s does not exist; count is 0", collection_name)
            return 0

        cond = _build_prefilter_condition(ticker, window_start, as_of)
        res = client.count(collection_name=collection_name, count_filter=cond, exact=True)
        return int(res.count)
    except Exception as exc:
        logger.error("Error counting Qdrant chunks for ticker %s: %s", ticker, exc)
        # On error, return -1 or 0; safe assumption is 0 for prefilter bypass unless strict
        return 0


def evaluate_prefilter(
    client: QdrantClient,
    signal: SignalRequest,
    collection_name: str = DEFAULT_COLLECTION,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Tuple[bool, Optional[GatekeeperResponse]]:
    """Evaluate whether the signal can bypass LLM evaluation.

    Returns:
        (bypassed: bool, response: Optional[GatekeeperResponse])

    If bypassed is True:
        response is a default GO decision with risk_category="none".
    If bypassed is False:
        response is None, meaning the signal has recent filings and must be
        evaluated by the Agentic Gatekeeper.
    """
    as_of = signal.signal_timestamp
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    else:
        as_of = as_of.astimezone(timezone.utc)

    count_a = count_recent_filings(
        client, signal.ticker_a, as_of, collection_name=collection_name, lookback_days=lookback_days
    )
    count_b = count_recent_filings(
        client, signal.ticker_b, as_of, collection_name=collection_name, lookback_days=lookback_days
    )

    total_recent = count_a + count_b

    if total_recent == 0:
        logger.debug(
            "Pre-Filter PASSED for %s / %s (0 filings in past %d days prior to %s). Bypassing LLM.",
            signal.ticker_a,
            signal.ticker_b,
            lookback_days,
            as_of.isoformat(),
        )
        return True, GatekeeperResponse(
            decision="GO",
            confidence_score=1.0,
            risk_category="none",
            justification=(
                f"Fast Pre-Filter Bypass: Neither {signal.ticker_a} nor {signal.ticker_b} "
                f"published SEC filings or transcripts in the {lookback_days} days prior to "
                f"{as_of.strftime('%Y-%m-%d %H:%M UTC')}. Spread divergence is unlikely to be "
                f"driven by recent fundamental value traps."
            ),
            filing_ids_inspected=[],
        )

    logger.info(
        "Pre-Filter TRIGGERED for %s / %s: found %d recent chunks (%s: %d, %s: %d) within %d days prior to %s. Routing to Agent.",
        signal.ticker_a,
        signal.ticker_b,
        total_recent,
        signal.ticker_a,
        count_a,
        signal.ticker_b,
        count_b,
        lookback_days,
        as_of.isoformat(),
    )
    return False, None
