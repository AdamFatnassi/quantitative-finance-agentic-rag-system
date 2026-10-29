"""tools.py

Qdrant tool and vector retrieval interface for the Agentic Risk Gatekeeper.

Enforces strict point-in-time constraints (zero look-ahead bias) on SEC filings
and transcripts when retrieving evidence for fundamental risk evaluation.

Also exposes `run_monte_carlo_simulation` — a quantitative spread stress-testing
tool backed by the Ornstein-Uhlenbeck Monte Carlo engine in `src.quant.monte_carlo`.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.agent.schemas import FilingChunk

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = os.getenv("RAG_QDRANT_COLLECTION", "financial_text_chunks")
DEFAULT_DENSE_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_SPARSE_MODEL = os.getenv("RAG_SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

_encoder_instance = None


def get_qdrant_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    """Return a configured Qdrant client."""
    return QdrantClient(host=host, port=port, timeout=10.0)


def _get_encoder(dense_model: str = DEFAULT_DENSE_MODEL, sparse_model: str = DEFAULT_SPARSE_MODEL):
    """Lazy-load the FastEmbed encoder for semantic queries."""
    global _encoder_instance
    if _encoder_instance is None:
        try:
            from src.rag.qdrant.ingest_chunks import FastEmbedEncoder
            _encoder_instance = FastEmbedEncoder(dense_model=dense_model, sparse_model=sparse_model)
        except Exception as exc:
            logger.warning("FastEmbedEncoder could not be loaded: %s; falling back to metadata scroll.", exc)
            return None
    return _encoder_instance


def _build_strict_pit_filter(ticker: str, as_of: datetime) -> qm.Filter:
    """Build a point-in-time filter for a ticker with available_at <= as_of."""
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
                range=qm.DatetimeRange(lte=as_of.isoformat()),
            ),
        ]
    )


def search_sec_filings(
    ticker: str,
    as_of_timestamp: datetime,
    query: str = "",
    limit: int = 5,
    client: Optional[QdrantClient] = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[FilingChunk]:
    """Retrieve top relevant SEC filing chunks for a given ticker strictly before as_of_timestamp.

    Guarantees:
      - available_at <= as_of_timestamp (zero look-ahead bias).
      - Returns top `limit` relevant chunks (default 3 to 5) with their published_at and section_type.

    Parameters:
      ticker: Equity ticker symbol (e.g. 'BA', 'AAPL').
      as_of_timestamp: Point-in-time cutoff.
      query: Optional semantic search query (e.g. 'restatement lawsuit bankruptcy').
             If empty or encoder unavailable, retrieves most recent filings by date.
      limit: Maximum chunks to return (default 5).
      client: Optional active Qdrant client. If None, a temporary client is created.
      collection_name: Qdrant collection name.

    Returns:
      List[FilingChunk] sorted by relevance or recency.
    """
    if as_of_timestamp.tzinfo is None:
        as_of = as_of_timestamp.replace(tzinfo=timezone.utc)
    else:
        as_of = as_of_timestamp.astimezone(timezone.utc)

    close_client = False
    if client is None:
        client = get_qdrant_client()
        close_client = True

    chunks: List[FilingChunk] = []

    try:
        if not client.collection_exists(collection_name):
            logger.warning("Collection %s does not exist in Qdrant.", collection_name)
            return []

        pit_filter = _build_strict_pit_filter(ticker, as_of)
        encoder = _get_encoder() if query.strip() else None

        if query.strip() and encoder is not None:
            # Semantic hybrid search with RRF fusion
            try:
                dense, sparse = encoder.embed([query.strip()])
                dense_vec = [float(v) for v in dense[0]]
                sparse_vec = qm.SparseVector(
                    indices=[int(idx) for idx in sparse[0].indices],
                    values=[float(val) for val in sparse[0].values],
                )

                response = client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        qm.Prefetch(
                            query=dense_vec,
                            using="dense",
                            filter=pit_filter,
                            limit=limit,
                        ),
                        qm.Prefetch(
                            query=sparse_vec,
                            using="sparse",
                            filter=pit_filter,
                            limit=limit,
                        ),
                    ],
                    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                    query_filter=pit_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )

                for pt in response.points:
                    payload = pt.payload or {}
                    # Secondary safety assertion against lookahead
                    avail = payload.get("available_at")
                    if avail:
                        dt = datetime.fromisoformat(avail)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt > as_of:
                            logger.error("LOOKAHEAD VIOLATION BLOCKED: chunk %s has %s > %s", pt.id, avail, as_of)
                            continue

                    chunk_id = str(payload.get("chunk_id") or pt.id)
                    chunks.append(
                        FilingChunk(
                            chunk_id=chunk_id,
                            ticker=str(payload.get("ticker", ticker)).upper(),
                            document_id=str(payload.get("document_id", "")),
                            accession_number=payload.get("accession_number"),
                            form_type=payload.get("form_type"),
                            section_type=payload.get("section_type"),
                            item_code=payload.get("item_code"),
                            available_at=str(payload.get("available_at", "")),
                            published_at=payload.get("published_at"),
                            text=str(payload.get("text", "")),
                            score=float(pt.score) if pt.score is not None else None,
                        )
                    )
                return chunks

            except Exception as search_err:
                logger.warning("Hybrid search failed (%s); falling back to recency scroll.", search_err)

        # Fallback: Scroll most recent filings before as_of
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=pit_filter,
            limit=limit,
            order_by=qm.OrderBy(key="available_at", direction=qm.Direction.DESC),
            with_payload=True,
            with_vectors=False,
        )

        for pt in points:
            payload = pt.payload or {}
            avail = payload.get("available_at")
            if avail:
                dt = datetime.fromisoformat(avail)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > as_of:
                    continue

            chunk_id = str(payload.get("chunk_id") or pt.id)
            chunks.append(
                FilingChunk(
                    chunk_id=chunk_id,
                    ticker=str(payload.get("ticker", ticker)).upper(),
                    document_id=str(payload.get("document_id", "")),
                    accession_number=payload.get("accession_number"),
                    form_type=payload.get("form_type"),
                    section_type=payload.get("section_type"),
                    item_code=payload.get("item_code"),
                    available_at=str(payload.get("available_at", "")),
                    published_at=payload.get("published_at"),
                    text=str(payload.get("text", "")),
                    score=None,
                )
            )

    except Exception as exc:
        logger.error("Qdrant retrieval error for ticker %s: %s", ticker, exc)
    finally:
        if close_client:
            client.close()

    return chunks


def retrieve_pair_filings(
    ticker_a: str,
    ticker_b: str,
    as_of_timestamp: datetime,
    query: str = "",
    limit_per_leg: int = 5,
    client: Optional[QdrantClient] = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, List[FilingChunk]]:
    """Retrieve top filing evidence for both legs of a pairs-trading candidate."""
    chunks_a = search_sec_filings(
        ticker=ticker_a,
        as_of_timestamp=as_of_timestamp,
        query=query,
        limit=limit_per_leg,
        client=client,
        collection_name=collection_name,
    )
    chunks_b = search_sec_filings(
        ticker=ticker_b,
        as_of_timestamp=as_of_timestamp,
        query=query,
        limit=limit_per_leg,
        client=client,
        collection_name=collection_name,
    )
    return {
        ticker_a: chunks_a,
        ticker_b: chunks_b,
    }


# ─── Monte Carlo Spread Simulation Tool ────────────────────────────────────────

_PROCESSED_BASE = Path(__file__).resolve().parents[2] / "data" / "lake" / "processed"
_PRICES_PATH = _PROCESSED_BASE / "assets" / "prices_clean.parquet"
if not _PRICES_PATH.exists():
    _PRICES_PATH = _PROCESSED_BASE / "prices_clean.parquet"

_ANALYTICS_BASE = Path(__file__).resolve().parents[2] / "data" / "lake" / "analytics"
_PAIR_METRICS_PATH = _ANALYTICS_BASE / "assets" / "pair_metrics.parquet"
if not _PAIR_METRICS_PATH.exists():
    _PAIR_METRICS_PATH = _ANALYTICS_BASE / "pair_metrics.parquet"

_prices_cache: Optional[Any] = None   # module-level cache to avoid repeated disk I/O


def _load_prices() -> Any:
    """Lazy-load and cache the prices_clean parquet (wide DataFrame, tickers as columns)."""
    global _prices_cache
    if _prices_cache is None:
        import pandas as pd
        _prices_cache = pd.read_parquet(_PRICES_PATH)
    return _prices_cache


def run_monte_carlo_simulation(
    ticker_a: str,
    ticker_b: str,
    as_of: datetime,
    lookback_days: int = 252,
    horizon_days: int = 10,
    num_simulations: int = 1000,
) -> dict:
    """Quantitative spread stress-test tool for the Agentic Risk Gatekeeper.

    Runs `num_simulations` Ornstein-Uhlenbeck Monte Carlo paths for the spread
    of a cointegrated pair, strictly using only price history available up to
    `as_of` (zero look-ahead bias).

    Parameters
    ----------
    ticker_a : str
        Ticker symbol for the long leg of the pair (e.g. 'JPM').
    ticker_b : str
        Ticker symbol for the short leg of the pair (e.g. 'BAC').
    as_of : datetime
        Point-in-time cutoff — only prices on or before this date are used.
    lookback_days : int
        Number of trailing trading days to use for OU parameter fitting.
        Default 252 (one calendar year of trading data).
    horizon_days : int
        Forward simulation horizon in trading days. Default 10.
    num_simulations : int
        Number of Monte Carlo paths. Default 1000.

    Returns
    -------
    dict with keys (JSON-serializable — numpy scalars converted to Python floats):
        probability_of_convergence : float   Fraction of paths that revert to |z| <= 0.5.
        breach_probability         : float   Fraction of paths that hit adverse |z| >= 4.0.
        expected_max_drawdown_bps  : float   Expected peak spread excursion (bps).
        median_final_z             : float   Median z-score at end of horizon.
        half_life_days             : float   OU mean-reversion half-life (trading days).
        horizon_days               : int     Horizon used.
        num_simulations            : int     Paths simulated.
        ticker_a                   : str
        ticker_b                   : str
        as_of                      : str     ISO 8601 cutoff date used.

    On error, returns a dict with a single key:
        ``error`` : str  — Human-readable error description.
    """
    import pandas as pd
    from src.quant.monte_carlo import simulate_spread_paths

    ticker_a = ticker_a.strip().upper()
    ticker_b = ticker_b.strip().upper()

    # Normalise as_of to tz-aware UTC
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    else:
        as_of = as_of.astimezone(timezone.utc)

    cutoff_ts = pd.Timestamp(as_of.date())  # date-level cutoff (no intraday lookahead)

    try:
        prices = _load_prices()
    except Exception as exc:
        logger.error("Failed to load prices_clean.parquet: %s", exc)
        return {"error": f"Price data unavailable: {exc}"}

    # ── Validate tickers are in the price universe ─────────────────────────────
    missing = [t for t in (ticker_a, ticker_b) if t not in prices.columns]
    if missing:
        return {"error": f"Ticker(s) not found in price universe: {missing}"}

    # ── Strict point-in-time slice (no lookahead) ──────────────────────────────
    pit_prices = prices.loc[prices.index <= cutoff_ts]
    if len(pit_prices) == 0:
        return {"error": f"No price data available on or before {cutoff_ts.date()}"}

    # ── Take trailing lookback_days rows ───────────────────────────────────────
    pit_slice = pit_prices.iloc[-lookback_days:]

    prices_a_series = pit_slice[ticker_a].dropna()
    prices_b_series = pit_slice[ticker_b].dropna()

    # ── Try to look up pre-computed hedge ratio from pair_metrics ──────────────
    hedge_ratio: Optional[float] = None
    try:
        pm = pd.read_parquet(_PAIR_METRICS_PATH)
        # Filter for this pair, most recent window up to cutoff
        mask = (
            (
                ((pm["stock_a"] == ticker_a) & (pm["stock_b"] == ticker_b)) |
                ((pm["stock_a"] == ticker_b) & (pm["stock_b"] == ticker_a))
            ) &
            (pd.to_datetime(pm["time"]) <= cutoff_ts)
        )
        pair_rows = pm[mask]
        if not pair_rows.empty:
            latest = pair_rows.sort_values("time").iloc[-1]
            raw_hr = float(latest.get("hedge_ratio", 0.0))
            # If pair is reversed, invert the hedge ratio
            if latest.get("stock_a") == ticker_b:
                raw_hr = 1.0 / raw_hr if abs(raw_hr) > 1e-8 else None
            if raw_hr and abs(raw_hr) > 1e-8:
                hedge_ratio = raw_hr
                logger.debug("Using pair_metrics hedge ratio %.4f for %s/%s", hedge_ratio, ticker_a, ticker_b)
    except Exception as hr_exc:
        logger.debug("Could not retrieve hedge ratio from pair_metrics: %s", hr_exc)

    # ── Run simulation ─────────────────────────────────────────────────────────
    try:
        result = simulate_spread_paths(
            prices_a=prices_a_series,
            prices_b=prices_b_series,
            hedge_ratio=hedge_ratio,
            horizon_days=horizon_days,
            num_simulations=num_simulations,
            z_exit=0.5,
            z_stop=4.0,
            random_seed=42,
        )
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as exc:
        logger.error("Monte Carlo simulation failed for %s/%s: %s", ticker_a, ticker_b, exc)
        return {"error": f"Simulation error: {exc}"}

    # ── Build JSON-serializable output (strip numpy arrays) ────────────────────
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "as_of": cutoff_ts.date().isoformat(),
        "probability_of_convergence": float(result["probability_of_convergence"]),
        "breach_probability": float(result["breach_probability"]),
        "expected_max_drawdown_bps": float(result["expected_max_drawdown_bps"]),
        "median_final_z": float(result["median_final_z"]),
        "half_life_days": float(result["half_life_days"]),
        "theta": float(result["theta"]),
        "mu": float(result["mu"]),
        "sigma_ou": float(result["sigma_ou"]),
        "horizon_days": result["horizon_days"],
        "num_simulations": result["num_simulations"],
        "hedge_ratio_used": round(hedge_ratio, 6) if hedge_ratio is not None else None,
    }
