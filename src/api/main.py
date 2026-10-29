"""main.py

FastAPI backend endpoints for the Agentic Risk Gatekeeper microservice.

Sits between the quantitative engine and execution layer to veto statistical
arbitrage trades driven by fundamental value traps.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from typing import List, Union

from pathlib import Path
from dotenv import load_dotenv

# Load project .env
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=False)

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from src.agent.agent import AgenticRiskGatekeeper
from src.agent.schemas import (
    BatchGatekeeperResponse,
    BatchSignalRequest,
    GatekeeperResponse,
    MonteCarloSimulationRequest,
    MonteCarloSimulationResponse,
    SignalRequest,
)
from src.agent.tools import get_qdrant_client, run_monte_carlo_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("gatekeeper_api")

# Global instances
_qdrant_client: QdrantClient | None = None
_gatekeeper_agent: AgenticRiskGatekeeper | None = None
_concurrency_limit = int(os.getenv("GATEKEEPER_MAX_CONCURRENCY", "10"))
_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    global _qdrant_client, _gatekeeper_agent, _semaphore
    logger.info("Initializing Agentic Risk Gatekeeper microservice...")
    _semaphore = asyncio.Semaphore(_concurrency_limit)
    try:
        _qdrant_client = get_qdrant_client()
        _gatekeeper_agent = AgenticRiskGatekeeper(qdrant_client=_qdrant_client)
        logger.info("Qdrant client and Agentic Risk Gatekeeper successfully initialized.")
    except Exception as exc:
        logger.error("Failed to connect to Qdrant on startup: %s", exc)
        _gatekeeper_agent = AgenticRiskGatekeeper()

    yield

    logger.info("Shutting down Agentic Risk Gatekeeper microservice...")
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None
    _gatekeeper_agent = None


app = FastAPI(
    title="Agentic Risk Gatekeeper API",
    version="1.0.0",
    description="Production-grade Risk Gatekeeper microservice for statistical arbitrage pairs trading.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> AgenticRiskGatekeeper:
    """Dependency / accessor for the Gatekeeper agent."""
    global _gatekeeper_agent
    if _gatekeeper_agent is None:
        _gatekeeper_agent = AgenticRiskGatekeeper()
    return _gatekeeper_agent


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Agentic Risk Gatekeeper",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
def health_check():
    """Verify Qdrant reachability and collection health."""
    agent = get_agent()
    client = agent.qdrant_client
    qdrant_ok = False
    point_count = 0
    collection = agent.collection_name

    try:
        exists = client.collection_exists(collection)
        if exists:
            point_count = client.count(collection, exact=True).count
            qdrant_ok = True
    except Exception as exc:
        logger.error("Health check Qdrant probe failed: %s", exc)

    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "qdrant_connected": qdrant_ok,
        "collection_name": collection,
        "points_indexed": point_count,
        "llm_model": agent.llm_model,
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post(
    "/api/v1/evaluate-signal",
    response_model=GatekeeperResponse,
    status_code=status.HTTP_200_OK,
    tags=["Evaluation"],
    summary="Evaluate single pairs-trading signal for fundamental value trap risk",
)
async def evaluate_signal(signal: SignalRequest) -> GatekeeperResponse:
    """Evaluate an individual spread divergence signal:
    1. Runs fast Qdrant metadata pre-filter (checks for filings within 7 days strictly prior).
    2. If no filings found: returns instant default GO (0 compute cost).
    3. If filings found: retrieves evidence from Qdrant and executes skeptical agentic evaluation.
    """
    agent = get_agent()
    try:
        # Run in thread pool to avoid blocking async event loop
        response = await asyncio.to_thread(agent.evaluate, signal)
        return response
    except Exception as exc:
        logger.error("Unexpected error evaluating signal %s / %s: %s", signal.ticker_a, signal.ticker_b, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gatekeeper evaluation error: {str(exc)}",
        )


@app.post(
    "/api/v1/batch-evaluate",
    response_model=List[GatekeeperResponse],
    status_code=status.HTTP_200_OK,
    tags=["Evaluation"],
    summary="Batch evaluate historical signals for backtest runs with deduplication & caching",
)
async def batch_evaluate(
    signals: Union[List[SignalRequest], BatchSignalRequest] = Body(...),
) -> List[GatekeeperResponse]:
    """Optimized batch evaluation for backtests:
    - In-memory deduplication of identical pair + timestamp requests.
    - Fast pre-filter executes concurrently; signals with no recent filings bypass LLM instantly.
    - Signals requiring LLM analysis are throttled via semaphore to avoid rate limits.
    """
    raw_signals: List[SignalRequest] = (
        signals.signals if isinstance(signals, BatchSignalRequest) else signals
    )

    if not raw_signals:
        return []

    agent = get_agent()
    cache: dict[tuple[str, str, str], GatekeeperResponse] = {}

    async def _evaluate_single(sig: SignalRequest) -> GatekeeperResponse:
        cache_key = (
            sig.ticker_a,
            sig.ticker_b,
            sig.signal_timestamp.isoformat(),
        )
        if cache_key in cache:
            return cache[cache_key]

        async with _semaphore or asyncio.Semaphore(_concurrency_limit):
            res = await asyncio.to_thread(agent.evaluate, sig)
            cache[cache_key] = res
            return res

    tasks = [_evaluate_single(sig) for sig in raw_signals]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_responses: List[GatekeeperResponse] = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error("Error evaluating signal %d: %s", i, res)
            sig = raw_signals[i]
            # Safe fallback response on error
            final_responses.append(
                GatekeeperResponse(
                    decision="GO",
                    confidence_score=0.5,
                    risk_category="none",
                    justification=f"Batch evaluation fallback error: {str(res)}",
                    filing_ids_inspected=[],
                )
            )
        else:
            final_responses.append(res)

    return final_responses


@app.post(
    "/api/v1/simulate-spread",
    response_model=MonteCarloSimulationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Simulation"],
    summary="Run Monte Carlo Ornstein-Uhlenbeck spread stress-test on a cointegrated pair",
)
async def simulate_spread_endpoint(
    req: MonteCarloSimulationRequest,
) -> MonteCarloSimulationResponse:
    """Stress-test a cointegrated spread using Ornstein-Uhlenbeck Monte Carlo paths.
    
    Guarantees strict point-in-time slice (no look-ahead) up to `as_of_timestamp`.
    Returns convergence probability, breach probability, OU half-life, and drawdowns.
    """
    res = await asyncio.to_thread(
        run_monte_carlo_simulation,
        ticker_a=req.ticker_a,
        ticker_b=req.ticker_b,
        as_of=req.as_of_timestamp,
        lookback_days=req.lookback_days,
        horizon_days=req.horizon_days,
        num_simulations=req.num_simulations,
    )

    if "error" in res:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res["error"],
        )

    return MonteCarloSimulationResponse(**res)
