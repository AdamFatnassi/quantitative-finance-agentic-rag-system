"""agent.py

The Agentic Gatekeeper Core using LangChain and Ollama (gemma3:4b).

Acts as a skeptical forensic risk manager sitting between the quant engine
and the execution layer. Evaluates qualitative SEC filings to veto trades driven
by fundamental value traps.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, List, Optional

from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load project .env
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=False)

from src.agent.prefilter import evaluate_prefilter
from src.agent.schemas import FilingChunk, GatekeeperResponse, SignalRequest
from src.agent.tools import get_qdrant_client, retrieve_pair_filings

logger = logging.getLogger(__name__)

# Model and Provider Configurations
DEFAULT_LLM_PROVIDER = os.getenv("GATEKEEPER_LLM_PROVIDER", "ollama").lower()
DEFAULT_LLM_MODEL = os.getenv("GATEKEEPER_LLM_MODEL", "qwen2.5:1.5b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_FALLBACK_DECISION = os.getenv("GATEKEEPER_ERROR_FALLBACK", "GO").upper()

# Hyperparameters
DEFAULT_TEMPERATURE = float(os.getenv("GATEKEEPER_LLM_TEMPERATURE", "0.0"))
DEFAULT_TOP_P = float(os.getenv("GATEKEEPER_LLM_TOP_P", "0.9"))
DEFAULT_NUM_CTX = int(os.getenv("GATEKEEPER_LLM_NUM_CTX", "2048"))
DEFAULT_MAX_TOKENS = int(os.getenv("GATEKEEPER_LLM_MAX_TOKENS", "512"))

SYSTEM_PROMPT = """You are a rigorous, highly skeptical forensic risk manager for a quantitative statistical arbitrage (pairs trading) hedge fund. Your default stance is skepticism.

OBJECTIVE:
You act as a pre-trade risk filter. The quantitative engine has triggered a mean-reverting spread divergence signal for a pair of equities.
Statistical arbitrage assumes the divergence is transitory market noise that will mean-revert.
Your mission is to VETO any trade where the divergence is driven by a fundamental structural shock or value trap.

EVALUATION RULES:
1. Carefully inspect the provided SEC filings (8-K, 10-K, 10-Q) and earnings transcripts available strictly up to the signal timestamp.
2. Issue a VETO if either ticker exhibits any of the following fundamental shocks:
   - 'financial_distress': Restatements (Item 4.02), debt covenant violations, bankruptcy/receivership (Item 1.03), going-concern warnings, severe credit downgrade.
   - 'legal_regulatory': SEC/DOJ subpoenas, formal regulatory investigations, catastrophic litigation verdicts, criminal charges.
   - 'executive_governance': Sudden unannounced CEO/CFO departures (Item 5.02), board mutiny, executive embezzlement or accounting malfeasance.
   - 'operational': Catastrophic product defects/groundings/recalls, critical customer contract termination (Item 1.02), major M&A re-pricing (Item 2.01), plant closures.
3. If there is ambiguous negative news or unexpected fundamental shocks near the signal date, YOUR DEFAULT STANCE IS SKEPTICISM: ISSUE A VETO.
4. Issue a GO decision ONLY if the filings reflect routine operations, predictable noise, or benign events that will mean-revert without breaking the fundamental relationship.
5. In your justification, cite the exact chunk_ids or accession_numbers and item codes inspected. Return risk_category='none' if and only if decision is GO.

TOOL USAGE:
You have access to two primary analytical tools:
1. `search_sec_filings` (qualitative): Retrieve SEC 8-K/10-K/10-Q filings and earnings transcripts from the Qdrant vector store. Use to identify fundamental risk events.
2. `run_monte_carlo_simulation` (quantitative): Stress-test the cointegrated spread using 1,000 Ornstein-Uhlenbeck simulated paths over a 10-day horizon.

QUANTITATIVE DECISION ENHANCEMENT RULE:
- If SEC disclosures reveal AMBIGUOUS risk (not clearly catastrophic), trigger `run_monte_carlo_simulation` to mathematically evaluate spread reversion likelihood.
- Issue a VETO on QUANTITATIVE grounds if:
    * convergence_probability < 0.50  (majority of simulated paths fail to revert to mean), OR
    * breach_probability > 0.25       (more than 1-in-4 paths hit the adverse stop threshold).
- Issue a GO if simulation confirms strong reversion: convergence_probability >= 0.70 AND breach_probability < 0.15. This may override minor filing ambiguity.
- When simulation is run, always set simulation_performed=True and populate convergence_probability, breach_probability, and simulation_notes in your response.
- simulation_notes must summarise: half-life in trading days, convergence probability, breach probability, and your quantitative decision basis."""


class AgenticRiskGatekeeper:
    """Production Agentic Risk Gatekeeper microservice component using LangChain."""

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        llm_provider: str = DEFAULT_LLM_PROVIDER,
        llm_model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        num_ctx: int = DEFAULT_NUM_CTX,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        collection_name: Optional[str] = None,
        lookback_days: int = 7,
    ):
        self.qdrant_client = qdrant_client or get_qdrant_client()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.collection_name = collection_name or os.getenv("RAG_QDRANT_COLLECTION", "financial_text_chunks")
        self.lookback_days = lookback_days
        self._chain = None

    def _build_langchain_chain(self):
        """Build and configure the LangChain structured output chain."""
        if self._chain is not None:
            return self._chain

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{user_content}"),
        ])

        if self.llm_provider == "ollama":
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=self.llm_model,
                base_url=self.base_url,
                temperature=self.temperature,
                top_p=self.top_p,
                num_ctx=self.num_ctx,
                num_predict=self.max_tokens,
            )
            structured_llm = llm.with_structured_output(GatekeeperResponse)
            self._chain = prompt | structured_llm
            return self._chain

        elif self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY is not set for openai provider.")
                return None

            llm = ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,
            )
            structured_llm = llm.with_structured_output(GatekeeperResponse)
            self._chain = prompt | structured_llm
            return self._chain

        else:
            logger.error("Unsupported LLM provider: %s", self.llm_provider)
            return None

    def _rule_based_forensic_fallback(
        self,
        signal: SignalRequest,
        evidence: dict[str, List[FilingChunk]],
        reason: str,
    ) -> GatekeeperResponse:
        """Heuristic safety scanner for when LLM inference is unavailable or fails."""
        logger.warning("Executing forensic fallback for %s / %s due to: %s", signal.ticker_a, signal.ticker_b, reason)

        inspected_ids: List[str] = []
        toxic_triggers: List[str] = []
        risk_category = "none"

        for ticker, chunks in evidence.items():
            for c in chunks:
                inspected_ids.append(c.chunk_id)
                text_lower = c.text.lower()

                # Item codes
                if c.item_code in {"4.02", "1.03"}:
                    toxic_triggers.append(f"{ticker} Item {c.item_code}")
                    risk_category = "financial_distress"
                elif c.item_code == "5.02":
                    if any(w in text_lower for w in ["resign", "terminat", "depart", "investigat"]):
                        toxic_triggers.append(f"{ticker} Item 5.02 executive departure")
                        risk_category = "executive_governance"
                elif c.item_code in {"1.01", "1.02", "2.01"}:
                    if any(w in text_lower for w in ["terminat", "default", "breach"]):
                        toxic_triggers.append(f"{ticker} Item {c.item_code} material contract issue")
                        risk_category = "operational"

                # Keywords
                if "restatement" in text_lower or "material weakness" in text_lower:
                    toxic_triggers.append(f"{ticker} accounting restatement/material weakness")
                    risk_category = "financial_distress"
                elif "subpoena" in text_lower or "formal investigation" in text_lower:
                    toxic_triggers.append(f"{ticker} regulatory subpoena/investigation")
                    risk_category = "legal_regulatory"

        if toxic_triggers:
            justification = (
                f"Forensic Rule Scanner VETO ({reason}): Detected high-risk fundamental items in filings "
                f"prior to signal: {', '.join(toxic_triggers[:3])}."
            )
            return GatekeeperResponse(
                decision="VETO",
                confidence_score=0.85,
                risk_category=risk_category if risk_category != "none" else "operational",
                justification=justification,
                filing_ids_inspected=inspected_ids[:10],
            )

        if DEFAULT_FALLBACK_DECISION == "VETO":
            return GatekeeperResponse(
                decision="VETO",
                confidence_score=0.5,
                risk_category="operational",
                justification=f"Skeptical Gatekeeper Fallback ({reason}): Unable to complete LLM analysis on recent filings. Defaulting to safe VETO.",
                filing_ids_inspected=inspected_ids[:10],
            )

        return GatekeeperResponse(
            decision="GO",
            confidence_score=0.75,
            risk_category="none",
            justification=f"Forensic Rule Scanner GO ({reason}): Inspected {len(inspected_ids)} filing chunks. No critical items (4.02, 1.03, subpoenas, restatements) detected.",
            filing_ids_inspected=inspected_ids[:10],
        )

    def evaluate(self, signal: SignalRequest) -> GatekeeperResponse:
        """Full pipeline: Pre-filter optimization -> Qdrant retrieval -> LangChain Ollama (gemma3:4b) adjudication."""
        # 1. Fast Pre-Filter Optimization
        try:
            bypassed, prefilter_resp = evaluate_prefilter(
                client=self.qdrant_client,
                signal=signal,
                collection_name=self.collection_name,
                lookback_days=self.lookback_days,
            )
            if bypassed and prefilter_resp is not None:
                return prefilter_resp
        except Exception as exc:
            logger.error("Pre-filter execution error for %s / %s: %s", signal.ticker_a, signal.ticker_b, exc)

        # 2. Vector & Relational Evidence Retrieval
        try:
            evidence = retrieve_pair_filings(
                ticker_a=signal.ticker_a,
                ticker_b=signal.ticker_b,
                as_of_timestamp=signal.signal_timestamp,
                query="material definitive agreement restatement departure lawsuit bankruptcy earnings risk",
                limit_per_leg=5,
                client=self.qdrant_client,
                collection_name=self.collection_name,
            )
        except Exception as exc:
            logger.error("Failed to retrieve filings from Qdrant: %s", exc)
            return self._rule_based_forensic_fallback(signal, {}, reason=f"Qdrant connection failure: {exc}")

        all_chunks: List[FilingChunk] = evidence[signal.ticker_a] + evidence[signal.ticker_b]
        inspected_ids = [c.chunk_id for c in all_chunks]

        if not all_chunks:
            # No chunks returned despite prefilter count (e.g. text filtering)
            return GatekeeperResponse(
                decision="GO",
                confidence_score=0.9,
                risk_category="none",
                justification=f"No material risk evidence retrieved for {signal.ticker_a} or {signal.ticker_b} strictly before signal.",
                filing_ids_inspected=[],
            )

        # 3. LangChain LLM Evaluation
        try:
            chain = self._build_langchain_chain()
        except Exception as build_exc:
            return self._rule_based_forensic_fallback(
                signal, evidence, reason=f"Failed to build LangChain chain ({self.llm_provider}): {build_exc}"
            )

        if chain is None:
            return self._rule_based_forensic_fallback(
                signal, evidence, reason=f"LLM chain could not be created for provider={self.llm_provider}"
            )

        # Build prompt payload
        evidence_summary = []
        for c in all_chunks:
            evidence_summary.append({
                "chunk_id": c.chunk_id,
                "ticker": c.ticker,
                "form_type": c.form_type,
                "item_code": c.item_code,
                "section_type": c.section_type,
                "available_at": c.available_at,
                "text_snippet": c.text[:600],
            })

        user_content = (
            f"SIGNAL CONTEXT:\n"
            f"- Pair: {signal.ticker_a} / {signal.ticker_b}\n"
            f"- Signal Timestamp: {signal.signal_timestamp.isoformat()}\n"
            f"- Spread Divergence Z-Score: {signal.z_score}\n\n"
            f"QUALITATIVE FILING EVIDENCE (Point-in-time <= {signal.signal_timestamp.isoformat()}):\n"
            f"{json.dumps(evidence_summary, indent=2)}"
        )

        try:
            raw_response = chain.invoke({"user_content": user_content})

            if isinstance(raw_response, GatekeeperResponse):
                response = raw_response
            elif isinstance(raw_response, dict):
                response = GatekeeperResponse.model_validate(raw_response)
            else:
                response = GatekeeperResponse.model_validate(raw_response)

            # Ensure filing_ids_inspected includes all retrieved IDs if model omitted them
            if not response.filing_ids_inspected:
                response.filing_ids_inspected = inspected_ids[:10]

            logger.info(
                "Gatekeeper decision for %s / %s via %s (%s): %s (confidence=%.2f, risk=%s)",
                signal.ticker_a,
                signal.ticker_b,
                self.llm_provider,
                self.llm_model,
                response.decision,
                response.confidence_score,
                response.risk_category,
            )
            return response

        except Exception as exc:
            logger.error("LangChain evaluation error for signal %s / %s: %s", signal.ticker_a, signal.ticker_b, exc)
            return self._rule_based_forensic_fallback(
                signal, evidence, reason=f"LLM API exception or malformed output: {exc}"
            )
