# 🛡️ Quantitative Finance Agentic RAG System

### Statistical Arbitrage with an LLM Forensic Risk Gatekeeper & Monte Carlo Spread Stress-Testing

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C.svg)](https://www.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Vector_Store-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-PostgreSQL_17-F58220.svg?logo=postgresql&logoColor=white)](https://www.timescale.com/)
[![Tests Passing](https://img.shields.io/badge/tests-34%2F34%20passed-brightgreen.svg)](#-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

Pairs trading assumes that a spread divergence between two cointegrated stocks is **transitory liquidity noise** that will mean-revert, following an Ornstein-Uhlenbeck (OU) process.

That assumption breaks when the divergence is caused by a **structural fundamental shock**: accounting fraud, an SEC subpoena, a failed merger, a sudden executive departure, a debt covenant breach. The cointegration relationship is then permanently broken, and a naive model keeps averaging down into a **value trap**.

This project places an **Agentic Risk Gatekeeper** between the quantitative signal engine and execution:

| Stage | What it does |
|:---|:---|
| **Point-in-Time (PIT) RAG** | Searches SEC 8-K / 10-K filings and earnings transcripts available *strictly before* the signal timestamp, with zero look-ahead bias. |
| **0 ms metadata pre-filter** | Clears routine pairs instantly when no recent filing exists, which removes 80–90% of the LLM workload. |
| **Forensic LLM adjudicator** | Classifies disclosures into `financial_distress`, `legal_regulatory`, `executive_governance`, `operational` using structured JSON output. |
| **OU Monte Carlo stress-test** | Simulates 1,000 forward spread paths to estimate reversion probability and stop-loss breach probability when the text is ambiguous. |

---

## 🏗️ System Architecture

Data flows from two sources (prices and regulatory text) into a storage layer, then through the quant engine, the risk gatekeeper, and finally the serving layer.

<p align="center">
  <img src="img/quant1.svg" alt="End-to-end system architecture: data sources, storage layer, quant engine, agentic risk gatekeeper, serving layer" width="720">
</p>

<p align="center"><em>Figure 1 — End-to-end architecture</em></p>

---

## ⚡ Gatekeeper Decision Flow

Every signal takes one of two paths. The **fast path** (no filing in the last 7 days) returns GO immediately. The **agent path** wakes the LLM, and only escalates to Monte Carlo when the text is ambiguous.

<p align="center">
  <img src="img/quant2.svg" alt="Gatekeeper decision flow: 7-day pre-filter, LLM text evaluation, Monte Carlo breach probability check, GO or VETO" width="600">
</p>

<p align="center"><em>Figure 2 — Risk gatekeeper decision logic</em></p>

**Outcomes**

- **GO**: no recent filing, clear safety in the text, or a low simulated breach probability.
- **VETO**: clear red flag in the text (for example SEC Item 4.02 restatement or Item 1.03 bankruptcy), or a high simulated breach probability.

---

## 🔬 Core Components

### 1. Quantitative Pairs Engine & ML Meta-Labeling

- **Cointegration screening** of S&P 500 pairs with Engle-Granger and Johansen tests over historical lookback windows.
- **Dynamic spread**:

  $$\text{Spread}_t = \ln(P_{A,t}) - \beta \cdot \ln(P_{B,t})$$

  where $\beta$ is the hedge ratio from rolling OLS or a Kalman filter.
- **Meta-labeling**: XGBoost / LightGBM classifiers trained on spread momentum, volatility ratios, rolling half-life and Hidden Markov Model regime probabilities, to filter false breakouts.

### 2. Point-in-Time Agentic RAG Gatekeeper

- **Strict PIT filtering**: retrieved filings must satisfy $\text{available\_at} \le t_{\text{signal}}$.
- **Hybrid search with Reciprocal Rank Fusion (RRF)**: dense vectors (`BAAI/bge-small-en-v1.5`) fused with sparse BM25 tokens (`Qdrant/bm25`).
- **Forensic risk taxonomy**:
  - `financial_distress`: Item 4.02 non-reliance on financials, Item 1.03 bankruptcy, going-concern warnings.
  - `legal_regulatory`: SEC/DOJ subpoenas, antitrust litigation, criminal investigations.
  - `executive_governance`: Item 5.02 unannounced CEO/CFO departures, board conflict, executive fraud.
  - `operational`: Item 2.01 failed M&A, Item 1.02 loss of a major customer.

### 3. Ornstein-Uhlenbeck Monte Carlo Engine

Implemented in [`src/quant/monte_carlo.py`](src/quant/monte_carlo.py) using the exact OU discretization:

$$S_{t+1} = S_t e^{-\theta \Delta t} + \mu(1 - e^{-\theta \Delta t}) + \sigma_{OU}\sqrt{\frac{1-e^{-2\theta\Delta t}}{2\theta}}\, Z_t, \quad Z_t \sim \mathcal{N}(0, 1)$$

- $\theta$, $\mu$ and $\sigma_{OU}$ are estimated by AR(1) OLS on point-in-time spread history.
- Half-life: $T_{1/2} = \ln(2) / \theta$.
- Simulates $N = 1{,}000$ paths over a 10-day horizon and reports:
  - **Probability of convergence**: share of paths reaching $|z| \le 0.5$.
  - **Breach probability**: share of paths reaching $|z| \ge 4.0$ (stop-loss).
  - **Expected max drawdown**: peak spread excursion in basis points.

---

## 📊 Backtest Results

Out-of-sample ML-selected trades on S&P 500 pairs, with and without the gatekeeper:

| Metric | ML filter only | ML filter + Gatekeeper | Delta |
|:---|:---:|:---:|:---:|
| **Sharpe ratio** | 1.14 | **1.78** | +0.64 (+56.1%) |
| **Win rate** | 62.4% | **71.2%** | +8.8 pts |
| **Catastrophic breakdowns blocked** | 0 / 25 | **18 / 25** | 72.0% |
| **Stop-loss trades vetoed** | n/a | **42** | |
| **Cumulative capital saved** | 0 bps | **+1,420 bps** | |
| **Average vetoed trade P&L** | n/a | **-24.8 bps** | Losses avoided |

> **Takeaway:** the gatekeeper accepts a small false-veto rate (about 3.8% of profitable trades) in exchange for blocking 72% of structural breakdowns, which are the trades that cause multi-sigma drawdowns.

---

## 💻 Streamlit Dashboard

```
app/
├── Home.py                  # Overview, architecture, key metrics
└── pages/
    ├── 1_ML_Performance.py  # PR/ROC curves, confusion matrix, feature importance
    ├── 2_Backtest.py        # Event-driven backtest, parameter sensitivity, equity curves
    ├── 3_Pairs_Regime.py    # Live cointegration tests, spread tracking, rolling half-life
    └── 4_Live_Gatekeeper.py # Single-signal inspector, batch review, backtest arena
```

**`4_Live_Gatekeeper.py` highlights**

- **Single signal inspection**: real-time RAG verdict, KPI cards, and a Plotly Monte Carlo fan chart (P10 / P25 / P50 / P75 / P90).
- **Batch evaluation**: concurrent review of signal baskets with live `Conv. %` and `Breach %` diagnostics.
- **Backtest arena**: out-of-sample ML trades run through the gatekeeper, with a forensic dossier inspector that cites the SEC disclosures behind each decision.

---

## 🔌 REST API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check for TimescaleDB and Qdrant |
| `POST` | `/api/v1/evaluate-signal` | Evaluate one pair divergence signal for fundamental value traps |
| `POST` | `/api/v1/batch-evaluate` | Concurrent batch evaluation with in-memory deduplication and caching |
| `POST` | `/api/v1/simulate-spread` | On-demand 1,000-path OU Monte Carlo simulation |

<details>
<summary><b>Example: <code>POST /api/v1/simulate-spread</code></b></summary>

**Request**

```json
{
  "ticker_a": "JPM",
  "ticker_b": "BAC",
  "as_of_timestamp": "2024-01-15T16:00:00Z",
  "lookback_days": 252,
  "horizon_days": 10,
  "num_simulations": 1000
}
```

**Response**

```json
{
  "ticker_a": "JPM",
  "ticker_b": "BAC",
  "as_of": "2024-01-15",
  "probability_of_convergence": 0.884,
  "breach_probability": 0.012,
  "expected_max_drawdown_bps": 28.4,
  "median_final_z": 0.15,
  "half_life_days": 6.8,
  "theta": 0.1019,
  "mu": -0.0032,
  "sigma_ou": 0.0142,
  "horizon_days": 10,
  "num_simulations": 1000,
  "hedge_ratio_used": 0.7241
}
```

</details>

---

## 🧪 Tests

The suite covers point-in-time constraints, schema validation, Monte Carlo maths and API endpoints.

```bash
pytest tests/ -v
```

| File | Tests |
|:---|:---:|
| `test_backtest_coverage.py` | 10 |
| `test_agentic_gatekeeper.py` | 9 |
| `test_monte_carlo.py` | 9 |
| `test_gatekeeper_api.py` | 6 |
| **Total** | **34 passed** |

---

## 🚀 Quickstart

**Prerequisites:** Python 3.10+ (tested on 3.12), Docker & Docker Compose, and optionally [Ollama](https://ollama.com) for local inference (`ollama pull qwen2.5:1.5b`).

```bash
# 1. Clone
git clone https://github.com/your-username/quantitative-finance-agentic-rag-system.git
cd quantitative-finance-agentic-rag-system

# 2. Configure (add your PostgreSQL credentials, API keys, model settings)
cp .env.example .env

# 3. Environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 4. Infrastructure (TimescaleDB + Qdrant)
docker compose up -d
```

Then run the two services in separate terminals:

```bash
# Terminal 1 — API
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Dashboard (opens at http://localhost:8501)
streamlit run app/Home.py
```

---

## 📂 Repository Layout

```
.
├── .env.example                     # Credentials & endpoints template
├── docker-compose.yaml              # TimescaleDB & Qdrant
├── requirements.txt                 # Pinned dependencies
├── img/
│   ├── quant1.svg                   # Figure 1: system architecture
│   └── quant2.svg                   # Figure 2: gatekeeper decision flow
├── app/                             # Streamlit application
│   ├── Home.py
│   ├── theme.py                     # Dark theme, metrics, Plotly styling
│   └── pages/
│       ├── 1_ML_Performance.py
│       ├── 2_Backtest.py
│       ├── 3_Pairs_Regime.py
│       └── 4_Live_Gatekeeper.py
├── src/
│   ├── agent/                       # Agentic risk gatekeeper
│   │   ├── agent.py                 # Skeptical forensic risk manager (LangChain)
│   │   ├── prefilter.py             # 0 ms Qdrant metadata pre-filter
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── tools.py                 # Hybrid retrieval & Monte Carlo tool
│   ├── api/
│   │   └── main.py                  # FastAPI endpoints
│   ├── quant/
│   │   ├── backtest.py              # Event-driven pairs backtester
│   │   ├── monte_carlo.py           # OU spread simulator
│   │   └── pairs_selection.py       # Cointegration screening & Kalman filter
│   └── rag/
│       ├── scraping/                # SEC EDGAR & transcript scrapers
│       └── qdrant/                  # FastEmbed chunking & ingestion
└── tests/                           # 34 pytest tests
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.