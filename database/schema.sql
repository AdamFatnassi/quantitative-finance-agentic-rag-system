-- Canonical fresh-install schema for Quant Finance Agentic RAG.
-- This file includes the final ML registry, regime, and signal tables; there is
-- no separate migration step for a new database. Existing databases need a
-- dedicated upgrade plan before this bootstrap file is applied.

BEGIN;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ─────────────────────────────────────────────
-- TABLE 1: Asset Metadata (connects tickers to SEC CIKs)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS assets (
    asset_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    cik CHAR(10) NOT NULL,                     -- zero-padded 10-digit SEC CIK; links to Vector DB
    sector VARCHAR(50),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    CONSTRAINT cik_format CHECK (cik ~ '^\d{10}$')
);

-- ─────────────────────────────────────────────
-- TABLE 2: Pair Definitions
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pairs (
    pair_id SERIAL PRIMARY KEY,
    asset_a_id INTEGER NOT NULL REFERENCES assets(asset_id),
    asset_b_id INTEGER NOT NULL REFERENCES assets(asset_id),

    -- Enforces a canonical (lower_id, higher_id) ordering so (A,B) and (B,A)
    -- can never both exist as separate pairs, and rules out a self-pair.
    CONSTRAINT pair_order CHECK (asset_a_id < asset_b_id),
    UNIQUE(asset_a_id, asset_b_id)
);

-- Per-run/window screener output, separate from stable pair identity.
CREATE TABLE IF NOT EXISTS pair_candidates (
    window_id INTEGER NOT NULL,
    t0 DATE NOT NULL,
    stock_a_asset_id INTEGER NOT NULL REFERENCES assets(asset_id),
    stock_b_asset_id INTEGER NOT NULL REFERENCES assets(asset_id),
    model_version VARCHAR(150) NOT NULL,
    sector VARCHAR(50),
    ssd DOUBLE PRECISION,
    ssd_rank INTEGER,
    correlation DOUBLE PRECISION,
    hedge_ratio DOUBLE PRECISION,
    intercept DOUBLE PRECISION,
    spread_mean DOUBLE PRECISION,
    spread_std DOUBLE PRECISION,
    adf_pvalue DOUBLE PRECISION,
    coint_pvalue DOUBLE PRECISION,
    coint_qvalue DOUBLE PRECISION,
    half_life DOUBLE PRECISION,
    hurst DOUBLE PRECISION,
    formation_n INTEGER,
    leg_a_adf_pvalue DOUBLE PRECISION,
    leg_b_adf_pvalue DOUBLE PRECISION,
    passes_correlation BOOLEAN,
    passes_cointegration BOOLEAN,
    passes_i1_legs BOOLEAN,
    passes_hedge_bounds BOOLEAN,
    passes_half_life BOOLEAN,
    passes_hurst BOOLEAN,
    is_valid BOOLEAN NOT NULL,

    PRIMARY KEY (
        window_id, t0, stock_a_asset_id, stock_b_asset_id, model_version
    ),
    CONSTRAINT pair_candidates_distinct_legs CHECK (stock_a_asset_id <> stock_b_asset_id)
);

-- ─────────────────────────────────────────────
-- TABLE 3: Raw Market Data (Hypertable)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_data (
    time DATE NOT NULL,                        -- use TIMESTAMPTZ if you add intraday data later
    asset_id INTEGER NOT NULL REFERENCES assets(asset_id),

    -- OHLC (DOUBLE PRECISION for fast math and good TimescaleDB compression)
    open_price DOUBLE PRECISION,
    high_price DOUBLE PRECISION,
    low_price DOUBLE PRECISION,
    close_price DOUBLE PRECISION NOT NULL,     -- adjusted close

    volume BIGINT,

    PRIMARY KEY(time, asset_id),               -- also gives the NOT NULL that UNIQUE didn't

    CONSTRAINT ohlc_bounds CHECK (
        (high_price IS NULL OR low_price IS NULL OR high_price >= low_price)
        AND (volume IS NULL OR volume >= 0)
    )
);
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_market_data_asset_time ON market_data (asset_id, time DESC);

-- Compress chunks older than 30 days; daily OHLCV rarely needs full-res
-- writes past that point, and this is a large storage/query win at scale.
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id'
);
SELECT add_compression_policy('market_data', INTERVAL '30 days', if_not_exists => TRUE);

-- ─────────────────────────────────────────────
-- TABLE 4: Pair Metrics (Features & Base Stats) (Hypertable)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pair_metrics (
    time DATE NOT NULL,
    pair_id INTEGER NOT NULL REFERENCES pairs(pair_id),
    window_id INTEGER,
    stock_a_asset_id INTEGER REFERENCES assets(asset_id),
    stock_b_asset_id INTEGER REFERENCES assets(asset_id),

    -- 1. BASE SPREAD METRICS
    hedge_ratio DOUBLE PRECISION NOT NULL,
    spread DOUBLE PRECISION NOT NULL,
    half_life DOUBLE PRECISION,
    adf_pvalue DOUBLE PRECISION,

    -- 2. KEY ENGINEERED FEATURES
    z_score DOUBLE PRECISION NOT NULL,
    dz_1 DOUBLE PRECISION,
    spread_vol_20 DOUBLE PRECISION,
    spread_skew_60 DOUBLE PRECISION,
    spread_slope_5 DOUBLE PRECISION,
    vol_ratio DOUBLE PRECISION,
    rolling_half_life DOUBLE PRECISION,
    pair_active BOOLEAN,
    coint_qvalue DOUBLE PRECISION,

    lookback_window INTEGER NOT NULL,

    PRIMARY KEY(time, pair_id),

    CONSTRAINT pair_metrics_bounds CHECK (
        (adf_pvalue IS NULL OR adf_pvalue BETWEEN 0 AND 1)
        AND (coint_qvalue IS NULL OR coint_qvalue BETWEEN 0 AND 1)
        AND (half_life IS NULL OR half_life > 0)
        AND (rolling_half_life IS NULL OR rolling_half_life > 0)
        AND (spread_vol_20 IS NULL OR spread_vol_20 >= 0)
        AND (vol_ratio IS NULL OR vol_ratio >= 0)
        AND (lookback_window > 0)
    )
);
SELECT create_hypertable('pair_metrics', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_pair_metrics_pair_time ON pair_metrics (pair_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_pair_metrics_window_pair ON pair_metrics (window_id, pair_id, time DESC);

ALTER TABLE pair_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'pair_id'
);
SELECT add_compression_policy('pair_metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- ─────────────────────────────────────────────
-- TABLE 5: Model registry
-- Shared registry for the HMM regime classifier and mean-reversion signal model.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_models (
    model_id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(150) NOT NULL,
    model_type VARCHAR(80) NOT NULL,
    task VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'trained',
    feature_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluation_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    training_rows BIGINT,
    training_start DATE,
    training_end DATE,
    artifact_uri TEXT,
    source_revision VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT ml_models_name_version_unique UNIQUE (model_name, model_version),
    CONSTRAINT ml_models_status_valid CHECK (
        status IN ('trained', 'validated', 'production', 'archived', 'failed', 'legacy')
    ),
    CONSTRAINT ml_models_training_rows_positive CHECK (training_rows IS NULL OR training_rows > 0),
    CONSTRAINT ml_models_training_dates_valid CHECK (
        training_start IS NULL OR training_end IS NULL OR training_start <= training_end
    )
);

CREATE INDEX IF NOT EXISTS idx_ml_models_name_status
    ON ml_models (model_name, status, created_at DESC);

-- TABLE 6: Machine Learning Regimes (Hypertable)
-- Stores a state and JSONB probabilities for any configured number of HMM states.
CREATE TABLE IF NOT EXISTS ml_regimes (
    time DATE NOT NULL,
    pair_id INTEGER NOT NULL REFERENCES pairs(pair_id),
    model_id BIGINT NOT NULL REFERENCES ml_models(model_id),
    model_version VARCHAR(50) NOT NULL,

    hmm_state INTEGER NOT NULL,                -- most likely state: 0 .. n_states-1
    state_probs JSONB NOT NULL,                -- {"0": 0.86, "1": 0.10, "2": 0.04, ...}

    PRIMARY KEY(time, pair_id, model_id),

    CONSTRAINT hmm_state_valid CHECK (hmm_state >= 0)
    
);
SELECT create_hypertable('ml_regimes', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ml_regimes_pair_time
    ON ml_regimes (pair_id, time DESC, model_version);

ALTER TABLE ml_regimes SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'pair_id'
);
SELECT add_compression_policy('ml_regimes', INTERVAL '30 days', if_not_exists => TRUE);

-- TABLE 7: Mean-reversion model predictions (Hypertable)
CREATE TABLE IF NOT EXISTS ml_signals (
    time DATE NOT NULL,
    pair_id INTEGER NOT NULL REFERENCES pairs(pair_id),
    model_id BIGINT NOT NULL REFERENCES ml_models(model_id),
    window_id INTEGER,
    signal SMALLINT NOT NULL DEFAULT 0,
    probability DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION,
    selected BOOLEAN NOT NULL,
    z_score DOUBLE PRECISION NOT NULL,
    pair_active BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY(time, pair_id, model_id),
    CONSTRAINT ml_signals_direction_valid CHECK (signal BETWEEN -1 AND 1),
    CONSTRAINT ml_signals_probability_valid CHECK (probability BETWEEN 0 AND 1),
    CONSTRAINT ml_signals_threshold_valid CHECK (threshold IS NULL OR threshold BETWEEN 0 AND 1),
    CONSTRAINT ml_signals_selection_consistent CHECK (selected OR signal = 0),
    CONSTRAINT ml_signals_inactive_no_trade CHECK (pair_active OR signal = 0)
);
SELECT create_hypertable('ml_signals', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ml_signals_pair_time
    ON ml_signals (pair_id, time DESC, model_id);
CREATE INDEX IF NOT EXISTS idx_ml_signals_model_time
    ON ml_signals (model_id, time DESC);

ALTER TABLE ml_signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'pair_id,model_id'
);
SELECT add_compression_policy('ml_signals', INTERVAL '30 days', if_not_exists => TRUE);

-- Query-time RAG evidence is keyed by the existing ML signal identity.
-- A NULL chunk_id/rank/score row records the verdict when a leg has no hits.
CREATE TABLE IF NOT EXISTS signal_evidence (
    evidence_id BIGSERIAL PRIMARY KEY,
    signal_time DATE NOT NULL,
    pair_id INTEGER NOT NULL,
    model_id BIGINT NOT NULL,
    leg_ticker VARCHAR(10) NOT NULL REFERENCES assets(ticker),
    chunk_id VARCHAR(300),
    rank INTEGER,
    score DOUBLE PRECISION,
    retrieved_at TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(150) NOT NULL,
    coverage_status VARCHAR(10) NOT NULL DEFAULT 'unknown',
    verdict VARCHAR(4) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    rationale TEXT NOT NULL,

    CONSTRAINT signal_evidence_signal_fk
        FOREIGN KEY (signal_time, pair_id, model_id)
        REFERENCES ml_signals (time, pair_id, model_id)
        ON DELETE CASCADE,
    CONSTRAINT signal_evidence_verdict_valid CHECK (verdict IN ('GO', 'VETO')),
    CONSTRAINT signal_evidence_coverage_valid CHECK (
        coverage_status IN ('full', 'partial', 'none', 'unknown')
    ),
    CONSTRAINT signal_evidence_confidence_valid CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT signal_evidence_chunk_fields_valid CHECK (
        (chunk_id IS NULL AND rank IS NULL AND score IS NULL)
        OR (chunk_id IS NOT NULL AND rank >= 1 AND score IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_evidence_chunk
    ON signal_evidence (signal_time, pair_id, model_id, leg_ticker, chunk_id)
    WHERE chunk_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_evidence_empty_leg
    ON signal_evidence (signal_time, pair_id, model_id, leg_ticker)
    WHERE chunk_id IS NULL;

CREATE TABLE IF NOT EXISTS ingestion_status (
    ticker VARCHAR(10) NOT NULL REFERENCES assets(ticker),
    source VARCHAR(20) NOT NULL,
    earliest_available_at TIMESTAMPTZ,
    latest_available_at TIMESTAMPTZ,
    chunk_count BIGINT NOT NULL DEFAULT 0,
    status VARCHAR(10) NOT NULL DEFAULT 'missing',
    status_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (ticker, source),
    CONSTRAINT ingestion_status_source_valid CHECK (source IN ('sec', 'transcripts')),
    CONSTRAINT ingestion_status_status_valid CHECK (
        status IN ('fetched', 'missing', 'deferred', 'failed', 'disabled')
    ),
    CONSTRAINT ingestion_status_chunk_count_nonnegative CHECK (chunk_count >= 0),
    CONSTRAINT ingestion_status_timestamp_order CHECK (
        earliest_available_at IS NULL OR latest_available_at IS NULL
        OR earliest_available_at <= latest_available_at
    )
);

COMMIT;