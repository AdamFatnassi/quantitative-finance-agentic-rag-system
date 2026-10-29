BEGIN;

ALTER TABLE signal_evidence
    ADD COLUMN IF NOT EXISTS coverage_status VARCHAR(10) NOT NULL DEFAULT 'unknown';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'signal_evidence_coverage_valid'
          AND conrelid = 'signal_evidence'::regclass
    ) THEN
        ALTER TABLE signal_evidence
            ADD CONSTRAINT signal_evidence_coverage_valid
            CHECK (coverage_status IN ('full', 'partial', 'none', 'unknown'));
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS ingestion_status (
    ticker VARCHAR(10) NOT NULL REFERENCES assets(ticker),
    source VARCHAR(20) NOT NULL,
    earliest_available_at TIMESTAMPTZ,
    latest_available_at TIMESTAMPTZ,
    chunk_count BIGINT NOT NULL DEFAULT 0,
    status_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (ticker, source),
    CONSTRAINT ingestion_status_source_valid CHECK (source IN ('sec', 'transcripts')),
    CONSTRAINT ingestion_status_chunk_count_nonnegative CHECK (chunk_count >= 0),
    CONSTRAINT ingestion_status_timestamp_order CHECK (
        earliest_available_at IS NULL OR latest_available_at IS NULL
        OR earliest_available_at <= latest_available_at
    )
);

COMMIT;
