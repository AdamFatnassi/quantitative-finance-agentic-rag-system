-- Migration 002: add status column to ingestion_status
-- Idempotent: safe to apply to a fresh schema or an existing deployment.

BEGIN;

ALTER TABLE ingestion_status
    ADD COLUMN IF NOT EXISTS status VARCHAR(10) NOT NULL DEFAULT 'missing';

-- Add CHECK constraint only if it doesn't already exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ingestion_status_status_valid'
          AND conrelid = 'ingestion_status'::regclass
    ) THEN
        ALTER TABLE ingestion_status
            ADD CONSTRAINT ingestion_status_status_valid
            CHECK (status IN ('fetched', 'missing', 'deferred', 'failed', 'disabled'));
    END IF;
END
$$;

COMMIT;
