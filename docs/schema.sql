-- VisionChess PostgreSQL Schema (MVP + extensible)
-- Run via Alembic migrations in production

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Enums ────────────────────────────────────────────

CREATE TYPE scan_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed'
);

CREATE TYPE analysis_status AS ENUM (
    'queued',
    'running',
    'completed',
    'failed'
);

-- ── Users (optional MVP, required for scale) ─────────

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE,
    display_name    VARCHAR(100),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Scans (image → FEN pipeline) ───────────────────────

CREATE TABLE scans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    status          scan_status NOT NULL DEFAULT 'pending',
    image_url       TEXT NOT NULL,
    image_hash      VARCHAR(64),                    -- SHA-256 dedup
    original_filename VARCHAR(255),
    mime_type       VARCHAR(50),
    file_size_bytes INTEGER,

    -- Pipeline outputs
    board_corners   JSONB,                          -- [{x,y}, ...] 4 corners
    warped_image_url TEXT,
    fen             VARCHAR(100),
    fen_confidence  REAL,                           -- 0.0–1.0 aggregate
    piece_map       JSONB,                          -- 8x8 grid of piece labels
    orientation     CHAR(1) CHECK (orientation IN ('w', 'b')),
    active_color    CHAR(1) DEFAULT 'w' CHECK (active_color IN ('w', 'b')),

    -- Metadata
    error_message   TEXT,
    processing_ms   INTEGER,
    model_versions  JSONB,                          -- {board: "v1", pieces: "v2"}
    metadata        JSONB DEFAULT '{}',

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_scans_user_id ON scans(user_id);
CREATE INDEX idx_scans_status ON scans(status);
CREATE INDEX idx_scans_created_at ON scans(created_at DESC);
CREATE INDEX idx_scans_image_hash ON scans(image_hash) WHERE image_hash IS NOT NULL;

-- ── Analyses (Stockfish) ───────────────────────────────

CREATE TABLE analyses (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id         UUID NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    status          analysis_status NOT NULL DEFAULT 'queued',
    fen             VARCHAR(100) NOT NULL,
    depth           INTEGER NOT NULL DEFAULT 18,
    multipv         INTEGER NOT NULL DEFAULT 1,

    -- Engine output
    evaluation_cp     INTEGER,                      -- centipawns (+ = white)
    evaluation_mate   INTEGER,                      -- mate in N (signed)
    best_move         VARCHAR(10),                  -- UCI e.g. "e2e4"
    principal_variation JSONB,                      -- ["e2e4", "e7e5", ...]
    lines             JSONB,                        -- MultiPV full output

    engine_version    VARCHAR(50),
    nodes             BIGINT,
    processing_ms     INTEGER,
    error_message     TEXT,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_analyses_scan_id ON analyses(scan_id);
CREATE INDEX idx_analyses_status ON analyses(status);

-- ── Audit / observability ──────────────────────────────

CREATE TABLE pipeline_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id         UUID NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    stage           VARCHAR(50) NOT NULL,           -- board_detect, warp, piece_detect, fen
    status          VARCHAR(20) NOT NULL,           -- started, completed, failed
    duration_ms     INTEGER,
    details         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_pipeline_events_scan_id ON pipeline_events(scan_id);

-- ── Updated_at trigger ─────────────────────────────────

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_scans_updated_at
    BEFORE UPDATE ON scans FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_analyses_updated_at
    BEFORE UPDATE ON analyses FOR EACH ROW EXECUTE FUNCTION set_updated_at();
