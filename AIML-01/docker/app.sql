CREATE SCHEMA IF NOT EXISTS app;
CREATE TABLE IF NOT EXISTS app.transactions (
    transaction_id BIGINT PRIMARY KEY,
    account_id TEXT NOT NULL,
    event_time DOUBLE PRECISION NOT NULL,
    score DOUBLE PRECISION NOT NULL CHECK(score BETWEEN 0 AND 1),
    model_version TEXT NOT NULL,
    transaction_data JSONB NOT NULL,
    result JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS app.alerts (
    transaction_id BIGINT PRIMARY KEY REFERENCES app.transactions(transaction_id),
    review_reason TEXT NOT NULL CHECK(review_reason IN ('risk','random_audit')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS app.cases (
    transaction_id BIGINT PRIMARY KEY REFERENCES app.transactions(transaction_id),
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','reviewing','closed')),
    analyst_note TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS app.labels (
    transaction_id BIGINT PRIMARY KEY REFERENCES app.transactions(transaction_id),
    label SMALLINT NOT NULL CHECK(label IN (0,1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS app.ring_members (
    account_id TEXT PRIMARY KEY,
    ring_id TEXT NOT NULL,
    ring_score DOUBLE PRECISION NOT NULL,
    ring_size INTEGER NOT NULL,
    snapshot_transaction_dt DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_created_at_idx ON app.alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS transactions_account_idx ON app.transactions(account_id);
CREATE INDEX IF NOT EXISTS rings_id_idx ON app.ring_members(ring_id);
