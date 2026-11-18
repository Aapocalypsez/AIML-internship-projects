-- Postgres init script: create schemas for MLflow and the fraud app
-- MLflow will create its own tables, but we need the fraud app tables later.

-- Schema for fraud detection app tables (alerts, cases, labels)
CREATE SCHEMA IF NOT EXISTS app;

-- Placeholder table to verify init ran
CREATE TABLE IF NOT EXISTS app.health_check (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO app.health_check DEFAULT VALUES;
