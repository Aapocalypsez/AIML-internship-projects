"""Shared configuration loader.

Reads config.yaml and provides typed access to all settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    """Immutable project configuration."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Seeds
    seed: int = 42

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_raw: str = "transactions.raw"
    kafka_topic_scored: str = "transactions.scored"
    kafka_topic_alerts: str = "alerts"
    kafka_topic_dlq: str = "transactions.dlq"

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6380

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "fraud"
    postgres_password: str = "fraud_dev"
    postgres_db: str = "fraud_detection"

    # Feast
    feast_repo_path: str = "./feature_repo"

    # Model
    blend_weight_xgboost: float = 0.7
    blend_weight_gnn: float = 0.3

    # SLA
    scoring_sla_p99_ms: int = 100
    scoring_sla_p95_ms: int = 50

    # Training
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    n_tuning_trials: int = 20

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def load_config(config_path: Path | None = None) -> Config:
    """Load config from YAML, falling back to defaults."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        return Config()

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    # Container deployments override the same YAML keys through environment variables.
    for key, value in raw.items():
        if key in os.environ:
            raw[key] = (
                type(value)(os.environ[key]) if isinstance(value, (int, float)) else os.environ[key]
            )

    return Config(
        seed=raw.get("SEED", 42),
        mlflow_tracking_uri=raw.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        kafka_bootstrap_servers=raw.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        kafka_topic_raw=raw.get("KAFKA_TOPIC_RAW", "transactions.raw"),
        kafka_topic_scored=raw.get("KAFKA_TOPIC_SCORED", "transactions.scored"),
        kafka_topic_alerts=raw.get("KAFKA_TOPIC_ALERTS", "alerts"),
        kafka_topic_dlq=raw.get("KAFKA_TOPIC_DLQ", "transactions.dlq"),
        redis_host=raw.get("REDIS_HOST", "127.0.0.1"),
        redis_port=raw.get("REDIS_PORT", 6380),
        postgres_host=raw.get("POSTGRES_HOST", "localhost"),
        postgres_port=raw.get("POSTGRES_PORT", 5433),
        postgres_user=raw.get("POSTGRES_USER", "fraud"),
        postgres_password=raw.get("POSTGRES_PASSWORD", "fraud_dev"),
        postgres_db=raw.get("POSTGRES_DB", "fraud_detection"),
        feast_repo_path=raw.get("FEAST_REPO_PATH", "./feature_repo"),
        blend_weight_xgboost=raw.get("MODEL_BLEND_WEIGHT_XGBOOST", 0.7),
        blend_weight_gnn=raw.get("MODEL_BLEND_WEIGHT_GNN", 0.3),
        scoring_sla_p99_ms=raw.get("SCORING_SLA_P99_MS", 100),
        scoring_sla_p95_ms=raw.get("SCORING_SLA_P95_MS", 50),
        train_ratio=raw.get("TRAIN_RATIO", 0.6),
        val_ratio=raw.get("VAL_RATIO", 0.2),
        test_ratio=raw.get("TEST_RATIO", 0.2),
        n_tuning_trials=raw.get("N_TUNING_TRIALS", 20),
    )
