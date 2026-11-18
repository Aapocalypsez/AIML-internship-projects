"""Transaction snapshots: current-amount features must not be keyed only by account."""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String

from src.features.engine import ENGINEERED_FEATURE_NAMES

transaction = Entity(
    name="transaction",
    join_keys=["TransactionID"],
    value_type=ValueType.INT64,
    description="Immutable transaction feature snapshot; live history is computed before update.",
)
transaction_source = FileSource(
    name="transaction_features_source",
    path=str(Path(__file__).resolve().parent / "data" / "transaction_features.parquet"),
    timestamp_field="event_timestamp",
)
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[Field(name=name, dtype=Float64) for name in ENGINEERED_FEATURE_NAMES],
    source=transaction_source,
    online=True,
)

account = Entity(name="account", join_keys=["pseudo_user_id"], value_type=ValueType.STRING)
account_risk_source = FileSource(
    name="account_risk_source",
    path=str(Path(__file__).resolve().parent / "data" / "account_risk.parquet"),
    timestamp_field="event_timestamp",
)
account_ring_risk = FeatureView(
    name="account_ring_risk",
    entities=[account],
    ttl=timedelta(days=365),
    schema=[
        Field(name="ring_score", dtype=Float64),
        Field(name="ring_id", dtype=String),
        Field(name="ring_size", dtype=Int64),
        Field(name="snapshot_transaction_dt", dtype=Float64),
    ],
    source=account_risk_source,
    online=True,
)
