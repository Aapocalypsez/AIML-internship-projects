"""Feast wiring; calculations stay in engine.py."""

from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from src.config import Config
from src.features.engine import ENGINEERED_FEATURE_NAMES

FEATURE_REFS = [f"transaction_features:{name}" for name in ENGINEERED_FEATURE_NAMES]


def feature_store(cfg: Config) -> FeatureStore:
    store = FeatureStore(repo_path=str((cfg.project_root / cfg.feast_repo_path).resolve()))
    store.config.online_store.connection_string = (
        f"{cfg.redis_host}:{cfg.redis_port},socket_timeout=30,socket_connect_timeout=5"
    )
    return store


def export_features(df: pd.DataFrame, repo_path: Path) -> Path:
    """Write exactly one snapshot per transaction, with original historical time."""
    if df["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique in feature snapshots")
    values = df[ENGINEERED_FEATURE_NAMES].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Engineered feature snapshots must be finite")
    snapshot = values.copy()
    snapshot.insert(0, "TransactionID", df["TransactionID"].astype("int64"))
    snapshot["event_timestamp"] = pd.to_datetime(df["transaction_datetime"], utc=True)
    path = repo_path / "data" / "transaction_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(path, index=False)
    return path


def historical_features(store: FeatureStore, df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve in transaction-ID order, regardless of Feast's output order."""
    entities = df[["TransactionID"]].copy()
    entities["event_timestamp"] = pd.to_datetime(df["transaction_datetime"], utc=True)
    features = store.get_historical_features(entities, features=FEATURE_REFS).to_df()
    features = features.set_index("TransactionID").loc[
        df["TransactionID"], ENGINEERED_FEATURE_NAMES
    ]
    if features.isna().any().any():
        raise ValueError("Historical feature lookup missed a transaction")
    features.index = df.index
    return features


def online_features(store: FeatureStore, df: pd.DataFrame) -> pd.DataFrame:
    """Bound Redis request size for bulk verification through Docker on Windows."""
    chunks = []
    for start in range(0, len(df), 500):
        rows = df[["TransactionID"]].iloc[start : start + 500].to_dict("records")
        chunks.append(store.get_online_features(features=FEATURE_REFS, entity_rows=rows).to_df())
    values = (
        pd.concat(chunks).set_index("TransactionID").loc[df.TransactionID, ENGINEERED_FEATURE_NAMES]
    )
    if values.isna().any().any():
        raise ValueError("Online feature lookup missed a transaction")
    return values
