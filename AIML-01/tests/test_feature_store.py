"""Exercise real Feast historical joins and materialization (SQLite; opt-in Redis)."""

import copy
import os
from datetime import timedelta, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from feast import FeatureStore, FileSource

from feature_repo.definitions import transaction, transaction_features
from src.config import load_config
from src.features.engine import ENGINEERED_FEATURE_NAMES, OnlineFeatureEngine, engineer_features
from src.features.store import FEATURE_REFS, export_features, historical_features, online_features
from src.training.data_loader import compute_transaction_datetime
from tests.test_feature_parity import replay_transactions


@pytest.mark.parametrize("backend", ["sqlite", "redis"])
def test_feast_offline_online_replay_parity(tmp_path, backend):
    if backend == "redis" and os.environ.get("FRAUD_TEST_REDIS") != "1":
        pytest.skip("Set FRAUD_TEST_REDIS=1 to test the running Redis service")
    cfg = load_config()
    connection = (
        f'connection_string: "{cfg.redis_host}:{cfg.redis_port},socket_timeout=30"'
        if backend == "redis"
        else 'path: "online.db"'
    )
    (tmp_path / "feature_store.yaml").write_text(
        f"project: fraud_parity_{uuid4().hex[:8]}\nregistry: registry.db\nprovider: local\n"
        f"offline_store:\n  type: file\nonline_store:\n  type: {backend}\n  {connection}\n"
        "entity_key_serialization_version: 3\n"
    )
    raw = compute_transaction_datetime(replay_transactions())
    batch = engineer_features(raw)
    path = export_features(batch, tmp_path)
    view = copy.deepcopy(transaction_features)
    view.batch_source = FileSource(path=str(path), timestamp_field="event_timestamp")
    store = FeatureStore(repo_path=str(tmp_path))
    store.apply([transaction, view])
    historical = historical_features(store, raw)
    np.testing.assert_allclose(historical, batch[ENGINEERED_FEATURE_NAMES], rtol=1e-8, atol=1e-8)
    start = raw.transaction_datetime.min().to_pydatetime().replace(tzinfo=timezone.utc)
    end = raw.transaction_datetime.max().to_pydatetime().replace(tzinfo=timezone.utc)
    store.materialize(start - timedelta(seconds=1), end + timedelta(seconds=1))
    online = online_features(store, raw)
    engine = OnlineFeatureEngine()
    replay = pd.DataFrame([engine.process(row) for row in raw.to_dict("records")])
    np.testing.assert_allclose(online, replay[ENGINEERED_FEATURE_NAMES], rtol=1e-8, atol=1e-8)
    # Feast must not return a snapshot before that snapshot's event timestamp.
    query = raw[["TransactionID"]].iloc[:1].copy()
    query["event_timestamp"] = pd.to_datetime(
        raw.transaction_datetime.iloc[:1], utc=True
    ) - timedelta(seconds=1)
    missing = store.get_historical_features(query, features=FEATURE_REFS).to_df()
    assert missing[ENGINEERED_FEATURE_NAMES].isna().all().all()


def test_export_rejects_duplicate_transactions(tmp_path):
    raw = compute_transaction_datetime(replay_transactions(2))
    raw["TransactionID"] = 1
    with pytest.raises(ValueError, match="unique"):
        export_features(engineer_features(raw), tmp_path)
