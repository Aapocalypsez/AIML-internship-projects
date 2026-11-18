import os
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import redis
from fastapi.testclient import TestClient

from src.config import load_config
from src.serving.app import create_app
from src.serving.scorer import Scorer


class Model:
    metadata = {
        "feature_columns": ["TransactionAmt", "txn_count_1h"],
        "ring_prior": 0.03,
        "threshold": 0.5,
        "graph_enabled": False,
    }

    def components(self, frame):
        score = np.full(len(frame), 0.2)
        return score, score, score, score

    def explain(self, frame):
        return [{"feature": "TransactionAmt", "log_odds_contribution": 0.1}]


class FakeScorer:
    version = "test"
    redis = type("Ping", (), {"ping": lambda self: True})()

    def score(self, value):
        return {"transaction_id": value["TransactionID"], "score": 0.2}


def transaction(id=1, time=100, amount=10):
    return {
        "TransactionID": id,
        "TransactionDT": time,
        "TransactionAmt": amount,
        "pseudo_user_id": "account-A",
    }


def test_api_contract_health_metrics_and_invalid_transaction():
    with TestClient(create_app(FakeScorer())) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/metrics").status_code == 200
        assert client.post("/score", json=transaction()).json()["score"] == 0.2
        assert client.post("/score", json=transaction(amount=-1)).status_code == 422
        assert client.post("/score", json={}).status_code == 422
        assert client.post("/score", json=transaction(id=10**30)).status_code == 422


def test_model_alias_reload_only_loads_a_changed_version(monkeypatch):
    scorer = Scorer(client=object(), model=Model())
    versions = iter(["1", "1", "2"])
    client = SimpleNamespace(
        get_model_version_by_alias=lambda *args: SimpleNamespace(version=next(versions))
    )
    loaded = []

    def load(uri):
        loaded.append(uri)
        return SimpleNamespace(unwrap_python_model=lambda: Model())

    monkeypatch.setattr("src.serving.scorer.mlflow.MlflowClient", lambda: client)
    monkeypatch.setattr("src.serving.scorer.mlflow.pyfunc.load_model", load)
    assert scorer.reload() == "1"
    assert scorer.reload() == "1"
    assert len(loaded) == 1
    assert scorer.reload() == "2"
    assert len(loaded) == 2


@pytest.mark.skipif(os.getenv("FRAUD_TEST_REDIS") != "1", reason="Requires project Redis")
def test_durable_history_duplicate_conflict_and_rollback():
    cfg = load_config()
    client = redis.Redis(host=cfg.redis_host, port=cfg.redis_port, db=4)
    namespace = "test:" + uuid4().hex
    first = Scorer(client, Model(), namespace)
    try:
        result = first.score(transaction())
        assert result["features"]["txn_count_1h"] == 0
        assert first.score(transaction())["duplicate"]
        with pytest.raises(ValueError, match="already exists"):
            first.score(transaction(amount=1000))
        restored = Scorer(client, Model(), namespace)
        second = restored.score(transaction(2, 200))
        assert second["features"]["txn_count_1h"] == 1
        with pytest.raises(ValueError, match="Out-of-order"):
            restored.score(transaction(3, 50))
        third = restored.score(transaction(4, 300))
        assert third["features"]["txn_count_1h"] == 2
        assert client.get(f"{namespace}:result:3") is None
    finally:
        keys = list(client.scan_iter(namespace + ":*"))
        if keys:
            client.delete(*keys)
