"""Redis WATCH makes state updates and duplicate-result caching atomic."""

import hashlib
import json
import os
import threading
import time

import mlflow
import numpy as np
import pandas as pd
import redis
from pydantic import BaseModel, ConfigDict, Field

from src.config import load_config
from src.features.engine import OnlineFeatureEngine
from src.training.registry import MODEL_NAME


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    TransactionID: int = Field(ge=0, le=2**63 - 1, strict=True)
    TransactionDT: float = Field(ge=0)
    TransactionAmt: float = Field(ge=0)
    pseudo_user_id: str = Field(min_length=1, max_length=200)
    DeviceType: str | None = None
    P_emaildomain: str | None = None


class Scorer:
    def __init__(self, client=None, model=None, namespace=None):
        self.cfg = load_config()
        self.redis = client or redis.Redis(
            host=self.cfg.redis_host,
            port=self.cfg.redis_port,
            socket_timeout=5,
            socket_connect_timeout=3,
        )
        self.namespace = namespace or os.getenv("STATE_NAMESPACE", "fraud:live")
        self.model_name = os.getenv("MODEL_NAME", MODEL_NAME)
        self.model = model
        self.version = "test" if model else None
        self.lock = threading.RLock()
        self.last_reload = 0
        self.store = None
        if model is None:
            mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
            self.reload()

    def reload(self):
        with self.lock:
            version = mlflow.MlflowClient().get_model_version_by_alias(self.model_name, "champion")
            if str(version.version) != self.version:
                model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@champion")
                self.model = model.unwrap_python_model()
                self.version = str(version.version)
            self.last_reload = time.monotonic()
        return self.version

    def ring_features(self, transaction):
        if not self.model.metadata.get("graph_enabled") or transaction[
            "TransactionDT"
        ] <= self.model.metadata.get("graph_snapshot_dt", -1):
            return {
                "ring_score": self.model.metadata["ring_prior"],
                "ring_id": None,
                "ring_size": 0,
            }
        if self.store is None:
            from src.features.store import feature_store

            self.store = feature_store(self.cfg)
        values = self.store.get_online_features(
            features=[
                f"account_ring_risk:{name}"
                for name in ["ring_score", "ring_id", "ring_size", "snapshot_transaction_dt"]
            ],
            entity_rows=[{"pseudo_user_id": transaction["pseudo_user_id"]}],
        ).to_dict()
        snapshot = values.get("snapshot_transaction_dt", [None])[0]
        if snapshot is None or snapshot >= transaction["TransactionDT"]:
            return {
                "ring_score": self.model.metadata["ring_prior"],
                "ring_id": None,
                "ring_size": 0,
            }
        return {name: values[name][0] for name in ["ring_score", "ring_id", "ring_size"]}

    def score(self, payload):
        transaction = Transaction.model_validate(payload).model_dump()
        encoded = json.dumps(transaction, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        account = transaction["pseudo_user_id"]
        state_key = f"{self.namespace}:account:{account}"
        result_key = f"{self.namespace}:result:{transaction['TransactionID']}"
        started = time.perf_counter()
        with self.lock:
            if self.version != "test" and time.monotonic() - self.last_reload > 30:
                self.reload()
            for _ in range(8):
                with self.redis.pipeline() as pipe:
                    try:
                        pipe.watch(state_key, result_key)
                        cached = pipe.get(result_key)
                        if cached:
                            stored = json.loads(cached)
                            if stored["fingerprint"] != fingerprint:
                                raise ValueError("TransactionID already exists with different data")
                            return {**stored["result"], "duplicate": True}
                        engine = OnlineFeatureEngine()
                        state = pipe.get(state_key)
                        if state:
                            engine.restore_state(account, state)
                        features = engine.process(transaction)
                        columns = self.model.metadata["feature_columns"]
                        values = {name: transaction.get(name, np.nan) for name in columns}
                        values.update(features)
                        frame = pd.DataFrame([values], columns=columns).astype("float32")
                        if np.isinf(frame.to_numpy()).any():
                            raise ValueError("Numeric feature exceeds finite float32 range")
                        ring = self.ring_features(transaction)
                        frame["ring_score"] = ring["ring_score"]
                        raw, xgb_score, ring_score, score = self.model.components(frame)
                        score = float(score[0])
                        audit = (
                            int(
                                hashlib.sha256(
                                    str(transaction["TransactionID"]).encode()
                                ).hexdigest()[:8],
                                16,
                            )
                            / 2**32
                            < 0.01
                        )
                        result = {
                            "transaction_id": transaction["TransactionID"],
                            "account_id": account,
                            "event_time": transaction["TransactionDT"],
                            "score": score,
                            "xgb_score": float(xgb_score[0]),
                            "ring_score": float(ring_score[0]),
                            "raw_xgb_score": float(raw[0]),
                            "ring_id": ring["ring_id"],
                            "ring_size": ring["ring_size"],
                            "model_version": self.version,
                            "threshold": self.model.metadata["threshold"],
                            "is_alert": score >= self.model.metadata["threshold"],
                            "is_audit": audit,
                            "duplicate": False,
                            "reasons": [],
                            "features": features,
                            "transaction": transaction,
                            "latency_ms": (time.perf_counter() - started) * 1000,
                            "scored_at": time.time(),
                        }
                        pipe.multi()
                        pipe.set(state_key, engine.export_state(account))
                        pipe.set(
                            result_key,
                            json.dumps(
                                {"fingerprint": fingerprint, "result": result}, allow_nan=False
                            ),
                        )
                        pipe.execute()
                        result["latency_ms"] = (time.perf_counter() - started) * 1000
                        return result
                    except redis.WatchError:
                        continue
        raise RuntimeError("Concurrent account updates; retry this transaction")
