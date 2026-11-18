"""Replay IEEE-CIS transactions at a chosen event-time multiplier, optionally shifted."""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import load_config
from src.streaming.kafka import ensure_topics, producer
from src.training.data_loader import build_pseudo_user_key, compute_transaction_datetime, merge_data


def load_transactions(limit=1000, offset=0):
    cfg = load_config()
    # CSV remains in chronological dataset order. Do not load 650 MB for a small demo.
    txn = pd.read_csv(cfg.data_raw_dir / "train_transaction.csv", nrows=offset + limit).iloc[
        offset:
    ]
    identity_path = cfg.data_raw_dir / "train_identity.csv"
    identity = pd.read_csv(identity_path) if identity_path.exists() else pd.DataFrame()
    if not identity.empty:
        identity = identity[identity.TransactionID.isin(txn.TransactionID)]
    frame = compute_transaction_datetime(build_pseudo_user_key(merge_data(txn, identity)))
    frame = frame.sort_values("TransactionDT", kind="stable")
    columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    columns += [
        c for c in ["pseudo_user_id", "DeviceType", "P_emaildomain"]
        if c in frame and c not in columns
    ]
    columns = [c for c in columns if c not in ["isFraud", "account_start_day"]]
    records = json.loads(frame[columns].to_json(orient="records"))
    for row in records:
        row["TransactionID"] = int(row["TransactionID"])
    return records


def run(limit=1000, speed=0, drift=False, offset=0):
    cfg = load_config()
    ensure_topics(cfg)
    client = producer(cfg)
    rows = load_transactions(limit, offset)
    previous = None
    failures = []

    def delivered(error, message):
        if error:
            failures.append(str(error))

    for row in rows:
        if speed > 0 and previous is not None:
            time.sleep(max(0, row["TransactionDT"] - previous) / speed)
        previous = row["TransactionDT"]
        if drift:
            row["TransactionAmt"] *= 10
        client.produce(
            cfg.kafka_topic_raw,
            key=row["pseudo_user_id"],
            value=json.dumps(row, allow_nan=False),
            on_delivery=delivered,
        )
        client.poll(0)
    if client.flush(30) or failures:
        raise RuntimeError(f"Kafka delivery failed: {failures}")
    print(json.dumps({"produced": len(rows), "drift": drift, "speed": speed}))
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--speed", type=float, default=0, help="0=as fast as possible")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()
    run(args.limit, args.speed, args.drift, args.offset)
