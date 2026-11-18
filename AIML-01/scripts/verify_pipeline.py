"""Real Kafka -> scorer -> Postgres proof, duplicate/DLQ checks, HTTP latency.

Uses isolated Kafka topics and reserved demo transaction IDs; never resets existing data.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import requests
from confluent_kafka import Consumer

from src.config import load_config
from src.dashboard.db import query, save_label
from src.streaming.kafka import ensure_topics, producer, publish
from src.streaming.producer import load_transactions
from src.streaming.sink import run as sink
from src.streaming.worker import run as worker


def collect(topic, expected, timeout=30):
    cfg = load_config()
    consumer = Consumer(
        {
            "bootstrap.servers": cfg.kafka_bootstrap_servers,
            "group.id": "verify-" + uuid4().hex,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "isolation.level": "read_committed",
        }
    )
    consumer.subscribe([topic])
    rows = []
    deadline = time.monotonic() + timeout
    try:
        while len(rows) < expected and time.monotonic() < deadline:
            message = consumer.poll(0.5)
            if message is not None:
                if message.error():
                    raise RuntimeError(str(message.error()))
                rows.append(json.loads(message.value()))
    finally:
        consumer.close()
    assert len(rows) == expected, (topic, len(rows), expected)
    return rows


def percentiles(values):
    return {f"p{pct}_ms": float(np.percentile(values, pct)) for pct in [50, 95, 99]}


def main(count=1000, api_url="http://127.0.0.1:8000"):
    run_id = uuid4().hex[:10]
    for key, suffix in [
        ("KAFKA_TOPIC_RAW", "raw"),
        ("KAFKA_TOPIC_SCORED", "scored"),
        ("KAFKA_TOPIC_ALERTS", "alerts"),
        ("KAFKA_TOPIC_DLQ", "dlq"),
    ]:
        os.environ[key] = f"verify.{run_id}.{suffix}"
    os.environ["STATE_NAMESPACE"] = "fraud:verify:" + run_id
    os.environ["SCORER_GROUP"] = "verify-scorer-" + run_id
    os.environ["SCORER_TRANSACTION_ID"] = "verify-scorer-" + run_id
    os.environ["SINK_GROUP"] = "verify-sink-" + run_id
    cfg = load_config()
    ensure_topics(cfg)
    rows = load_transactions(count)
    id_offset = 10**12 + int(time.time()) * 10000
    for row in rows:
        row["TransactionID"] += id_offset
        row["demo_run"] = run_id
    client = producer(cfg)
    started = time.perf_counter()
    with ThreadPoolExecutor(2) as pool:
        scored_future = pool.submit(worker, count + 2, 120)
        sink_future = pool.submit(sink, count + 1, 120)
        for row in [*rows, rows[-1]]:
            publish(client, cfg.kafka_topic_raw, row["pseudo_user_id"], row)
        publish(client, cfg.kafka_topic_raw, "bad", {"TransactionID": -1})
        assert client.flush(30) == 0
        assert scored_future.result(300) == count + 2
        assert sink_future.result(300) == count + 1
    stream_seconds = time.perf_counter() - started
    scored = collect(cfg.kafka_topic_scored, count + 1)
    dlq = collect(cfg.kafka_topic_dlq, 1)
    assert len({row["transaction_id"] for row in scored}) == count
    assert sum(row["duplicate"] for row in scored) == 1
    stored = query(
        "SELECT transaction_id FROM app.transactions WHERE transaction_id=ANY(%s)",
        ([row["TransactionID"] for row in rows],),
    )
    assert len(stored) == count
    # Demonstration feedback label on a reserved demo ID, never used as real training feedback.
    reviewed = next(row for row in scored if row["is_alert"] or row["is_audit"])
    save_label(reviewed["transaction_id"], 0)
    assert (
        query(
            "SELECT label FROM app.labels WHERE transaction_id=%s", (reviewed["transaction_id"],)
        )[0]["label"]
        == 0
    )

    response = requests.get(api_url + "/health", timeout=10)
    response.raise_for_status()
    http_latencies = []
    with requests.Session() as session:
        for row in rows:
            row = {**row, "TransactionID": row["TransactionID"] + 10**10}
            row["pseudo_user_id"] = f"api-{run_id}:" + row["pseudo_user_id"]
            before = time.perf_counter()
            response = session.post(api_url + "/score", json=row, timeout=30)
            response.raise_for_status()
            http_latencies.append((time.perf_counter() - before) * 1000)
    result = {
        "run_id": run_id,
        "produced_valid_unique": count,
        "scored_unique": count,
        "scored_including_duplicate": len(scored),
        "dlq": len(dlq),
        "postgres_unique": len(stored),
        "label_persistence": True,
        "stream_elapsed_seconds": stream_seconds,
        "scoring": percentiles([row["latency_ms"] for row in scored if not row["duplicate"]]),
        "http_round_trip": percentiles(http_latencies),
        "http_requests": len(http_latencies),
        "topics": {
            key: os.environ[key]
            for key in ["KAFKA_TOPIC_RAW", "KAFKA_TOPIC_SCORED", "KAFKA_TOPIC_DLQ"]
        },
        "model_versions": sorted({row["model_version"] for row in scored}),
        "latency_scope": (
            "Sequential local requests; HTTP includes transport; "
            "Kafka startup excluded from per-score latency"
        ),
    }
    result["scoring_sla_p99_pass"] = result["scoring"]["p99_ms"] <= cfg.scoring_sla_p99_ms
    directory = cfg.data_processed_dir / "integration"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pipeline.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    main(args.count, args.api_url)
