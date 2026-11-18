"""Idempotent Kafka-to-Postgres sink; commit offset only after DB commit."""

import argparse
import json
import os
import time

from confluent_kafka import Consumer

from src.config import load_config
from src.dashboard.db import migrate, store_result
from src.streaming.kafka import ensure_topics


def run(limit=0, idle_timeout=0):
    cfg = load_config()
    ensure_topics(cfg)
    migrate()
    consumer = Consumer(
        {
            "bootstrap.servers": cfg.kafka_bootstrap_servers,
            "group.id": os.getenv("SINK_GROUP", "fraud-postgres-sink"),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "isolation.level": "read_committed",
        }
    )
    consumer.subscribe([cfg.kafka_topic_scored])
    count = 0
    last = time.monotonic()
    try:
        while not limit or count < limit:
            message = consumer.poll(1)
            if message is None:
                if idle_timeout and time.monotonic() - last > idle_timeout:
                    break
                continue
            if message.error():
                raise RuntimeError(str(message.error()))
            store_result(json.loads(message.value()))
            consumer.commit(message=message, asynchronous=False)
            count += 1
            last = time.monotonic()
    finally:
        consumer.close()
    print(json.dumps({"stored": count}))
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--idle-timeout", type=int, default=0)
    args = parser.parse_args()
    run(args.limit, args.idle_timeout)
