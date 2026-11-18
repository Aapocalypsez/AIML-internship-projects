"""Kafka transactional output + offset commit; cached scoring handles crash retries."""

import argparse
import json
import os
import time

from confluent_kafka import Consumer, TopicPartition
from pydantic import ValidationError

from src.config import load_config
from src.serving.scorer import Scorer
from src.streaming.kafka import ensure_topics, producer, publish


def run(limit=0, idle_timeout=0):
    cfg = load_config()
    ensure_topics(cfg)
    scorer = Scorer()
    consumer = Consumer(
        {
            "bootstrap.servers": cfg.kafka_bootstrap_servers,
            "group.id": os.getenv("SCORER_GROUP", "fraud-scorer"),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "isolation.level": "read_committed",
            "max.poll.interval.ms": 600000,
        }
    )
    output = producer(
        cfg, **{"transactional.id": os.getenv("SCORER_TRANSACTION_ID", "fraud-scorer-v1")}
    )
    output.init_transactions(30)
    consumer.subscribe([cfg.kafka_topic_raw])
    count = 0
    last_message = time.monotonic()
    try:
        while not limit or count < limit:
            message = consumer.poll(1)
            if message is None:
                if idle_timeout and time.monotonic() - last_message > idle_timeout:
                    break
                continue
            if message.error():
                raise RuntimeError(str(message.error()))
            last_message = time.monotonic()
            output.begin_transaction()
            try:
                try:
                    payload = json.loads(message.value())
                    result = scorer.score(payload)
                except (ValueError, ValidationError, KeyError, TypeError) as exc:
                    result = None
                    publish(
                        output,
                        cfg.kafka_topic_dlq,
                        message.key() or "invalid",
                        {
                            "error": str(exc),
                            "payload": message.value().decode(errors="replace"),
                            "partition": message.partition(),
                            "offset": message.offset(),
                        },
                    )
                if result is not None:
                    result["stream_received_at"] = time.time()
                    publish(output, cfg.kafka_topic_scored, result["account_id"], result)
                    if result["is_alert"] or result["is_audit"]:
                        publish(output, cfg.kafka_topic_alerts, result["account_id"], result)
                output.send_offsets_to_transaction(
                    [TopicPartition(message.topic(), message.partition(), message.offset() + 1)],
                    consumer.consumer_group_metadata(),
                    30,
                )
                output.commit_transaction(30)
                count += 1
            except Exception:
                output.abort_transaction(30)
                raise
    finally:
        consumer.close()
    print(json.dumps({"processed": count}))
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--idle-timeout", type=int, default=0)
    args = parser.parse_args()
    run(args.limit, args.idle_timeout)
