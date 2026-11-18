"""Shared Kafka setup and delivery helpers."""

import json

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.config import load_config


def ensure_topics(cfg=None):
    cfg = cfg or load_config()
    admin = AdminClient({"bootstrap.servers": cfg.kafka_bootstrap_servers})
    existing = admin.list_topics(timeout=15).topics
    topics = [
        cfg.kafka_topic_raw,
        cfg.kafka_topic_scored,
        cfg.kafka_topic_alerts,
        cfg.kafka_topic_dlq,
    ]
    pending = [
        NewTopic(name, num_partitions=1, replication_factor=1)
        for name in topics
        if name not in existing
    ]
    if pending:
        for result in admin.create_topics(pending).values():
            result.result(20)


def producer(cfg=None, **options):
    cfg = cfg or load_config()
    return Producer(
        {"bootstrap.servers": cfg.kafka_bootstrap_servers, "enable.idempotence": True, **options}
    )


def publish(client, topic, key, value):
    client.produce(topic, key=str(key), value=json.dumps(value, allow_nan=False).encode())
    client.poll(0)
