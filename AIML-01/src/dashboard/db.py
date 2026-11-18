"""Database schema and analyst workflow; each write uses one database transaction."""

from pathlib import Path

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from src.config import load_config


def connect():
    cfg = load_config()
    return psycopg2.connect(cfg.postgres_url, connect_timeout=5)


def migrate():
    sql = (Path(__file__).resolve().parents[2] / "docker" / "app.sql").read_text()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(sql)


def store_result(result):
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO app.transactions(transaction_id, account_id, event_time, score,
                                         model_version, transaction_data, result)
            VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(transaction_id) DO NOTHING
        """,
            (
                result["transaction_id"],
                result["account_id"],
                result["event_time"],
                result["score"],
                result["model_version"],
                Json(result["transaction"]),
                Json(result),
            ),
        )
        if result["is_alert"] or result["is_audit"]:
            reason = "risk" if result["is_alert"] else "random_audit"
            cursor.execute(
                """INSERT INTO app.alerts(transaction_id, review_reason)
                VALUES (%s,%s) ON CONFLICT(transaction_id) DO NOTHING""",
                (result["transaction_id"], reason),
            )
            cursor.execute(
                """INSERT INTO app.cases(transaction_id) VALUES (%s)
                ON CONFLICT(transaction_id) DO NOTHING""",
                (result["transaction_id"],),
            )


def query(sql, params=()):
    with connect() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def set_case(transaction_id, status, note=""):
    if status not in ["open", "reviewing", "closed"]:
        raise ValueError("Unknown case status")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """UPDATE app.cases SET status=%s, analyst_note=%s, updated_at=NOW()
                          WHERE transaction_id=%s""",
            (status, note, transaction_id),
        )


def save_label(transaction_id, label):
    if label not in [0, 1]:
        raise ValueError("Label must be 0 or 1")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO app.labels(transaction_id,label) VALUES (%s,%s)
            ON CONFLICT(transaction_id) DO UPDATE SET label=EXCLUDED.label, created_at=NOW()""",
            (transaction_id, label),
        )
        cursor.execute(
            "UPDATE app.cases SET status='closed', updated_at=NOW() WHERE transaction_id=%s",
            (transaction_id,),
        )


def store_rings(rings):
    from psycopg2.extras import execute_values

    columns = ["pseudo_user_id", "ring_id", "ring_score", "ring_size", "snapshot_transaction_dt"]
    with connect() as connection, connection.cursor() as cursor:
        execute_values(
            cursor,
            """INSERT INTO app.ring_members
            (account_id,ring_id,ring_score,ring_size,snapshot_transaction_dt) VALUES %s
            ON CONFLICT(account_id) DO UPDATE SET ring_id=EXCLUDED.ring_id,
            ring_score=EXCLUDED.ring_score,ring_size=EXCLUDED.ring_size,
            snapshot_transaction_dt=EXCLUDED.snapshot_transaction_dt""",
            list(rings[columns].itertuples(index=False, name=None)),
            page_size=1000,
        )


if __name__ == "__main__":
    migrate()
    print("Application schema ready")
