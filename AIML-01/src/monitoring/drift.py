"""Evidently 0.7 distribution checks with an explicit, inspectable trigger."""

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import load_config
from src.dashboard.db import query
from src.training.matrices import prepare_matrices

MONITORED_COLUMNS = ["TransactionAmt", "C1", "D1", "V1"]


def detect_drift(reference, current, output):
    if len(reference) < 100 or len(current) < 100:
        raise ValueError("At least 100 reference and recent rows required")
    columns = [name for name in MONITORED_COLUMNS if name in reference and name in current]
    if not columns:
        raise ValueError("No monitored columns available")
    reference = reference[columns].apply(pd.to_numeric, errors="coerce")
    current = current[columns].apply(pd.to_numeric, errors="coerce")
    report = Report(
        [
            DataDriftPreset(
                columns=columns, drift_share=0.25, num_method="wasserstein", num_threshold=0.1
            )
        ]
    )
    snapshot = report.run(Dataset.from_pandas(current), Dataset.from_pandas(reference))
    output.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output / "report.html"))
    snapshot.save_json(str(output / "evidently.json"))
    evidence = json.loads(snapshot.json())
    metric = next(
        row for row in evidence["metrics"] if row["metric_name"].startswith("DriftedColumnsCount")
    )
    share = float(metric["value"]["share"])
    fingerprint = hashlib.sha256(current.to_json().encode()).hexdigest()
    summary = {
        "drift_detected": share >= 0.25,
        "drifted_column_share": share,
        "trigger_share": 0.25,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "columns": columns,
        "method": "normalized Wasserstein; per-column threshold 0.1",
        "current_fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def current_traffic(limit=5000):
    rows = query(
        "SELECT transaction_data FROM app.transactions ORDER BY received_at DESC LIMIT %s", (limit,)
    )
    return pd.DataFrame([row["transaction_data"] for row in rows])


def run_monitor(retrain=False, device="cpu"):
    cfg = load_config()
    directory, _ = prepare_matrices()
    reference = pd.read_parquet(directory / "train.parquet", columns=MONITORED_COLUMNS)
    reference = reference.sample(min(10000, len(reference)), random_state=cfg.seed)
    current = current_traffic()
    folder = (
        cfg.data_processed_dir
        / "monitoring"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if len(current) < 100:
        folder.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "insufficient_data",
            "drift_detected": False,
            "current_rows": len(current),
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    summary = detect_drift(reference, current, folder)
    if retrain and summary["drift_detected"]:
        import redis

        from src.training.retrain import retrain as train

        client = redis.Redis(host=cfg.redis_host, port=cfg.redis_port)
        key = "fraud:retrain:" + summary["current_fingerprint"]
        if client.set(key, "running", nx=True, ex=3600):
            try:
                summary["retraining"] = train(device=device)
                client.set(key, "complete", ex=86400)
            except Exception:
                client.delete(key)
                raise
        else:
            summary["retraining"] = {"decision": "already processed or running"}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
