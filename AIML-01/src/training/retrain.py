"""A drift-triggered challenger must improve the same held-out validation period."""

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
import redis
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.config import load_config
from src.dashboard.db import query
from src.serving.model import calibrate
from src.training.calibration import fit_calibrator
from src.training.matrices import prepare_matrices
from src.training.registry import MODEL_NAME, log_bundle


def promotion_decision(champion_pr_auc, challenger_pr_auc, minimum_gain=0.001):
    if not np.isfinite(challenger_pr_auc):
        return False
    return challenger_pr_auc > champion_pr_auc + minimum_gain


def retrain(device="cpu"):
    cfg = load_config()
    directory, cache = prepare_matrices()
    train = pd.read_parquet(directory / "train.parquet")
    val = pd.read_parquet(directory / "val.parquet")
    feedback = query("""SELECT l.transaction_id,l.label,t.event_time FROM app.labels l
        JOIN app.transactions t USING(transaction_id)
        WHERE l.created_at <= NOW() AND NOT(t.transaction_data ? 'demo_run')""")
    applied = 0
    for row in feedback:
        mask = train.TransactionID == row["transaction_id"]
        if mask.any() and row["event_time"] <= cache["ranges"]["train"]["max"]:
            train.loc[mask, "isFraud"] = row["label"]
            applied += 1
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    name = os.getenv("MODEL_NAME", MODEL_NAME)
    incumbent = client.get_model_version_by_alias(name, "champion")
    champion = mlflow.pyfunc.load_model(f"models:/{name}@champion").unwrap_python_model()
    metadata = copy.deepcopy(champion.metadata)
    columns = cache["feature_columns"]
    labels = train.isFraud
    middle = len(val) // 2
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=float((len(labels) - labels.sum()) / labels.sum()),
        random_state=cfg.seed,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        n_jobs=4,
        tree_method="hist",
        device=device,
    )
    model.fit(
        train[columns],
        labels,
        eval_set=[(val[columns].iloc[:middle], val.isFraud.iloc[:middle])],
        verbose=40,
    )
    raw = model.predict_proba(val[columns])[:, 1]
    metadata["xgb_calibration"] = fit_calibrator(raw[:middle], val.isFraud.iloc[:middle])
    metadata["best_iteration"] = model.best_iteration
    metadata["source_run_id"] = "drift-retrain-from-" + str(incumbent.version)
    rings_path = cfg.data_processed_dir / "graph" / "account_risk.parquet"
    ring = (
        pd.read_parquet(rings_path).set_index("pseudo_user_id").ring_score
        if rings_path.exists()
        else pd.Series(dtype=float)
    )
    val["ring_score"] = val.pseudo_user_id.map(ring).fillna(metadata["ring_prior"])
    ring_probability = calibrate(val.ring_score.to_numpy(), metadata["gnn_calibration"])
    challenger_scores = (
        metadata["xgb_weight"] * calibrate(raw, metadata["xgb_calibration"])
        + (1 - metadata["xgb_weight"]) * ring_probability
    )
    incumbent_scores = champion.components(val)[-1]
    y = val.isFraud.to_numpy()[middle:]
    champion_ap = float(average_precision_score(y, incumbent_scores[middle:]))
    challenger_ap = float(average_precision_score(y, challenger_scores[middle:]))
    promote = promotion_decision(champion_ap, challenger_ap)
    metadata["threshold"] = float(np.quantile(challenger_scores[middle:], 0.99))
    folder = (
        cfg.data_processed_dir
        / "retraining"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(folder / "model.ubj")
    registered = log_bundle(
        folder / "model.ubj",
        metadata,
        {
            "champion_validation_pr_auc": champion_ap,
            "challenger_validation_pr_auc": challenger_ap,
            "feedback_labels_applied": applied,
        },
        "drift_challenger",
        False,
    )
    connection = redis.Redis(host=cfg.redis_host, port=cfg.redis_port)
    with connection.lock("fraud:promotion", timeout=60, blocking_timeout=10):
        current = client.get_model_version_by_alias(name, "champion")
        if str(current.version) != str(incumbent.version):
            promote = False
        if promote:
            client.set_registered_model_alias(name, "champion", registered["version"])
    result = {
        "decision": "promoted" if promote else "rejected",
        "registry": registered,
        "incumbent_version": incumbent.version,
        "champion_pr_auc": champion_ap,
        "challenger_pr_auc": challenger_ap,
        "minimum_gain": 0.001,
        "feedback_labels_available": len(feedback),
        "feedback_labels_applied": applied,
        "validation_gate_rows": len(y),
        "test_used_for_selection": False,
        "training_cutoff_dt": cache["ranges"]["train"]["max"],
        "device": device,
        "note": (
            "Unlabeled drift traffic is never training truth; only existing training labels "
            "and eligible analyst feedback are used"
        ),
    }
    (folder / "decision.json").write_text(json.dumps(result, indent=2))
    client.log_artifact(registered["run_id"], str(folder / "decision.json"))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    retrain(args.device)
