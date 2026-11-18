"""Train one fixed XGBoost experiment; compare to an existing logged baseline.

Run: python -m src.training.train_with_features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient
from mlflow.models import infer_signature

from feature_repo.definitions import transaction, transaction_features
from src.config import Config, load_config
from src.features.engine import ENGINEERED_FEATURE_NAMES, engineer_features
from src.features.store import export_features, feature_store, historical_features
from src.training.data_loader import prepare_dataset
from src.training.train_baseline import compute_metrics, plot_precision_recall_curve

BASELINE_RUN_ID = "fcb25615e762440292d6fa9a5b49f04b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_with_features(
    cfg: Config | None = None, baseline_run_id=BASELINE_RUN_ID, device="cpu"
) -> dict:
    cfg = cfg or load_config()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()
    baseline = client.get_run(baseline_run_id)
    print("Loading IEEE-CIS data...", flush=True)
    data = prepare_dataset(cfg)
    parts = [data[name] for name in ["train", "val", "test"]]
    original_columns = data["feature_cols"]
    required = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "pseudo_user_id",
        "transaction_datetime",
        "transaction_hour",
        "transaction_dayofweek",
        "DeviceType",
        "P_emaildomain",
    ]
    raw = pd.concat([part[[c for c in required if c in part]] for part in parts])
    print(f"Engineering {len(raw):,} rows with strictly earlier history...", flush=True)
    engineered = engineer_features(raw)
    repo = (cfg.project_root / cfg.feast_repo_path).resolve()
    export_features(engineered, repo)
    store = feature_store(cfg)
    store.apply([transaction, transaction_features])
    print("Retrieving engineered features through Feast point-in-time joins...", flush=True)
    stored = historical_features(store, raw)
    expected = engineered.set_index("TransactionID").loc[
        raw.TransactionID, ENGINEERED_FEATURE_NAMES
    ]
    np.testing.assert_allclose(stored.to_numpy(), expected.to_numpy(), rtol=1e-8, atol=1e-8)
    del engineered, expected, raw
    feature_columns = original_columns + ENGINEERED_FEATURE_NAMES
    matrices = []
    for part in parts:
        matrix = part[original_columns].astype("float32")
        matrix[ENGINEERED_FEATURE_NAMES] = stored.loc[part.index].astype("float32")
        matrices.append(matrix)
    del stored
    # Release the 400-column raw frames before XGBoost allocates its training buffers.
    parts = [part[["TransactionID", "isFraud"]].copy() for part in parts]
    for name, part in zip(["train", "val", "test"], parts):
        data[name] = part
    train, val, test = parts
    for name, part in [("val", val), ("test", test)]:
        if baseline.data.metrics[f"{name}_n_total"] != len(part):
            raise ValueError("Baseline split size differs; comparison would be invalid")
    weight = float((len(train) - train.isFraud.sum()) / train.isFraud.sum())
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": weight,
        "random_state": cfg.seed,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 20,
        "n_jobs": 4,
        "tree_method": "hist",
        "device": device,
    }
    print("Training fixed baseline settings + 18 engineered features...", flush=True)
    model = xgb.XGBClassifier(**params)
    model.fit(matrices[0], train.isFraud, eval_set=[(matrices[1], val.isFraud)], verbose=20)
    output = cfg.data_processed_dir / "features"
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / "model.ubj")
    metrics, comparison = {}, {}
    for name, part, matrix in [("val", val, matrices[1]), ("test", test, matrices[2])]:
        proba = model.predict_proba(matrix)[:, 1]
        metrics[name] = compute_metrics(part.isFraud.to_numpy(), proba, part.isFraud.mean())
        base = baseline.data.metrics[f"{name}_pr_auc"]
        comparison[name] = {
            "baseline_pr_auc": base,
            "engineered_pr_auc": metrics[name]["pr_auc"],
            "absolute_delta": metrics[name]["pr_auc"] - base,
        }
        pd.DataFrame(
            {
                "TransactionID": part.TransactionID,
                "isFraud": part.isFraud,
                "fraud_probability": proba,
            }
        ).to_parquet(output / f"{name}_predictions.parquet")
        plot_precision_recall_curve(
            part.isFraud.to_numpy(), proba, part.isFraud.mean(), output / f"pr_curve_{name}.png"
        )
    inputs = [cfg.data_raw_dir / "train_transaction.csv", cfg.data_raw_dir / "train_identity.csv"]
    result = {
        "baseline_run_id": baseline_run_id,
        "baseline_run_status": baseline.info.status,
        "params": params,
        "best_iteration": model.best_iteration,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "metrics": metrics,
        "comparison": comparison,
        "split_rows": {name: len(data[name]) for name in ["train", "val", "test"]},
        "split_id_sha256": {
            name: hashlib.sha256(data[name].TransactionID.to_numpy().tobytes()).hexdigest()
            for name in ["train", "val", "test"]
        },
        "inputs_sha256": {path.name: sha256(path) for path in inputs if path.exists()},
        "engine_sha256": sha256(cfg.project_root / "src" / "features" / "engine.py"),
        "versions": {
            name: version(name)
            for name in ["pandas", "numpy", "scikit-learn", "xgboost", "feast", "mlflow"]
        },
        "historical_lookup_verified_rows": sum(len(part) for part in parts),
        "selection": "Fixed settings; validation-only early stopping; no test tuning or promotion",
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(comparison, indent=2), flush=True)

    experiment_name = "fraud-detection-features"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            experiment_name, artifact_location="mlflow-artifacts:/fraud-detection-features"
        )
    )
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="xgboost_engineered_features"
    ) as run:
        mlflow.log_params(
            {
                **params,
                "baseline_run_id": baseline_run_id,
                "n_features": len(feature_columns),
                "best_iteration": model.best_iteration,
            }
        )
        mlflow.set_tags(
            {
                "phase": "2",
                "feature_history": "strictly_before_event_time",
                "baseline_provenance": "Existing run logged from prior saved results",
            }
        )
        for name in ["val", "test"]:
            mlflow.log_metrics(
                {f"{name}_{k}".replace("%", "pct"): v for k, v in metrics[name].items()}
            )
            mlflow.log_metric(f"{name}_pr_auc_delta", comparison[name]["absolute_delta"])
        example = matrices[0].head(5)
        mlflow.xgboost.log_model(
            model,
            "model",
            signature=infer_signature(example, model.predict_proba(example)),
            input_example=example,
            pip_requirements=[
                f"xgboost=={version('xgboost')}",
                f"pandas=={version('pandas')}",
                f"numpy=={version('numpy')}",
                f"scikit-learn=={version('scikit-learn')}",
            ],
        )
        result["run_id"] = run.info.run_id
        result["model_uri"] = f"runs:/{run.info.run_id}/model"
        result_path.write_text(json.dumps(result, indent=2))
        for path in output.iterdir():
            if path.suffix in [".json", ".png"]:
                mlflow.log_artifact(str(path))
    print(f"MLflow run: {result['run_id']}", flush=True)
    return result


if __name__ == "__main__":
    # MLflow prints Unicode links; Windows redirected stdout otherwise uses cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-id", default=BASELINE_RUN_ID)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    train_with_features(baseline_run_id=args.baseline_run_id, device=args.device)
