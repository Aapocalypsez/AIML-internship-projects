"""Log a portable bundle and bootstrap the first evaluated champion."""

import json
import os

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.config import load_config
from src.serving.model import FraudModel

MODEL_NAME = "fraud-detector"


def log_bundle(model_path, metadata, metrics, run_name, promote=False):
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("fraud-platform")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            "fraud-platform", artifact_location="mlflow-artifacts:/fraud-platform"
        )
    )
    directory = cfg.data_processed_dir / "registry"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}.json"
    path.write_text(json.dumps(metadata, indent=2))
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "xgb_weight": metadata["xgb_weight"],
                "source_run_id": metadata.get("source_run_id", "retrain"),
                "threshold": metadata["threshold"],
            }
        )
        info = mlflow.pyfunc.log_model(
            "model",
            python_model=FraudModel(),
            artifacts={"xgboost": str(model_path), "metadata": str(path)},
            code_paths=[str(cfg.project_root / "src")],
            pip_requirements=[
                "mlflow==2.22.5",
                "xgboost==2.1.4",
                "scikit-learn==1.6.1",
                "pandas==2.3.3",
                "numpy==2.4.6",
            ],
        )
        version = mlflow.register_model(info.model_uri, os.getenv("MODEL_NAME", MODEL_NAME))
        if promote:
            client.set_registered_model_alias(version.name, "champion", version.version)
        return {"run_id": run.info.run_id, "version": version.version, "name": version.name}


def bootstrap():
    cfg = load_config()
    output = cfg.data_processed_dir / "features"
    result = json.loads((output / "results.json").read_text())
    validation = pd.read_parquet(output / "val_predictions.parquet")
    metadata = {
        "feature_columns": result["feature_columns"],
        "best_iteration": result["best_iteration"],
        "xgb_calibration": {"coef": 1.0, "intercept": 0.0},
        "gnn_calibration": {"coef": 1.0, "intercept": 0.0},
        "xgb_weight": 1.0,
        "ring_prior": float(validation.isFraud.mean()),
        "threshold": float(np.quantile(validation.fraud_probability, 0.99)),
        "source_run_id": result["run_id"],
        "graph_enabled": False,
    }
    registered = log_bundle(
        output / "model.ubj",
        metadata,
        {f"{split}_pr_auc": result["metrics"][split]["pr_auc"] for split in ["val", "test"]},
        "phase3_initial_champion",
        True,
    )
    (cfg.data_processed_dir / "registry" / "bootstrap.json").write_text(
        json.dumps(registered, indent=2)
    )
    print(json.dumps(registered))


if __name__ == "__main__":
    bootstrap()
