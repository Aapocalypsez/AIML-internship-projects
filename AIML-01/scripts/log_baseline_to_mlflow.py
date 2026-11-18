"""Log the already-computed baseline results to MLflow.

This avoids re-training and re-running SHAP (which took ~60 min).
Just logs the metrics, params, and saved artifacts.
"""
from pathlib import Path

import mlflow

# Metrics from the actual training run
VAL_METRICS = {
    "val_pr_auc": 0.5204838047862572,
    "val_roc_auc": 0.8992353064581733,
    "val_random_baseline_pr_auc": 0.03891976248477467,
    "val_pr_auc_lift_over_random": 13.37325234165726,
    "val_precision_at_top_1pct": 0.8838095238095238,
    "val_recall_at_top_1pct": 0.22689486552567237,
    "val_n_positive": 4090,
    "val_n_total": 105088,
}

TEST_METRICS = {
    "test_pr_auc": 0.47368622352612194,
    "test_roc_auc": 0.8743014697559907,
    "test_random_baseline_pr_auc": 0.0340414958379923,
    "test_pr_auc_lift_over_random": 13.914965011539254,
    "test_precision_at_top_1pct": 0.8824091778202677,
    "test_recall_at_top_1pct": 0.2591240875912409,
    "test_n_positive": 3562,
    "test_n_total": 104637,
}

ERROR_METRICS = {
    "error_total_FP": 9084,
    "error_total_FN": 1220,
    "error_total_TP": 2870,
    "error_total_TN": 91914,
    "error_FP_mean_amount": 181.99,
    "error_FN_mean_amount": 150.18,
}

PARAMS = {
    "model_type": "xgboost",
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 28.3,
    "early_stopping_rounds": 20,
    "best_iteration": 138,
    "train_rows": 380815,
    "val_rows": 105088,
    "test_rows": 104637,
    "n_features": 400,
    "fraud_rate": 0.035,
    "seed": 42,
}

DATASET_METRICS = {
    "dataset_fraud_rate": 0.03499,
    "dataset_total_rows": 590540,
    "dataset_n_pseudo_users": 217779,
    "dataset_time_span_days": 182.0,
}


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud-detection-baseline")

    with mlflow.start_run(run_name="baseline_xgboost_class_weights"):
        mlflow.log_params(PARAMS)

        all_metrics = {**VAL_METRICS, **TEST_METRICS, **ERROR_METRICS, **DATASET_METRICS}
        for k, v in all_metrics.items():
            mlflow.log_metric(k, v)

        # Log saved artifacts
        artifact_dir = Path(__file__).parent.parent / "data" / "processed" / "baseline"
        for f in artifact_dir.glob("*.png"):
            mlflow.log_artifact(str(f), "plots")
        for f in artifact_dir.glob("*.csv"):
            mlflow.log_artifact(str(f), "error_analysis")

        run_id = mlflow.active_run().info.run_id
        print(f"Logged to MLflow: run_id={run_id}")
        print("View at: http://localhost:5000/#/experiments")


if __name__ == "__main__":
    main()
