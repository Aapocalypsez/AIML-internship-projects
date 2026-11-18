"""Baseline XGBoost trainer with MLflow tracking.

Trains an XGBoost classifier with class weights for imbalanced fraud detection,
runs hyperparameter tuning, logs everything to MLflow, and produces an
error analysis report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import Config, load_config
from src.training.data_loader import prepare_dataset

# Lazy import — only needed if MLflow server is running
try:
    import mlflow
    import mlflow.xgboost

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Lazy import for SHAP
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fraud_rate: float,
    top_k_pct: float = 0.01,
) -> dict:
    """Compute all evaluation metrics.

    PR-AUC (Precision-Recall Area Under Curve):
    Unlike ROC-AUC, PR-AUC focuses on the positive (fraud) class.
    The random baseline for PR-AUC equals the fraud rate (~3.5%).
    A PR-AUC of 0.5 on this dataset means we're ~14x better than random.

    Args:
        y_true: true binary labels
        y_proba: predicted probabilities for the positive class
        fraud_rate: the fraud rate (used as random baseline for PR-AUC)
        top_k_pct: what fraction of traffic analysts can review (default 1%)
    """
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    # Precision/recall at top-k% threshold
    # If analysts can only review the top 1% of transactions,
    # what fraction of those are actually fraud (precision)?
    # And what fraction of all fraud do they catch (recall)?
    n_top = max(1, int(len(y_true) * top_k_pct))
    top_k_idx = np.argsort(y_proba)[-n_top:]
    top_k_labels = y_true[top_k_idx]

    precision_at_k = top_k_labels.sum() / len(top_k_labels) if len(top_k_labels) > 0 else 0.0
    recall_at_k = top_k_labels.sum() / y_true.sum() if y_true.sum() > 0 else 0.0

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "random_baseline_pr_auc": float(fraud_rate),
        "pr_auc_lift_over_random": float(pr_auc / fraud_rate) if fraud_rate > 0 else 0.0,
        f"precision_at_top_{top_k_pct:.0%}": float(precision_at_k),
        f"recall_at_top_{top_k_pct:.0%}": float(recall_at_k),
        "n_positive": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def plot_precision_recall_curve(
    y_true: np.ndarray, y_proba: np.ndarray, fraud_rate: float, save_path: Path
) -> None:
    """Save a precision-recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label="XGBoost", linewidth=2)
    ax.axhline(y=fraud_rate, color="red", linestyle="--",
               label=f"Random baseline ({fraud_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    save_dir: Path,
) -> dict:
    """Analyze where the model fails: FP/FN breakdown by amount and card type.

    Returns a summary dict for logging.
    """
    y_pred = (y_proba >= threshold).astype(int)

    df_analysis = df[["TransactionAmt"]].copy()
    if "card4" in df.columns:
        df_analysis["card4"] = df["card4"]
    if "card6" in df.columns:
        df_analysis["card6"] = df["card6"]

    df_analysis["y_true"] = y_true
    df_analysis["y_pred"] = y_pred
    df_analysis["y_proba"] = y_proba

    # Classify each prediction
    conditions = [
        (df_analysis["y_true"] == 0) & (df_analysis["y_pred"] == 0),
        (df_analysis["y_true"] == 0) & (df_analysis["y_pred"] == 1),
        (df_analysis["y_true"] == 1) & (df_analysis["y_pred"] == 0),
        (df_analysis["y_true"] == 1) & (df_analysis["y_pred"] == 1),
    ]
    labels = ["TN", "FP", "FN", "TP"]
    df_analysis["category"] = np.select(conditions, labels, default="?")

    # Amount buckets
    df_analysis["amount_bucket"] = pd.cut(
        df_analysis["TransactionAmt"],
        bins=[0, 50, 100, 200, 500, 1000, float("inf")],
        labels=["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"],
    )

    # Error rates by amount bucket
    error_by_amount = (
        df_analysis.groupby("amount_bucket", observed=True)["category"]
        .value_counts()
        .unstack(fill_value=0)
    )
    error_by_amount.to_csv(save_dir / "error_by_amount.csv")

    # Error rates by card type
    if "card4" in df_analysis.columns:
        error_by_card = (
            df_analysis.groupby("card4")["category"]
            .value_counts()
            .unstack(fill_value=0)
        )
        error_by_card.to_csv(save_dir / "error_by_card4.csv")

    summary = {
        "total_FP": int((df_analysis["category"] == "FP").sum()),
        "total_FN": int((df_analysis["category"] == "FN").sum()),
        "total_TP": int((df_analysis["category"] == "TP").sum()),
        "total_TN": int((df_analysis["category"] == "TN").sum()),
        "FP_mean_amount": float(
            df_analysis.loc[df_analysis["category"] == "FP", "TransactionAmt"].mean()
        ) if (df_analysis["category"] == "FP").any() else 0.0,
        "FN_mean_amount": float(
            df_analysis.loc[df_analysis["category"] == "FN", "TransactionAmt"].mean()
        ) if (df_analysis["category"] == "FN").any() else 0.0,
    }

    return summary


def run_shap_analysis(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    save_dir: Path,
    max_samples: int = 1000,
) -> None:
    """Generate SHAP feature importance plots.

    SHAP (SHapley Additive exPlanations) shows how much each feature
    contributes to a prediction. Positive SHAP = pushes toward fraud,
    negative SHAP = pushes toward legit. The summary plot shows which
    features matter most across all predictions.
    """
    if not HAS_SHAP:
        print("SHAP not installed, skipping SHAP analysis")
        return

    # Sample for speed
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Global feature importance (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Bar plot (mean absolute SHAP)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Top 3 individual waterfall plots
    for i in range(min(3, len(X_sample))):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=X_sample.iloc[i],
            feature_names=X_sample.columns.tolist(),
        ), show=False)
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_waterfall_{i}.png", dpi=150, bbox_inches="tight")
        plt.close("all")


def train_baseline(cfg: Config | None = None, use_mlflow: bool = True) -> dict:
    """Train baseline XGBoost with class weights.

    Class weights: because only ~3.5% of transactions are fraud,
    an unweighted model would learn to predict "legit" for everything
    and still be 96.5% accurate. scale_pos_weight tells XGBoost to
    penalize missing fraud cases more heavily:
      scale_pos_weight = n_legit / n_fraud ≈ 27
    """
    if cfg is None:
        cfg = load_config()

    print("Loading and preparing dataset...")
    data = prepare_dataset(cfg)
    train, val, test = data["train"], data["val"], data["test"]
    feature_cols = data["feature_cols"]
    stats = data["stats"]

    print(f"Dataset stats: {json.dumps({k: str(v) for k, v in stats.items()}, indent=2)}")

    X_train = train[feature_cols]
    y_train = train["isFraud"]
    X_val = val[feature_cols]
    y_val = val["isFraud"]

    # Class weight = n_negative / n_positive
    n_fraud = y_train.sum()
    n_legit = len(y_train) - n_fraud
    scale_pos_weight = n_legit / n_fraud if n_fraud > 0 else 1.0

    output_dir = cfg.project_root / "data" / "processed" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase A: Default XGBoost with class weights ---
    print(f"\n=== Training baseline XGBoost (scale_pos_weight={scale_pos_weight:.1f}) ===")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.seed,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    # Evaluate on validation
    val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val.values, val_proba, stats["val_fraud_rate"])
    print(f"\nValidation metrics: {json.dumps(val_metrics, indent=2)}")

    # Evaluate on test (held out — report only, never tune on this)
    X_test = test[feature_cols]
    y_test = test["isFraud"]
    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test.values, test_proba, stats["test_fraud_rate"])
    print(f"\nTest metrics: {json.dumps(test_metrics, indent=2)}")

    # Plots
    plot_precision_recall_curve(y_val.values, val_proba, stats["val_fraud_rate"],
                                output_dir / "pr_curve_val.png")
    plot_precision_recall_curve(y_test.values, test_proba, stats["test_fraud_rate"],
                                output_dir / "pr_curve_test.png")

    # Confusion matrix at threshold=0.5
    plot_confusion_matrix(y_val.values, (val_proba >= 0.5).astype(int),
                          output_dir / "confusion_matrix_val.png")

    # Error analysis
    error_summary = error_analysis(val, y_val.values, val_proba, 0.5, output_dir)
    print(f"\nError analysis: {json.dumps(error_summary, indent=2)}")

    # SHAP
    print("\nRunning SHAP analysis...")
    run_shap_analysis(model, X_val, output_dir)

    # --- Log to MLflow ---
    if use_mlflow and HAS_MLFLOW:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment("fraud-detection-baseline")

        with mlflow.start_run(run_name="baseline_xgboost_class_weights"):
            # Log params
            mlflow.log_params({
                "model_type": "xgboost",
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": round(scale_pos_weight, 2),
                "train_rows": stats["train_rows"],
                "val_rows": stats["val_rows"],
                "test_rows": stats["test_rows"],
                "n_features": stats["n_features"],
                "fraud_rate": round(stats["fraud_rate"], 4),
                "seed": cfg.seed,
            })

            # Log validation metrics
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}".replace("%", "pct"), v)

            # Log test metrics
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}".replace("%", "pct"), v)

            # Log error analysis
            for k, v in error_summary.items():
                mlflow.log_metric(f"error_{k}", v)

            # Log dataset stats
            mlflow.log_metric("dataset_fraud_rate", stats["fraud_rate"])
            mlflow.log_metric("dataset_total_rows", stats["total_rows"])

            # Log artifacts (plots)
            for f in output_dir.glob("*.png"):
                mlflow.log_artifact(str(f), "plots")
            for f in output_dir.glob("*.csv"):
                mlflow.log_artifact(str(f), "error_analysis")

            # Log model
            mlflow.xgboost.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id
            print(f"\nMLflow run logged: {run_id}")

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "error_summary": error_summary,
        "stats": stats,
        "feature_cols": feature_cols,
    }


def run_hyperparameter_tuning(cfg: Config | None = None, use_mlflow: bool = True) -> dict:
    """Run RandomizedSearchCV for hyperparameter tuning.

    Hyperparameter tuning: instead of guessing learning_rate=0.1 and
    max_depth=6, we try many combinations randomly and pick the one
    with the best validation PR-AUC. Each trial is logged to MLflow
    so we can compare them all.
    """
    if cfg is None:
        cfg = load_config()

    print("Loading and preparing dataset...")
    data = prepare_dataset(cfg)
    train, val, test = data["train"], data["val"], data["test"]
    feature_cols = data["feature_cols"]
    stats = data["stats"]

    X_train = train[feature_cols]
    y_train = train["isFraud"]
    X_val = val[feature_cols]
    y_val = val["isFraud"]

    n_fraud = y_train.sum()
    n_legit = len(y_train) - n_fraud
    scale_pos_weight = n_legit / n_fraud if n_fraud > 0 else 1.0

    # Param search space
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.seed,
        eval_metric="aucpr",
        n_jobs=-1,
    )

    print(f"\n=== Hyperparameter tuning ({cfg.n_tuning_trials} trials) ===")

    search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=cfg.n_tuning_trials,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=3),
        random_state=cfg.seed,
        verbose=1,
        n_jobs=1,  # XGBoost already uses all cores internally
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV score (avg precision): {search.best_score_:.4f}")

    # Evaluate best model on validation
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val.values, val_proba, stats["val_fraud_rate"])
    print(f"\nValidation metrics (best model): {json.dumps(val_metrics, indent=2)}")

    # Test
    X_test = test[feature_cols]
    y_test = test["isFraud"]
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test.values, test_proba, stats["test_fraud_rate"])
    print(f"\nTest metrics (best model): {json.dumps(test_metrics, indent=2)}")

    output_dir = cfg.project_root / "data" / "processed" / "tuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_precision_recall_curve(y_val.values, val_proba, stats["val_fraud_rate"],
                                output_dir / "pr_curve_val_tuned.png")

    # Log to MLflow
    if use_mlflow and HAS_MLFLOW:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment("fraud-detection-baseline")

        with mlflow.start_run(run_name="tuned_xgboost"):
            mlflow.log_params(search.best_params_)
            mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))
            mlflow.log_param("n_tuning_trials", cfg.n_tuning_trials)
            mlflow.log_metric("cv_best_score", search.best_score_)

            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}".replace("%", "pct"), v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}".replace("%", "pct"), v)

            mlflow.xgboost.log_model(best_model, "model")

            for f in output_dir.glob("*.png"):
                mlflow.log_artifact(str(f), "plots")

            run_id = mlflow.active_run().info.run_id
            print(f"\nMLflow run logged: {run_id}")

    return {
        "model": best_model,
        "best_params": search.best_params_,
        "cv_score": search.best_score_,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: Baseline XGBoost Training")
    print("=" * 60)

    # Train baseline with class weights
    baseline_results = train_baseline(use_mlflow=True)

    print("\n" + "=" * 60)
    print("PHASE 1: Hyperparameter Tuning")
    print("=" * 60)

    # Run tuning
    tuned_results = run_hyperparameter_tuning(use_mlflow=True)

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON: Baseline vs Tuned")
    print("=" * 60)
    print(f"Baseline val PR-AUC: {baseline_results['val_metrics']['pr_auc']:.4f}")
    print(f"Tuned    val PR-AUC: {tuned_results['val_metrics']['pr_auc']:.4f}")
    print(f"Random baseline:     {baseline_results['stats']['val_fraud_rate']:.4f}")
