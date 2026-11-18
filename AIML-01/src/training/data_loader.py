"""Data loading and preprocessing for IEEE-CIS Fraud Detection dataset.

Handles:
- Loading raw CSV files (transaction + identity)
- Merging them on TransactionID
- Building the pseudo-user key (card + address + account start day)
- Time-based train/val/test split
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config


def load_raw_data(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw transaction and identity CSVs.

    Returns:
        (train_transaction, train_identity) DataFrames
    """
    if cfg is None:
        cfg = load_config()

    raw_dir = cfg.data_raw_dir
    txn_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    if not txn_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {txn_path}. "
            "Download from https://www.kaggle.com/c/ieee-fraud-detection/data "
            f"and place files in {raw_dir}"
        )

    txn = pd.read_csv(txn_path)
    identity = pd.read_csv(id_path) if id_path.exists() else pd.DataFrame()

    return txn, identity


def merge_data(txn: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Merge transaction and identity tables on TransactionID."""
    if identity.empty:
        return txn
    return txn.merge(identity, on="TransactionID", how="left")


def build_pseudo_user_key(df: pd.DataFrame) -> pd.DataFrame:
    """Build a pseudo-user key from card + address + account start day.

    The IEEE-CIS dataset has no explicit user ID. We create one by combining:
    - card1 (card number hash)
    - addr1 (billing address)
    - account_start_day: derived from D1, which is "days since account opened"
      so account_start_day = TransactionDT - D1 * 86400

    This groups transactions that likely belong to the same cardholder.
    """
    df = df.copy()

    # D1 is "days since account opened" — convert to account start reference
    # TransactionDT is in seconds from some reference point
    df["account_start_day"] = np.where(
        df["D1"].notna(),
        ((df["TransactionDT"] - df["D1"] * 86400) // 86400).astype("Int64"),
        pd.NA,
    )

    # Build the key from available fields
    # When D1 is missing, use -1 as account_start_day placeholder
    asd_str = df["account_start_day"].fillna(-1).astype(int).astype(str)
    df["pseudo_user_id"] = (
        df["card1"].astype(str)
        + "_"
        + df["addr1"].fillna(-1).astype(int).astype(str)
        + "_"
        + asd_str
    )

    return df


def compute_transaction_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TransactionDT (seconds offset) to a datetime column.

    The IEEE-CIS TransactionDT is seconds from a reference start date.
    We use 2017-12-01 as the reference based on Kaggle discussions.
    """
    df = df.copy()
    reference_date = pd.Timestamp("2017-12-01")
    df["transaction_datetime"] = reference_date + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["transaction_date"] = df["transaction_datetime"].dt.date
    df["transaction_hour"] = df["transaction_datetime"].dt.hour
    df["transaction_dayofweek"] = df["transaction_datetime"].dt.dayofweek
    return df


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by time into train/validation/test sets.

    Why time-based split (not random)?
    In fraud detection, patterns evolve over time — new fraud tactics emerge,
    spending patterns shift. A random split would let the model "peek" at future
    patterns during training, inflating metrics. A time split simulates real
    deployment: train on past, predict the future.

    Args:
        df: DataFrame sorted by TransactionDT
        train_ratio: fraction of time range for training
        val_ratio: fraction of time range for validation
        (test gets the remainder)

    Returns:
        (train_df, val_df, test_df)
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    min_t = df["TransactionDT"].min()
    max_t = df["TransactionDT"].max()
    total_range = max_t - min_t

    train_cutoff = min_t + total_range * train_ratio
    val_cutoff = min_t + total_range * (train_ratio + val_ratio)

    train_df = df[df["TransactionDT"] < train_cutoff].copy()
    val_df = df[(df["TransactionDT"] >= train_cutoff) & (df["TransactionDT"] < val_cutoff)].copy()
    test_df = df[df["TransactionDT"] >= val_cutoff].copy()

    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable for model training.

    Excludes: target, IDs, datetime columns, and the pseudo-user key.
    """
    exclude = {
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "pseudo_user_id",
        "account_start_day",
        "transaction_datetime",
        "transaction_date",
        "transaction_hour",
        "transaction_dayofweek",
    }

    # Keep only numeric columns (XGBoost handles NaN natively)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    return feature_cols


def prepare_dataset(cfg: Config | None = None) -> dict:
    """Full pipeline: load → merge → pseudo-user → datetime → time-split.

    Returns a dict with keys:
        train, val, test: DataFrames
        feature_cols: list of feature column names
        stats: dict with dataset statistics
    """
    if cfg is None:
        cfg = load_config()

    # Load and merge
    txn, identity = load_raw_data(cfg)
    df = merge_data(txn, identity)

    # Build pseudo-user key and datetime
    df = build_pseudo_user_key(df)
    df = compute_transaction_datetime(df)

    # Time-based split
    train, val, test = time_based_split(df, cfg.train_ratio, cfg.val_ratio)

    feature_cols = get_feature_columns(train)

    stats = {
        "total_rows": len(df),
        "fraud_rate": df["isFraud"].mean(),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_fraud_rate": train["isFraud"].mean(),
        "val_fraud_rate": val["isFraud"].mean(),
        "test_fraud_rate": test["isFraud"].mean(),
        "n_features": len(feature_cols),
        "n_pseudo_users": df["pseudo_user_id"].nunique(),
        "time_span_days": (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400,
        "train_time_range": (
            train["transaction_datetime"].min(),
            train["transaction_datetime"].max(),
        ),
        "val_time_range": (
            val["transaction_datetime"].min(),
            val["transaction_datetime"].max(),
        ),
        "test_time_range": (
            test["transaction_datetime"].min(),
            test["transaction_datetime"].max(),
        ),
    }

    return {
        "train": train,
        "val": val,
        "test": test,
        "feature_cols": feature_cols,
        "stats": stats,
    }
