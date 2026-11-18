"""Tests for data loading, time split, and leakage prevention."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.data_loader import (
    build_pseudo_user_key,
    compute_transaction_datetime,
    get_feature_columns,
    time_based_split,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small synthetic dataset matching IEEE-CIS schema."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "TransactionID": range(1, n + 1),
        "TransactionDT": np.sort(np.random.randint(86400, 86400 * 180, n)),
        "TransactionAmt": np.random.lognormal(4, 1, n),
        "isFraud": np.random.choice([0, 1], n, p=[0.965, 0.035]),
        "card1": np.random.randint(1000, 9999, n),
        "card4": np.random.choice(["visa", "mastercard", "discover"], n),
        "card6": np.random.choice(["debit", "credit"], n),
        "addr1": np.random.randint(100, 500, n),
        "D1": np.where(np.random.random(n) > 0.1, np.random.randint(0, 365, n), np.nan),
        "V1": np.random.random(n),
        "V2": np.random.random(n),
        "C1": np.random.randint(1, 10, n),
    })


def test_fraud_rate_matches_expected(sample_df: pd.DataFrame):
    """Fraud rate should be approximately 3.5% (within sampling noise)."""
    rate = sample_df["isFraud"].mean()
    # With 1000 samples and p=0.035, expect ~35 ± 15 fraud cases
    assert 0.01 < rate < 0.08, f"Fraud rate {rate:.3f} is outside expected range"


def test_time_split_no_temporal_overlap(sample_df: pd.DataFrame):
    """Train, val, test must not overlap in time."""
    train, val, test = time_based_split(sample_df, 0.6, 0.2)

    assert train["TransactionDT"].max() <= val["TransactionDT"].min(), \
        "Train and val overlap in time!"
    assert val["TransactionDT"].max() <= test["TransactionDT"].min(), \
        "Val and test overlap in time!"


def test_time_split_preserves_all_rows(sample_df: pd.DataFrame):
    """All rows must appear in exactly one split."""
    train, val, test = time_based_split(sample_df, 0.6, 0.2)
    total = len(train) + len(val) + len(test)
    assert total == len(sample_df), f"Split lost rows: {total} vs {len(sample_df)}"


def test_time_split_correct_proportions(sample_df: pd.DataFrame):
    """Split sizes should roughly match the requested ratios."""
    train, val, test = time_based_split(sample_df, 0.6, 0.2)
    total = len(sample_df)

    # Allow 10% tolerance because time-based splits won't be exact
    assert abs(len(train) / total - 0.6) < 0.15, f"Train ratio: {len(train)/total:.2f}"
    assert abs(len(val) / total - 0.2) < 0.15, f"Val ratio: {len(val)/total:.2f}"
    assert abs(len(test) / total - 0.2) < 0.15, f"Test ratio: {len(test)/total:.2f}"


def test_no_leakage_future_features(sample_df: pd.DataFrame):
    """Features in training set must not use data from after the training period.

    The key leakage vector is: features computed using future transactions.
    Here we verify that each row's TransactionDT is within the train period.
    (Feature-level leakage tests come in Phase 2 when we build window features.)
    """
    train, val, test = time_based_split(sample_df, 0.6, 0.2)
    train_max_t = train["TransactionDT"].max()

    # No train row should reference a time beyond the training period
    assert (train["TransactionDT"] <= train_max_t).all()

    # Val and test must be strictly after train
    assert val["TransactionDT"].min() >= train_max_t
    assert test["TransactionDT"].min() >= val["TransactionDT"].min()


def test_pseudo_user_key_built(sample_df: pd.DataFrame):
    """Pseudo-user key should be created for all rows (no NaN keys)."""
    result = build_pseudo_user_key(sample_df)
    assert "pseudo_user_id" in result.columns
    assert "account_start_day" in result.columns
    assert result["pseudo_user_id"].notna().all(), "Some pseudo_user_id values are NaN"


def test_pseudo_user_key_groups_same_card(sample_df: pd.DataFrame):
    """Rows with same card1 + addr1 + D1 pattern should share a pseudo-user."""
    df = sample_df.copy()
    # Force two rows to share the same card/address/account
    df.loc[0, "card1"] = 9999
    df.loc[0, "addr1"] = 100
    df.loc[0, "D1"] = 10
    df.loc[0, "TransactionDT"] = 86400 * 50

    df.loc[1, "card1"] = 9999
    df.loc[1, "addr1"] = 100
    df.loc[1, "D1"] = 10
    df.loc[1, "TransactionDT"] = 86400 * 50  # same day → same account_start_day

    result = build_pseudo_user_key(df)
    assert result.loc[0, "pseudo_user_id"] == result.loc[1, "pseudo_user_id"]


def test_transaction_datetime_computed(sample_df: pd.DataFrame):
    """Datetime columns should be computed correctly."""
    result = compute_transaction_datetime(sample_df)
    assert "transaction_datetime" in result.columns
    assert "transaction_hour" in result.columns
    assert "transaction_dayofweek" in result.columns
    assert result["transaction_hour"].between(0, 23).all()
    assert result["transaction_dayofweek"].between(0, 6).all()


def test_feature_columns_exclude_target(sample_df: pd.DataFrame):
    """Feature columns must not include the target or ID columns."""
    feature_cols = get_feature_columns(sample_df)
    forbidden = {"isFraud", "TransactionID", "TransactionDT"}
    assert forbidden.isdisjoint(set(feature_cols)), \
        f"Feature cols contain forbidden: {forbidden & set(feature_cols)}"
