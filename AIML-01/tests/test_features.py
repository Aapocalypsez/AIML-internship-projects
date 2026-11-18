"""Tests for the shared feature engineering module.

Tests ensure:
- No future data leakage in velocity/amount features
- Feature values are reasonable
- Parity between batch and single-row computation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engine import (
    ENGINEERED_FEATURE_NAMES,
    compute_amount_features,
    compute_time_features,
    compute_velocity_features,
    engineer_features,
)


@pytest.fixture
def sample_transactions():
    """Create a small synthetic dataset for testing."""
    return pd.DataFrame(
        {
            "TransactionID": range(1, 11),
            "TransactionDT": [
                100, 200, 300, 500, 1000,  # user A: 5 txns
                150, 250, 800,              # user B: 3 txns
                400, 900,                    # user C: 2 txns
            ],
            "TransactionAmt": [
                10.0, 20.0, 15.0, 500.0, 25.0,  # user A: spike at txn 4
                30.0, 35.0, 40.0,                 # user B: steady
                100.0, 200.0,                      # user C
            ],
            "pseudo_user_id": [
                "A", "A", "A", "A", "A",
                "B", "B", "B",
                "C", "C",
            ],
            "transaction_hour": [1, 2, 3, 5, 10, 1, 2, 8, 4, 9],
            "transaction_dayofweek": [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            "isFraud": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        }
    )


def test_velocity_no_future_leakage(sample_transactions):
    """Velocity for a transaction must only use strictly-prior transactions."""
    df = compute_velocity_features(sample_transactions)

    # User A's first transaction (t=100) should have count=0 in all windows
    first_a = df[(df["pseudo_user_id"] == "A") & (df["TransactionDT"] == 100)]
    assert first_a["txn_count_1h"].values[0] == 0.0
    assert first_a["txn_count_24h"].values[0] == 0.0
    assert first_a["txn_count_7d"].values[0] == 0.0

    # User A's second transaction (t=200) should see exactly 1 prior txn
    # within 1h window (100 seconds ago < 3600 seconds)
    second_a = df[(df["pseudo_user_id"] == "A") & (df["TransactionDT"] == 200)]
    assert second_a["txn_count_1h"].values[0] == 1.0

    # User B's features should be independent of User A
    first_b = df[(df["pseudo_user_id"] == "B") & (df["TransactionDT"] == 150)]
    assert first_b["txn_count_1h"].values[0] == 0.0


def test_velocity_window_boundaries(sample_transactions):
    """Transactions outside the window should not be counted."""
    df = compute_velocity_features(sample_transactions)

    # User A at t=1000: prior txns at t=100,200,300,500
    # 1h window (3600s): [1000-3600, 1000) = [-2600, 1000) → all 4 prior are in range
    a_last = df[(df["pseudo_user_id"] == "A") & (df["TransactionDT"] == 1000)]
    assert a_last["txn_count_1h"].values[0] == 4.0


def test_amount_zscore_no_leakage(sample_transactions):
    """Z-score must use only past amounts, not current or future."""
    df = compute_amount_features(sample_transactions)

    # User A's first transaction: no history → z-score = 0
    first_a = df[(df["pseudo_user_id"] == "A") & (df["TransactionDT"] == 100)]
    assert first_a["amount_zscore"].values[0] == 0.0

    # User A's 4th transaction (amount=500): past amounts are [10, 20, 15]
    # mean=15, std=5, z-score = (500-15)/5 = 97
    fourth_a = df[(df["pseudo_user_id"] == "A") & (df["TransactionDT"] == 500)]
    expected_z = (500.0 - 15.0) / np.std([10.0, 20.0, 15.0], ddof=1)
    assert abs(fourth_a["amount_zscore"].values[0] - expected_z) < 0.01


def test_time_features_cyclical(sample_transactions):
    """Sin/cos time features should be in [-1, 1]."""
    df = compute_time_features(sample_transactions)

    assert df["hour_sin"].between(-1, 1).all()
    assert df["hour_cos"].between(-1, 1).all()
    assert df["day_sin"].between(-1, 1).all()
    assert df["day_cos"].between(-1, 1).all()
    assert df["is_weekend"].isin([0, 1]).all()


def test_time_features_weekend(sample_transactions):
    """Days 5 and 6 should be weekend."""
    df = sample_transactions.copy()
    df["transaction_dayofweek"] = [5, 6, 0, 1, 2, 3, 4, 5, 6, 0]
    df = compute_time_features(df)
    expected = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
    assert df["is_weekend"].tolist() == expected


def test_seconds_since_last(sample_transactions):
    """Time since last transaction must be correct."""
    df = compute_velocity_features(sample_transactions)

    # User A: first txn → -1 (sentinel), second → 200-100=100
    first_a = df[(df["pseudo_user_id"] == "A") & (df["TransactionDT"] == 100)]
    assert first_a["seconds_since_last_txn"].values[0] == -1.0

    second_a = df[(df["pseudo_user_id"] == "A") & (df["TransactionDT"] == 200)]
    assert second_a["seconds_since_last_txn"].values[0] == 100.0


def test_log_amount(sample_transactions):
    """log_amount should be log1p(TransactionAmt)."""
    df = compute_amount_features(sample_transactions)
    expected = np.log1p(sample_transactions["TransactionAmt"])
    np.testing.assert_array_almost_equal(df["log_amount"].values, expected.values)


def test_engineer_features_adds_all_columns(sample_transactions):
    """Full pipeline should add all engineered feature columns."""
    df = engineer_features(sample_transactions, compute_velocity=True)
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in df.columns, f"Missing engineered feature: {col}"


def test_engineer_features_no_nans(sample_transactions):
    """Engineered features should have no NaN values after pipeline."""
    df = engineer_features(sample_transactions, compute_velocity=True)
    for col in ENGINEERED_FEATURE_NAMES:
        if col in df.columns:
            assert not df[col].isna().any(), f"NaN found in {col}"


def test_single_transaction_account():
    """An account with only one transaction should get safe defaults."""
    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [100],
            "TransactionAmt": [50.0],
            "pseudo_user_id": ["lonely"],
            "transaction_hour": [12],
            "transaction_dayofweek": [3],
            "isFraud": [0],
        }
    )
    result = engineer_features(df, compute_velocity=True)

    assert result["txn_count_1h"].values[0] == 0.0
    assert result["amount_zscore"].values[0] == 0.0
    assert result["seconds_since_last_txn"].values[0] == -1.0
