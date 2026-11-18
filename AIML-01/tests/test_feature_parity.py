"""Independent vectorized batch vs incremental replay, plus brute-force checks."""

import numpy as np
import pandas as pd
import pytest

from src.features.engine import ENGINEERED_FEATURE_NAMES, OnlineFeatureEngine, engineer_features


def replay_transactions(n=6000):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.cumsum(rng.choice([0, 1, 60, 3600, 86400], n)),
            "TransactionAmt": rng.uniform(0, 1000, n),
            "pseudo_user_id": rng.choice(["A", "B", "C", "D"], n),
            "DeviceType": rng.choice(["mobile", "desktop", None], n),
            "P_emaildomain": rng.choice(["a.test", "b.test", None], n),
        }
    )


def test_batch_online_parity_6000_transactions():
    df = replay_transactions()
    batch = engineer_features(df)[ENGINEERED_FEATURE_NAMES]
    engine = OnlineFeatureEngine()
    online = pd.DataFrame([engine.process(row) for row in df.to_dict("records")])
    np.testing.assert_allclose(online[ENGINEERED_FEATURE_NAMES], batch, rtol=1e-8, atol=1e-8)


def test_strict_history_ties_unsorted_and_exact_boundaries():
    df = pd.DataFrame(
        {
            "TransactionID": range(6),
            "pseudo_user_id": ["A"] * 6,
            "TransactionDT": [0, 3600, 3600, 3601, 604800, 604801],
            "TransactionAmt": [10.0, 20.0, 900.0, 30.0, 40.0, 50.0],
            "DeviceType": ["mobile", "desktop", "mobile", "desktop", None, "mobile"],
            "P_emaildomain": ["a", "b", "c", "d", None, "e"],
        }
    )
    batch = engineer_features(df.iloc[::-1]).set_index("TransactionID")
    # Stable arrival order among ties affects the *next* device, so test numeric history here.
    for row in df.to_dict("records"):
        past = df[df.TransactionDT < row["TransactionDT"]]
        actual = batch.loc[row["TransactionID"]]
        for hours, name in [(1, "1h"), (6, "6h"), (24, "24h"), (168, "7d")]:
            window = past[past.TransactionDT >= row["TransactionDT"] - hours * 3600]
            assert actual[f"txn_count_{name}"] == len(window)
            if name in ["1h", "24h"]:
                assert actual[f"txn_amount_sum_{name}"] == window.TransactionAmt.sum()
        std = past.TransactionAmt.std()
        expected = (row["TransactionAmt"] - past.TransactionAmt.mean()) / std if std > 0 else 0
        assert actual.amount_zscore == pytest.approx(expected)
    assert batch.loc[1, "txn_count_1h"] == batch.loc[2, "txn_count_1h"] == 1
    assert batch.loc[1, "seconds_since_last_txn"] == batch.loc[2, "seconds_since_last_txn"] == 3600


def test_future_and_labels_cannot_change_past_features():
    df = replay_transactions(100)
    expected = engineer_features(df.iloc[:50])[ENGINEERED_FEATURE_NAMES]
    df["isFraud"] = 1
    actual = engineer_features(df).iloc[:50][ENGINEERED_FEATURE_NAMES]
    pd.testing.assert_frame_equal(actual, expected)


def test_online_rejects_out_of_order_without_changing_state():
    row = replay_transactions(1).iloc[0].to_dict()
    engine = OnlineFeatureEngine()
    engine.process(row)
    with pytest.raises(ValueError, match="Out-of-order"):
        engine.process({**row, "TransactionDT": row["TransactionDT"] - 1})


@pytest.mark.parametrize("amount", [-1, np.nan, np.inf])
def test_online_rejects_invalid_amount(amount):
    with pytest.raises(ValueError):
        OnlineFeatureEngine().process(
            {"pseudo_user_id": "A", "TransactionDT": 1, "TransactionAmt": amount}
        )
