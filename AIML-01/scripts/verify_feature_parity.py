"""Replay 6,000 real transactions; compare incremental features to saved batch snapshots.

Run after training and materialization: python -m scripts.verify_feature_parity
"""

import json

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.engine import ENGINEERED_FEATURE_NAMES, OnlineFeatureEngine
from src.features.store import feature_store, online_features
from src.training.data_loader import prepare_dataset


def main():
    cfg = load_config()
    data = prepare_dataset(cfg)
    # Use the same stable arrival order as training, including any equal-time rows.
    raw = data["train"].iloc[:6000].copy()
    del data
    columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "pseudo_user_id",
        "transaction_hour",
        "transaction_dayofweek",
        "DeviceType",
        "P_emaildomain",
    ]
    raw = raw[[column for column in columns if column in raw]]
    engine = OnlineFeatureEngine()
    replay = pd.DataFrame([engine.process(row) for row in raw.to_dict("records")])
    replay = replay[ENGINEERED_FEATURE_NAMES].to_numpy()
    repo = (cfg.project_root / cfg.feast_repo_path).resolve()
    saved = pd.read_parquet(repo / "data" / "transaction_features.parquet")
    batch = saved.set_index("TransactionID").loc[raw.TransactionID, ENGINEERED_FEATURE_NAMES]
    online = online_features(feature_store(cfg), raw)
    for values in [batch, online]:
        np.testing.assert_allclose(values.to_numpy(), replay, rtol=1e-8, atol=1e-8)
    report = {
        "transactions": len(raw),
        "features": len(ENGINEERED_FEATURE_NAMES),
        "rtol": 1e-8,
        "atol": 1e-8,
        "batch_vs_replay_max_abs_error": float(np.max(np.abs(batch.to_numpy() - replay))),
        "redis_vs_replay_max_abs_error": float(np.max(np.abs(online.to_numpy() - replay))),
        "first_transaction_id": int(raw.TransactionID.iloc[0]),
        "last_transaction_id": int(raw.TransactionID.iloc[-1]),
    }
    path = cfg.data_processed_dir / "features" / "parity.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
