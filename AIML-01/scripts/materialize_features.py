"""Apply Feast definitions, materialize actual dataset dates, verify Redis readback.

Run: python -m scripts.materialize_features
"""

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from feature_repo.definitions import transaction, transaction_features
from src.config import load_config
from src.features.engine import ENGINEERED_FEATURE_NAMES
from src.features.store import feature_store, online_features


def main():
    cfg = load_config()
    repo = (cfg.project_root / cfg.feast_repo_path).resolve()
    source = pd.read_parquet(repo / "data" / "transaction_features.parquet")
    store = feature_store(cfg)
    store.apply([transaction, transaction_features])
    start = source.event_timestamp.min().to_pydatetime() - timedelta(seconds=1)
    end = source.event_timestamp.max().to_pydatetime() + timedelta(seconds=1)
    store.materialize(start, end, feature_views=["transaction_features"])
    sample = source.sample(min(5000, len(source)), random_state=cfg.seed)
    actual = online_features(store, sample)
    expected = sample[ENGINEERED_FEATURE_NAMES].to_numpy()
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-8, atol=1e-8)
    report = {
        "materialized_rows": len(source),
        "verified_rows": len(sample),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "max_abs_error": float(np.max(np.abs(actual.to_numpy() - expected))),
        "features": ENGINEERED_FEATURE_NAMES,
    }
    output = cfg.data_processed_dir / "features" / "materialization.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
