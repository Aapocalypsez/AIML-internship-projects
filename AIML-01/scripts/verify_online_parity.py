"""Replay actual producer records through persisted scoring, then compare batch features."""

import json
from uuid import uuid4

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.engine import ENGINEERED_FEATURE_NAMES
from src.serving.scorer import Scorer
from src.streaming.producer import load_transactions


def main():
    cfg = load_config()
    rows = load_transactions(6000)
    namespace = "fraud:parity:" + uuid4().hex
    scorer = Scorer(namespace=namespace)
    values = [scorer.score(row)["features"] for row in rows]
    actual = pd.DataFrame(values)[ENGINEERED_FEATURE_NAMES]
    path = cfg.project_root / cfg.feast_repo_path / "data" / "transaction_features.parquet"
    expected = (
        pd.read_parquet(path)
        .set_index("TransactionID")
        .loc[[row["TransactionID"] for row in rows], ENGINEERED_FEATURE_NAMES]
    )
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-8, atol=1e-8)
    report = {
        "transactions": len(rows),
        "features": len(ENGINEERED_FEATURE_NAMES),
        "path": "producer -> validation -> Redis-restored Scorer -> persisted history",
        "model_version": scorer.version,
        "namespace": namespace,
        "max_abs_error": float(np.max(np.abs(actual.to_numpy() - expected.to_numpy()))),
        "rtol": 1e-8,
        "atol": 1e-8,
    }
    output = cfg.data_processed_dir / "integration"
    actual.to_parquet(output / "online_parity_features.parquet", index=False)
    (output / "online_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
