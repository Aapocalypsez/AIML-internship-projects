"""Initialize a fresh deployment using the already evaluated portable artifacts."""

import json
from datetime import timedelta

import pandas as pd

from feature_repo.definitions import account, account_ring_risk
from src.config import load_config
from src.dashboard.db import migrate, store_rings
from src.features.store import feature_store
from src.training.registry import bootstrap, log_bundle


def main():
    cfg = load_config()
    migrate()
    graph_result = cfg.data_processed_dir / "graph" / "results.json"
    if not graph_result.exists():
        bootstrap()
        return
    result = json.loads(graph_result.read_text())
    store = feature_store(cfg)
    store.apply([account, account_ring_risk])
    source = cfg.project_root / cfg.feast_repo_path / "data" / "account_risk.parquet"
    rows = pd.read_parquet(source)
    timestamp = rows.event_timestamp.iloc[0].to_pydatetime()
    store.materialize(
        timestamp - timedelta(seconds=1),
        timestamp + timedelta(seconds=1),
        feature_views=["account_ring_risk"],
    )
    store_rings(rows)
    registered = log_bundle(
        cfg.data_processed_dir / "features" / "model.ubj",
        result["metadata"],
        {f"{split}_pr_auc": metrics["blend"] for split, metrics in result["ablation"].items()},
        "deployment_bootstrap",
        True,
    )
    print(json.dumps(registered))


if __name__ == "__main__":
    main()
