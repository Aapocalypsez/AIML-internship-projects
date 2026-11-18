"""Scheduled inference with the evaluated GNN; never replace the champion blindly."""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from feature_repo.definitions import account, account_ring_risk
from src.config import load_config
from src.dashboard.db import migrate, store_rings
from src.features.store import feature_store
from src.graph.build import build_graph
from src.graph.train import GraphSAGE
from src.training.data_loader import build_pseudo_user_key, merge_data


def main():
    cfg = load_config()
    torch.set_num_threads(2)
    folder = cfg.data_processed_dir / "graph"
    checkpoint = torch.load(folder / "graphsage.pt", map_location="cpu", weights_only=True)
    cutoff = checkpoint["audit"]["training_max_transaction_dt"]
    columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
        "card1",
        "addr1",
        "D1",
        "P_emaildomain",
    ]
    rows = pd.read_csv(cfg.data_raw_dir / "train_transaction.csv", usecols=columns)
    rows = rows[rows.TransactionDT <= cutoff]
    identity = pd.read_csv(
        cfg.data_raw_dir / "train_identity.csv", usecols=["TransactionID", "DeviceInfo"]
    )
    rows = build_pseudo_user_key(merge_data(rows, identity))
    graph, accounts, audit = build_graph(rows, checkpoint["audit"]["hub_cap"])
    assert accounts == checkpoint["accounts"]
    assert audit["training_max_transaction_dt"] == cutoff
    homogeneous = graph.to_homogeneous(node_attrs=["x"])
    model = GraphSAGE()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        scores = model(homogeneous.x, homogeneous.edge_index).sigmoid()[: len(accounts)].numpy()
    source = cfg.project_root / cfg.feast_repo_path / "data" / "account_risk.parquet"
    frozen = pd.read_parquet(source).set_index("pseudo_user_id").loc[accounts].reset_index()
    # A changed graph/model requires validation; do not mix new scores with an old calibrator.
    np.testing.assert_allclose(scores, frozen.ring_score.to_numpy(), rtol=1e-4, atol=1e-5)
    store = feature_store(cfg)
    store.apply([account, account_ring_risk])
    timestamp = frozen.event_timestamp.iloc[0].to_pydatetime()
    store.materialize(
        timestamp - timedelta(seconds=1),
        timestamp + timedelta(seconds=1),
        feature_views=["account_ring_risk"],
    )
    migrate()
    store_rings(frozen)
    result = {
        "scored_accounts": len(accounts),
        "cutoff": cutoff,
        "max_abs_error_vs_evaluated_snapshot": float(np.max(np.abs(scores - frozen.ring_score))),
        "champion_changed": False,
    }
    (folder / "batch_scoring.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
