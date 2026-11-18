"""Train GraphSAGE on past data; calibrate/tune on validation; evaluate test once."""

import argparse
import json
import sys
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv

from src.config import load_config
from src.dashboard.db import migrate
from src.features.store import feature_store
from src.graph.build import build_graph, sample_neighbors
from src.serving.model import calibrate
from src.training.calibration import fit_calibrator, tune_blend
from src.training.data_loader import build_pseudo_user_key, merge_data
from src.training.matrices import prepare_matrices
from src.training.registry import log_bundle


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        self.first = SAGEConv(input_dim, hidden_dim)
        self.second = SAGEConv(hidden_dim, 1)

    def forward(self, x, first_edges, second_edges=None):
        hidden = self.first(x, first_edges).relu()
        return self.second(hidden, first_edges if second_edges is None else second_edges).squeeze(
            -1
        )


def train(epochs=8, materialize=True):
    cfg = load_config()
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print("Loading data for a training-only graph...", flush=True)
    directory, cache = prepare_matrices()
    needed = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
        "card1",
        "addr1",
        "D1",
        "P_emaildomain",
    ]
    transactions = pd.read_csv(cfg.data_raw_dir / "train_transaction.csv", usecols=needed)
    transactions = transactions[transactions.TransactionDT <= cache["ranges"]["train"]["max"]]
    identity_path = cfg.data_raw_dir / "train_identity.csv"
    identity = (
        pd.read_csv(identity_path, usecols=["TransactionID", "DeviceInfo"])
        if identity_path.exists()
        else pd.DataFrame()
    )
    train_frame = build_pseudo_user_key(merge_data(transactions, identity))
    small_columns = ["TransactionID", "TransactionDT", "pseudo_user_id", "isFraud"]
    val = pd.read_parquet(directory / "val.parquet", columns=small_columns)
    test = pd.read_parquet(directory / "test.parquet", columns=small_columns)
    assert len(train_frame) == cache["ranges"]["train"]["rows"]
    del transactions, identity
    heterogeneous, accounts, audit = build_graph(train_frame)
    assert audit["training_max_transaction_dt"] < val.TransactionDT.min()
    assert val.TransactionDT.max() < test.TransactionDT.min()
    graph = heterogeneous.to_homogeneous(node_attrs=["x"])
    labels = heterogeneous["account"].y
    account_count = len(accounts)
    model = GraphSAGE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    positive_weight = (len(labels) - labels.sum()) / labels.sum().clamp(min=1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        first_edges = sample_neighbors(graph.edge_index, 8, cfg.seed + epoch)
        second_edges = sample_neighbors(graph.edge_index, 8, cfg.seed + 100 + epoch)
        logits = model(graph.x, first_edges, second_edges)[:account_count]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"Graph epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        account_scores = model(graph.x, graph.edge_index).sigmoid()[:account_count].numpy()
    score_map = pd.Series(account_scores, index=accounts)
    prior = float(train_frame.isFraud.mean())
    output = cfg.data_processed_dir / "graph"
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((cfg.data_processed_dir / "features" / "results.json").read_text())
    predictions = {}
    for name, frame in [("val", val), ("test", test)]:
        saved = pd.read_parquet(cfg.data_processed_dir / "features" / f"{name}_predictions.parquet")
        saved = saved.set_index("TransactionID").loc[frame.TransactionID]
        np.testing.assert_array_equal(saved.isFraud, frame.isFraud)
        predictions[name] = {
            "xgb": saved.fraud_probability.to_numpy(),
            "gnn": frame.pseudo_user_id.map(score_map).fillna(prior).to_numpy(),
            "labels": frame.isFraud.to_numpy(),
        }
    # First half calibrates; second half chooses the weight. Test is untouched.
    middle = len(val) // 2
    xgb_calibration = fit_calibrator(predictions["val"]["xgb"][:middle], val.isFraud.iloc[:middle])
    gnn_calibration = fit_calibrator(predictions["val"]["gnn"][:middle], val.isFraud.iloc[:middle])
    for name in predictions:
        predictions[name]["xgb"] = calibrate(predictions[name]["xgb"], xgb_calibration)
        predictions[name]["gnn"] = calibrate(predictions[name]["gnn"], gnn_calibration)
    weight, candidates = tune_blend(
        predictions["val"]["xgb"][middle:],
        predictions["val"]["gnn"][middle:],
        val.isFraud.iloc[middle:],
    )
    validation_blend = weight * predictions["val"]["xgb"] + (1 - weight) * predictions["val"]["gnn"]
    threshold = float(np.quantile(validation_blend[middle:], 0.99))
    ablation = {}
    for name, values in predictions.items():
        blended = weight * values["xgb"] + (1 - weight) * values["gnn"]
        ablation[name] = {
            "xgboost": float(average_precision_score(values["labels"], values["xgb"])),
            "gnn": float(average_precision_score(values["labels"], values["gnn"])),
            "blend": float(average_precision_score(values["labels"], blended)),
            "account_coverage": float(
                (val if name == "val" else test).pseudo_user_id.isin(accounts).mean()
            ),
        }
        pd.DataFrame(
            {
                "TransactionID": (val if name == "val" else test).TransactionID,
                "xgb_score": values["xgb"],
                "gnn_score": values["gnn"],
                "blend_score": blended,
                "isFraud": values["labels"],
            }
        ).to_parquet(output / f"{name}_ablation.parquet", index=False)
    adjacency = coo_matrix(
        (np.ones(graph.edge_index.shape[1]), graph.edge_index.numpy()),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    _, components = connected_components(adjacency, directed=False)
    account_components = components[:account_count]
    sizes = pd.Series(account_components).value_counts()
    snapshot_dt = float(train_frame.TransactionDT.max())
    snapshot_time = pd.Timestamp("2017-12-01", tz="UTC") + pd.Timedelta(seconds=snapshot_dt)
    rings = pd.DataFrame(
        {
            "pseudo_user_id": accounts,
            "ring_score": account_scores.astype(float),
            "ring_id": [f"ring-{value}" for value in account_components],
            "ring_size": [int(sizes[value]) for value in account_components],
            "snapshot_transaction_dt": snapshot_dt,
            "event_timestamp": snapshot_time,
        }
    )
    rings.to_parquet(output / "account_risk.parquet", index=False)
    source = cfg.project_root / cfg.feast_repo_path / "data" / "account_risk.parquet"
    rings.to_parquet(source, index=False)
    torch.save(
        {"model_state": model.state_dict(), "accounts": accounts, "audit": audit},
        output / "graphsage.pt",
    )
    metadata = {
        "feature_columns": reference["feature_columns"],
        "best_iteration": reference["best_iteration"],
        "xgb_calibration": xgb_calibration,
        "gnn_calibration": gnn_calibration,
        "xgb_weight": weight,
        "ring_prior": prior,
        "threshold": threshold,
        "source_run_id": reference["run_id"],
        "graph_enabled": True,
        "graph_snapshot_dt": snapshot_dt,
        "validation_calibration_rows": middle,
    }
    registered = log_bundle(
        cfg.data_processed_dir / "features" / "model.ubj",
        metadata,
        {
            f"{split}_{name}_pr_auc": value
            for split, metrics in ablation.items()
            for name, value in metrics.items()
            if name != "account_coverage"
        },
        "phase4_calibrated_blend",
        False,
    )
    result = {
        "ablation": ablation,
        "xgb_weight": weight,
        "weight_selection_pr_auc": candidates,
        "calibration_rows": middle,
        "selection_rows": len(val) - middle,
        "graph_audit": audit,
        "epochs": epochs,
        "losses": losses,
        "registry": registered,
        "metadata": metadata,
        "leakage_check": (
            "PASS: graph edges/features/labels end before validation; no test tuning"
        ),
        "calibration_note": (
            "Validation includes calibration rows; test is the held-out comparison"
        ),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    mlflow.MlflowClient().log_artifact(registered["run_id"], str(output / "results.json"))
    if materialize:
        from feature_repo.definitions import account, account_ring_risk

        store = feature_store(cfg)
        store.apply([account, account_ring_risk])
        store.materialize(
            snapshot_time.to_pydatetime() - timedelta(seconds=1),
            snapshot_time.to_pydatetime() + timedelta(seconds=1),
            feature_views=["account_ring_risk"],
        )
        migrate()
        from src.dashboard.db import store_rings

        store_rings(rings)
    if materialize:
        mlflow.MlflowClient().set_registered_model_alias(
            registered["name"], "champion", registered["version"]
        )
    print(json.dumps({"ablation": ablation, "weight": weight, "registry": registered}, indent=2))
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train(args.epochs)
