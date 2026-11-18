"""Training-only heterogeneous account/identifier graph with hub exclusion."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_graph(transactions, hub_cap=20):
    if transactions.empty:
        raise ValueError("Graph needs training transactions")
    grouped = transactions.groupby("pseudo_user_id", sort=True)
    accounts = grouped.TransactionAmt.agg(["count", "mean", "max", "std"]).fillna(0)
    features = np.zeros((len(accounts), 6), dtype="float32")
    values = np.log1p(accounts.to_numpy(dtype="float32"))
    features[:, :4] = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)
    features[:, 4] = 1
    graph = HeteroData()
    graph["account"].x = torch.from_numpy(features)
    graph["account"].y = torch.tensor(
        grouped.isFraud.max().loc[accounts.index].to_numpy(), dtype=torch.float32
    )
    node_map = pd.Series(np.arange(len(accounts)), index=accounts.index)
    audit = {
        "training_max_transaction_dt": float(transactions.TransactionDT.max()),
        "transactions": len(transactions),
        "accounts": len(accounts),
        "hub_cap": hub_cap,
        "identifier_types": {},
        "labels_used_as_features": False,
    }
    for kind, column in [
        ("card", "card1"),
        ("device", "DeviceInfo"),
        ("email", "P_emaildomain"),
        ("address", "addr1"),
    ]:
        if column not in transactions:
            continue
        edges = transactions[["pseudo_user_id", column]].dropna().drop_duplicates()
        degree = edges.groupby(column).size()
        allowed = degree[degree <= hub_cap].index
        edges = edges[edges[column].isin(allowed)]
        if edges.empty:
            continue
        entity_codes, entities = pd.factorize(edges[column], sort=True)
        entity_x = torch.zeros((len(entities), 6))
        entity_x[:, 5] = 1
        graph[kind].x = entity_x
        edge_index = torch.tensor(
            np.stack([edges.pseudo_user_id.map(node_map).to_numpy(), entity_codes]),
            dtype=torch.long,
        )
        graph["account", "uses_" + kind, kind].edge_index = edge_index
        graph[kind, "rev_uses_" + kind, "account"].edge_index = edge_index.flip(0)
        audit["identifier_types"][kind] = {
            "nodes": len(entities),
            "edges": len(edges),
            "excluded_hubs": int((degree > hub_cap).sum()),
        }
    return graph, accounts.index.tolist(), audit


def sample_neighbors(edge_index, fanout=8, seed=42):
    """Uniform bounded fanout by destination; no optional compiled sampler required."""
    edges = edge_index.numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(edges.shape[1])
    order = order[np.argsort(edges[1, order], kind="stable")]
    destinations = edges[1, order]
    starts = np.r_[0, np.flatnonzero(np.diff(destinations)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    return edge_index[:, order[rank < fanout]]
