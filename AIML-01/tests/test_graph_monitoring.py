import numpy as np
import pandas as pd
import pytest
import torch

from src.graph.build import build_graph, sample_neighbors
from src.monitoring.drift import detect_drift
from src.training.calibration import fit_calibrator, tune_blend
from src.training.retrain import promotion_decision


def test_graph_hubs_and_labels_never_enter_features():
    frame = pd.DataFrame(
        {
            "pseudo_user_id": ["a", "b", "c"],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "TransactionDT": [1.0, 2.0, 3.0],
            "isFraud": [0, 1, 0],
            "card1": [1, 1, 1],
            "DeviceInfo": ["x", "x", "y"],
        }
    )
    graph, accounts, audit = build_graph(frame, hub_cap=2)
    assert "card" not in graph.node_types
    assert audit["training_max_transaction_dt"] == 3
    altered, _, _ = build_graph(frame.assign(isFraud=1), hub_cap=2)
    assert torch.equal(graph["account"].x, altered["account"].x)
    assert accounts == ["a", "b", "c"]


def test_neighbor_sampler_caps_each_destination_and_is_reproducible():
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 6, 6, 6, 7, 7]])
    first = sample_neighbors(edges, fanout=2, seed=42)
    assert first.shape[1] == 4
    assert torch.equal(first, sample_neighbors(edges, 2, 42))
    assert np.bincount(first[1].numpy()).max() <= 2


def test_calibration_and_validation_gate():
    p = np.array([0.05, 0.1, 0.8, 0.9] * 20)
    y = np.array([0, 0, 1, 1] * 20)
    parameters = fit_calibrator(p, y)
    assert parameters["coef"] > 0
    weight, scores = tune_blend(p, 1 - p, y)
    assert scores[str(weight)] == max(scores.values())
    assert not promotion_decision(0.5, 0.5005)
    assert not promotion_decision(0.5, np.nan)
    assert promotion_decision(0.5, 0.51)


def test_evidently_unchanged_and_shifted_data(tmp_path):
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"TransactionAmt": rng.lognormal(3, 0.5, 300)})
    unchanged = detect_drift(reference, reference.copy(), tmp_path / "same")
    shifted = detect_drift(reference, reference * 10, tmp_path / "shift")
    assert not unchanged["drift_detected"]
    assert shifted["drift_detected"]
    assert (tmp_path / "shift" / "report.html").exists()
    with pytest.raises(ValueError, match="100"):
        detect_drift(reference.iloc[:2], reference, tmp_path / "small")
