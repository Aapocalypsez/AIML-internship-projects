"""Validation-only probability calibration and blend selection."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def fit_calibrator(probabilities, labels):
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(np.log(p / (1 - p)).reshape(-1, 1), labels)
    return {"coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def tune_blend(xgb_scores, gnn_scores, labels):
    scores = {
        str(weight): float(
            average_precision_score(labels, weight * xgb_scores + (1 - weight) * gnn_scores)
        )
        for weight in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    }
    best = max(scores, key=scores.get)
    return float(best), scores
