"""Portable calibrated model bundle: exact evaluated XGBoost + batch GNN lookup."""

import json

import mlflow.pyfunc
import numpy as np
import xgboost as xgb


def calibrate(probability, parameters):
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    logit = np.log(probability / (1 - probability))
    value = np.clip(parameters["coef"] * logit + parameters["intercept"], -40, 40)
    return 1 / (1 + np.exp(-value))


class FraudModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.metadata = json.loads(open(context.artifacts["metadata"], encoding="utf-8").read())
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts["xgboost"])
        self.booster.set_param({"device": "cpu", "nthread": 1})

    def components(self, frame):
        raw = self.booster.inplace_predict(
            frame[self.metadata["feature_columns"]].to_numpy(dtype=np.float32),
            iteration_range=(0, self.metadata["best_iteration"] + 1),
        )
        xgb_score = calibrate(raw, self.metadata["xgb_calibration"])
        ring = (
            frame["ring_score"].to_numpy()
            if "ring_score" in frame
            else np.full(len(frame), self.metadata["ring_prior"])
        )
        ring_score = calibrate(ring, self.metadata["gnn_calibration"])
        weight = self.metadata["xgb_weight"]
        return raw, xgb_score, ring_score, weight * xgb_score + (1 - weight) * ring_score

    def predict(self, context, model_input, params=None):
        return self.components(model_input)[-1]

    def explain(self, frame):
        columns = self.metadata["feature_columns"]
        values = self.booster.predict(
            xgb.DMatrix(frame[columns], feature_names=columns),
            pred_contribs=True,
            iteration_range=(0, self.metadata["best_iteration"] + 1),
        )[0, :-1]
        return [
            {"feature": columns[i], "log_odds_contribution": float(values[i])}
            for i in np.argsort(np.abs(values))[-3:][::-1]
        ]
