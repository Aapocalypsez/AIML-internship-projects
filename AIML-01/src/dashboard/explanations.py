"""Compute SHAP for a reviewed case using its exact immutable model version."""

import os
from functools import lru_cache

import mlflow
import pandas as pd

from src.config import load_config


@lru_cache(maxsize=4)
def explanation_model(version):
    mlflow.set_tracking_uri(load_config().mlflow_tracking_uri)
    name = os.getenv("MODEL_NAME", "fraud-detector")
    return mlflow.pyfunc.load_model(f"models:/{name}/{version}").unwrap_python_model()


def explain_case(result):
    model = explanation_model(result["model_version"])
    values = {
        name: result["transaction"].get(name, float("nan"))
        for name in model.metadata["feature_columns"]
    }
    values.update(result["features"])
    frame = pd.DataFrame([values], columns=model.metadata["feature_columns"]).astype("float32")
    return model.explain(frame)
