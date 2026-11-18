"""Catch dependency incompatibilities that only surface when saving a fitted model."""

import numpy as np
import pandas as pd
import xgboost as xgb


def test_xgboost_model_save_load_preserves_predictions(tmp_path):
    features = pd.DataFrame({"amount": np.arange(20, dtype="float32")})
    labels = np.array([0, 1] * 10)
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, random_state=42)
    model.fit(features, labels)
    path = tmp_path / "model.ubj"
    model.save_model(path)
    restored = xgb.XGBClassifier()
    restored.load_model(path)
    np.testing.assert_allclose(model.predict_proba(features), restored.predict_proba(features))
