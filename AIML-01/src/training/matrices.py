"""Compact, ID-aligned training cache for scheduled jobs on a small machine."""

import json

import pandas as pd

from src.config import load_config
from src.features.engine import ENGINEERED_FEATURE_NAMES
from src.training.data_loader import prepare_dataset


def prepare_matrices():
    cfg = load_config()
    directory = cfg.data_processed_dir / "training"
    expected = [directory / f"{split}.parquet" for split in ["train", "val", "test"]]
    metadata_path = directory / "metadata.json"
    if all(path.exists() for path in expected) and metadata_path.exists():
        return directory, json.loads(metadata_path.read_text())
    reference = json.loads((cfg.data_processed_dir / "features" / "results.json").read_text())
    snapshot = pd.read_parquet(
        cfg.project_root / cfg.feast_repo_path / "data" / "transaction_features.parquet"
    )
    snapshot = snapshot.set_index("TransactionID")
    data = prepare_dataset(cfg)
    columns = reference["feature_columns"]
    directory.mkdir(parents=True, exist_ok=True)
    ranges = {}
    for name in ["train", "val", "test"]:
        part = data.pop(name)
        matrix = part[[c for c in columns if c not in ENGINEERED_FEATURE_NAMES]].astype("float32")
        engineered = snapshot.loc[part.TransactionID, ENGINEERED_FEATURE_NAMES].copy()
        engineered.index = part.index
        matrix[ENGINEERED_FEATURE_NAMES] = engineered.astype("float32")
        for column in ["TransactionID", "TransactionDT", "pseudo_user_id", "isFraud"]:
            matrix[column] = part[column]
        matrix.to_parquet(directory / f"{name}.parquet", index=False)
        ranges[name] = {
            "min": float(part.TransactionDT.min()),
            "max": float(part.TransactionDT.max()),
            "rows": len(part),
        }
        del part, matrix, engineered
    metadata = {
        "feature_columns": columns,
        "ranges": ranges,
        "source_run_id": reference["run_id"],
        "input_hashes": reference["inputs_sha256"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return directory, metadata


if __name__ == "__main__":
    print(prepare_matrices()[0])
