"""Controlled unchanged/shifted replay -> Evidently -> genuine challenger gate."""

import argparse
import json
import sys

import pandas as pd

from src.config import load_config
from src.monitoring.drift import MONITORED_COLUMNS, detect_drift
from src.streaming.producer import load_transactions
from src.training.retrain import retrain


def main(device="cuda"):
    cfg = load_config()
    current = pd.DataFrame(load_transactions(2000))[MONITORED_COLUMNS]
    folder = cfg.data_processed_dir / "monitoring"
    unchanged = detect_drift(current, current.copy(), folder / "controlled_unchanged")
    shifted = current.copy()
    shifted["TransactionAmt"] *= 10
    drifted = detect_drift(current, shifted, folder / "controlled_drift")
    assert not unchanged["drift_detected"]
    assert drifted["drift_detected"]
    decision = retrain(device)
    result = {
        "controlled_demo": True,
        "unchanged": unchanged,
        "shifted": drifted,
        "retraining": decision,
        "note": (
            "Amounts are multiplied by ten; this demonstrates the trigger, "
            "not a claim of production drift."
        ),
    }
    (folder / "demo.json").write_text(json.dumps(result, indent=2))
    (folder / "controlled_drift" / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    main(args.device)
