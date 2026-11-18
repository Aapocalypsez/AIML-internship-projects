"""Sequential real-HTTP latency, with fresh reserved IDs and recorded scope."""

import argparse
import json
import time

import numpy as np
import requests

from src.config import load_config
from src.streaming.producer import load_transactions


def main(url="http://127.0.0.1:8000", count=1000):
    deadline = time.monotonic() + 120
    while True:
        try:
            requests.get(url + "/health", timeout=3).raise_for_status()
            break
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    rows = load_transactions(count)
    offset = 10**13 + int(time.time()) * 10000
    elapsed, scoring = [], []
    with requests.Session() as session:
        for row in rows:
            row["TransactionID"] += offset
            row["pseudo_user_id"] = f"load-{offset}:" + row["pseudo_user_id"]
            row["demo_run"] = "load-test"
            started = time.perf_counter()
            response = session.post(url + "/score", json=row, timeout=30)
            response.raise_for_status()
            elapsed.append((time.perf_counter() - started) * 1000)
            scoring.append(response.json()["latency_ms"])
    cfg = load_config()
    result = {
        "url": url,
        "requests": len(rows),
        "http_round_trip": {f"p{q}_ms": float(np.percentile(elapsed, q)) for q in [50, 95, 99]},
        "scoring": {f"p{q}_ms": float(np.percentile(scoring, q)) for q in [50, 95, 99]},
        "scope": (
            "Sequential HTTP, reserved account IDs and fresh state; Redis commit included, "
            "SHAP generated during review"
        ),
        "scoring_sla_p99_pass": np.percentile(scoring, 99) <= cfg.scoring_sla_p99_ms,
    }
    result["scoring_sla_p99_pass"] = bool(result["scoring_sla_p99_pass"])
    path = cfg.data_processed_dir / "integration" / "load_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    main(args.url, args.count)
