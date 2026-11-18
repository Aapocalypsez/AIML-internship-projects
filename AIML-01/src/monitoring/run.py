import argparse
import sys

from src.monitoring.drift import run_monitor

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Check recent traffic and optionally retrain")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    run_monitor(args.retrain, args.device)
