"""Health check script for all infrastructure services.

Run: python scripts/health_check.py
Returns exit code 0 if all services are healthy, 1 otherwise.
"""
import socket
import sys
import urllib.request


def check_tcp(host: str, port: int, name: str) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"  {name}: OK (port {port} open)")
            return True
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"  {name}: FAIL ({e})")
        return False


def check_http(url: str, name: str) -> bool:
    """Check if an HTTP endpoint returns 200."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print(f"  {name}: OK ({url})")
                return True
            print(f"  {name}: FAIL (status {resp.status})")
            return False
    except Exception as e:
        print(f"  {name}: FAIL ({e})")
        return False


def main() -> int:
    print("=== Infrastructure Health Checks ===\n")
    results = [
        check_tcp("localhost", 9092, "Kafka"),
        check_tcp("127.0.0.1", 6380, "Redis"),
        check_tcp("localhost", 5433, "Postgres"),
        check_http("http://localhost:5000/health", "MLflow"),
    ]
    print()
    if all(results):
        print("All services healthy!")
        return 0
    else:
        failed = sum(1 for r in results if not r)
        print(f"{failed} service(s) unhealthy.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
