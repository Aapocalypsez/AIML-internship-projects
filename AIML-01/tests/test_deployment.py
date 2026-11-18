from pathlib import Path

import pytest
import yaml


def test_kubernetes_workloads_have_resource_limits_and_probes():
    objects = list(yaml.safe_load_all(Path("k8s/platform.yaml").read_text()))
    deployments = [obj for obj in objects if obj["kind"] == "Deployment"]
    assert {obj["metadata"]["name"] for obj in deployments} >= {
        "api",
        "worker",
        "sink",
        "dashboard",
        "kafka",
        "redis",
        "postgres",
        "mlflow",
    }
    for obj in deployments:
        for container in obj["spec"]["template"]["spec"]["containers"]:
            assert {"cpu", "memory"} <= container["resources"]["limits"].keys()
            assert "livenessProbe" in container
            if container.get("ports"):
                assert "readinessProbe" in container
    jobs = [obj for obj in objects if obj["kind"] == "CronJob"]
    assert {obj["metadata"]["name"] for obj in jobs} == {"graph-risk", "drift-retraining"}
    assert all(obj["spec"]["concurrencyPolicy"] == "Forbid" for obj in jobs)


def test_development_database_and_redis_ports_are_consistent():
    from src.config import load_config

    cfg = load_config()
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    assert f"{cfg.redis_port}:6379" in compose["services"]["redis"]["ports"]
    assert f"{cfg.postgres_port}:5432" in compose["services"]["postgres"]["ports"]


@pytest.mark.parametrize("unavailable", ["postgres", "redis", "mlflow"])
def test_drift_job_waits_for_dependencies_after_restart(monkeypatch, unavailable):
    import socket
    import time
    import urllib.request
    from contextlib import nullcontext

    objects = list(yaml.safe_load_all(Path("k8s/platform.yaml").read_text()))
    job = next(
        obj for obj in objects
        if obj["kind"] == "CronJob" and obj["metadata"]["name"] == "drift-retraining"
    )
    initializer = job["spec"]["jobTemplate"]["spec"]["template"]["spec"]["initContainers"][0]
    attempted = []
    sleeps = []
    failed = False

    def connect(service):
        nonlocal failed
        attempted.append(service)
        if service == unavailable and not failed:
            failed = True
            raise OSError("dependency is still starting")
        return nullcontext()

    def sleep(seconds):
        sleeps.append(seconds)
        assert len(sleeps) == 1, "Readiness loop did not recover"

    monkeypatch.setattr(socket, "create_connection", lambda address, timeout: connect(address[0]))
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: connect("mlflow"))
    monkeypatch.setattr(time, "sleep", sleep)
    exec(compile(initializer["command"][-1], "drift-job-readiness", "exec"), {})

    assert failed
    assert sleeps == [3]
    assert attempted[-3:] == ["postgres", "redis", "mlflow"]


def test_small_node_schedules_monitoring_without_full_training():
    objects = list(yaml.safe_load_all(Path("k8s/platform.yaml").read_text()))
    job = next(
        obj for obj in objects
        if obj["kind"] == "CronJob" and obj["metadata"]["name"] == "drift-retraining"
    )
    pod = job["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    assert not job["spec"]["suspend"]
    assert "src.monitoring.run" in pod["containers"][0]["command"]
    assert "--retrain" not in pod["containers"][0]["command"]
