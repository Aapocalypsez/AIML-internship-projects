"""Smoke tests: verify the project structure and imports work."""
import importlib


def test_src_package_imports():
    """All src subpackages should be importable."""
    packages = [
        "src",
        "src.features",
        "src.training",
        "src.serving",
        "src.streaming",
        "src.graph",
        "src.dashboard",
        "src.monitoring",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_project_structure():
    """Key directories must exist."""
    from pathlib import Path

    root = Path(__file__).parent.parent
    required_dirs = [
        "src/features",
        "src/training",
        "src/serving",
        "src/streaming",
        "src/graph",
        "src/dashboard",
        "src/monitoring",
        "feature_repo",
        "k8s",
        "docker",
        "tests",
        "docs",
        "notebooks",
    ]
    for d in required_dirs:
        assert (root / d).is_dir(), f"Missing directory: {d}"
