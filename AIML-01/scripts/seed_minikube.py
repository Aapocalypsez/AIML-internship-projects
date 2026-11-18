"""Stream files into the node: docker cp cannot see some Minikube mount paths."""
import argparse
import subprocess
import tarfile

from src.config import load_config


def main(profile="fraud-demo"):
    cfg = load_config()
    archive = cfg.data_processed_dir / "deploy-data.tar"
    folders = {"raw": cfg.data_raw_dir,
               "processed/features": cfg.data_processed_dir / "features",
               "processed/graph": cfg.data_processed_dir / "graph",
               "processed/training": cfg.data_processed_dir / "training",
               "processed/monitoring": cfg.data_processed_dir / "monitoring"}
    with tarfile.open(archive, "w") as tar:
        for relative, path in folders.items():
            if path.exists():
                tar.add(path, arcname=relative)
        repo = cfg.project_root / cfg.feast_repo_path / "data"
        for path in repo.glob("*.parquet"):
            tar.add(path, arcname="feature_repo/" + path.name)
    subprocess.run(["docker", "exec", profile, "mkdir", "-p", "/data/fraud"], check=True)
    with archive.open("rb") as data:
        subprocess.run(["docker", "exec", "-i", profile, "tar", "-xf", "-", "-C", "/data/fraud"],
                       stdin=data, check=True)
    print("Evaluated models, feature snapshots and training data seeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="fraud-demo")
    main(parser.parse_args().profile)
