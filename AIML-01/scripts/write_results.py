"""Generate the presentation's numbers from measured artifacts, never typed metrics."""
# ruff: noqa: E501

import json
from importlib.metadata import version
from xml.etree import ElementTree

from src.config import load_config


def main():
    cfg = load_config()
    data = cfg.data_processed_dir

    def read(path):
        return json.loads((data / path).read_text())

    baseline = read("features/results.json")
    graph = read("graph/results.json")
    pipeline = read("integration/pipeline.json")
    load = read("integration/load_test.json")
    drift = read("monitoring/demo.json")
    dashboard = read("integration/dashboard.json")
    parity = read("integration/online_parity.json")
    suite = ElementTree.parse(data / "integration/tests.xml").getroot().find("testsuite")
    cluster_path = data / "integration/kubernetes.json"
    cluster = json.loads(cluster_path.read_text()) if cluster_path.exists() else {"verified": False}
    report = [
        "# Measured project results",
        "",
        "All values below are read from saved run artifacts.",
        "",
        "## Model ranking",
        "",
        "| PR-AUC (average precision) | Validation | Test |",
        "|---|---:|---:|",
        f"| Original logged baseline | {baseline['comparison']['val']['baseline_pr_auc']:.6f} | {baseline['comparison']['test']['baseline_pr_auc']:.6f} |",
        f"| Engineered XGBoost | {baseline['metrics']['val']['pr_auc']:.6f} | {baseline['metrics']['test']['pr_auc']:.6f} |",
        f"| Calibrated GNN alone | {graph['ablation']['val']['gnn']:.6f} | {graph['ablation']['test']['gnn']:.6f} |",
        f"| Selected calibrated combination | {graph['ablation']['val']['blend']:.6f} | {graph['ablation']['test']['blend']:.6f} |",
        "",
        f"Selected XGBoost weight: **{graph['xgb_weight']:.1f}**. GNN did not improve ranking.",
        f"Graph account coverage: validation {graph['ablation']['val']['account_coverage']:.1%}, test {graph['ablation']['test']['account_coverage']:.1%}.",
        "Validation ablation includes calibration rows; the second validation half selected weights. Test was held out.",
        "",
        "## Live delivery and review",
        "",
        f"- {pipeline['produced_valid_unique']:,} unique inputs -> {pipeline['scored_unique']:,} unique scored outputs -> {pipeline['postgres_unique']:,} database rows.",
        f"- {pipeline['scored_including_duplicate'] - pipeline['scored_unique']} duplicate output carried the cached prediction; {pipeline['dlq']} invalid input reached the DLQ.",
        f"- Database label persistence: {pipeline['label_persistence']}; dashboard case/label button proof: {dashboard['case_status_saved']} / {dashboard['label_button_persisted']}.",
        f"- Kafka integration run: `{pipeline['run_id']}`; model versions: {', '.join(pipeline['model_versions'])}.",
        f"- Actual persisted-scorer parity: {parity['transactions']:,} transactions, {parity['features']} features; max error {parity['max_abs_error']:.4g}.",
        "",
        "## Latency",
        "",
        "| Measurement | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|",
    ]
    for name, row in [
        ("Initial Kafka scoring timer", pipeline["scoring"]),
        ("Initial HTTP round trip", pipeline["http_round_trip"]),
        ("Optimized scorer including Redis commit", load["scoring"]),
        ("Optimized HTTP round trip", load["http_round_trip"]),
    ]:
        report.append(
            f"| {name} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['p99_ms']:.2f} |"
        )
    report += [
        "",
        f"Environment: {load.get('environment', load.get('url', 'see source artifact'))}.",
        f"Latest HTTP benchmark: {load['requests']:,} requests. Scoring p99 <= {cfg.scoring_sla_p99_ms} ms: **{load['scoring_sla_p99_pass']}**.",
        "The initial timer includes SHAP and excludes the final Redis write; the optimized scorer moves SHAP to review and includes that write. These runs also had different background loads. They are not a controlled production throughput comparison.",
        "",
        "## Drift and challenger",
        "",
        f"- Unchanged input drifted share: {drift['unchanged']['drifted_column_share']:.2f}; tenfold amount-shift share: {drift['shifted']['drifted_column_share']:.2f}.",
        f"- Decision: **{drift['retraining']['decision']}**; incumbent validation PR-AUC {drift['retraining']['champion_pr_auc']:.6f}, challenger {drift['retraining']['challenger_pr_auc']:.6f}.",
        f"- Required improvement: {drift['retraining']['minimum_gain']}; eligible feedback applied: {drift['retraining']['feedback_labels_applied']}.",
        "This was a controlled demonstration, not observed production drift. No synthetic fraud labels were used.",
        "",
        "## Tests and deployment",
        "",
        f"- Tests: {suite.attrib['tests']}; failures: {suite.attrib['failures']}; errors: {suite.attrib['errors']}; skipped: {suite.attrib.get('skipped', '0')}.",
        f"- Kubernetes verification completed: **{cluster['verified']}**.",
    ]
    if cluster.get("verified"):
        report += [
            f"- Cluster: {cluster['context']}; ready deployments: {cluster['ready_deployments']}.",
            f"- Kafka replay stored in cluster: {cluster.get('scored_transactions', 'not measured')}.",
        ]
    report += [
        "",
        "## Exact sources",
        "",
        f"- Original baseline reference: `{baseline['baseline_run_id']}`. It was imported by the earlier logging script and is not a fresh baseline reproduction.",
        f"- Engineered run: `{baseline['run_id']}`.",
        f"- Graph/blend run: `{graph['registry']['run_id']}`.",
        f"- Challenger run: `{drift['retraining']['registry']['run_id']}`.",
        "- Local JSON, prediction Parquet, model files and XML are under `data/processed/` (ignored by Git). Model metrics and verification evidence are attached to MLflow.",
        "- Core versions at report time: "
        + ", ".join(
            f"{name} {version(name)}"
            for name in [
                "xgboost",
                "scikit-learn",
                "feast",
                "mlflow",
                "SQLAlchemy",
                "evidently",
                "torch",
                "torch-geometric",
            ]
        )
        + ".",
        "",
        "## Limits",
        "",
        "This is a single-node local prototype, not a production bank deployment. Pseudo-account uncertainty, frozen graph coverage, no demonstrated GNN ranking gain, fixed training/validation feedback eligibility, ordered-event requirements and any unmet latency target remain explicit limitations. Production authentication, TLS, HA and a remote backup policy are outside this build.",
        "",
    ]
    (cfg.project_root / "docs/results.md").write_text("\n".join(report), encoding="utf-8")
    print("Generated docs/results.md from measured artifacts")


if __name__ == "__main__":
    main()
