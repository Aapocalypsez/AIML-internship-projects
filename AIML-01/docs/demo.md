# Demo guide

Use the measured reports already saved in `data/processed/`; do not start a full
baseline training run during a ten-minute presentation. Have the dashboard and
API health page open first. Both Compose and Minikube are local development modes.

## Preparation

1. Start one deployment using README.md, not both on a memory-constrained laptop.
2. Check API `/health`, dashboard health, Kafka, Postgres and Redis.
3. Open `docs/results.md`, the architecture diagram, the latest MLflow champion
   and the saved controlled-drift report.
4. Keep the saved integration/test evidence available in case internet or the
   local Docker environment interrupts the live demonstration.

## Live path

For the host/Compose environment, start worker and sink if the app profile is not
already running, then:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_pipeline --count 1000
```

The script uses new Kafka topics and reserved IDs rather than deleting data. It
injects a duplicate and an invalid message, checks scored output and the DLQ,
checks 1,000 unique database rows, saves a demonstration label and measures HTTP
latency. Demo labels are marked and excluded from retraining.

Open a queue item. Show its risk threshold, model version, computed features and
SHAP reasons. Show linked accounts as evidence of shared identifiers, not as
confirmed fraud. Save status `reviewing`, then a label. Refresh to show persistence.
Audit samples may be legitimate; low-risk transactions are deliberately reviewed
to reduce the bias of collecting labels only for high-risk alerts.

The Streamlit queue refreshes every five seconds. Explanation calculation may
take longer on first access because the exact case model is loaded lazily.

## Drift and retraining

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.demo_drift --device cuda
```

This compares unchanged data against an explicit tenfold amount shift, then trains
a real challenger. The saved demonstration rejected the challenger because it
performed worse on the held-out validation half. Show that rejection is successful
protection, not a failed pipeline. Test labels never decide promotion.

For monitoring actual stored traffic rather than the controlled demo:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.monitoring.run --retrain --device cuda
```

This reads recent Postgres traffic. It needs 100 observations, checks four numeric
columns, and uses a fingerprint/cooldown to avoid repeating the same retraining
job. No new labeled production data was available in the saved demonstration;
the challenger used historical training truth, not synthetic fraud labels.

## Kubernetes path

```powershell
kubectl --context fraud-demo -n fraud get pods,jobs,cronjobs
kubectl --context fraud-demo -n fraud logs job/bootstrap --tail=30
kubectl --context fraud-demo -n fraud exec deployment/api -- python -m src.streaming.producer --limit 20
```

Use port-forwards from README.md to view the deployed API/dashboard. The scheduled
graph and drift jobs use CPU limits and persistent data. The hourly monitoring
Job now creates reports without full retraining: a November 1 training attempt
exhausted the 4 GiB node and was suspended. Run a challenger separately in a
configured host/GPU environment, outside the presentation. The previously saved
GPU challenger decision remains the completed training/gate demonstration.

## Questions and honest limitations

- Quote the latency result with its environment and scope. Do not imply that a
  local single-client test measures production throughput or satisfies an SLA if
  its report says otherwise.
- The GNN did not improve ranking; validation selected zero GNN weight.
- Pseudo-accounts and shared identifiers are uncertain links, not real identities.
- Kafka is single-broker; the prototype expects ordered per-account delivery.
- This is a local demo without production authentication, TLS or high availability.
