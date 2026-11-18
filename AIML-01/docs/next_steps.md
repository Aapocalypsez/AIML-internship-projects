# Your next steps — November 1, 2026

The local application has been started and its dashboard opened in this chat.
No installation or model training is needed for the following exercise.

## What has already been done

- Docker Desktop and the existing Minikube cluster were started.
- All nine deployments were verified available.
- The next 1,000 dataset transactions were sent through Kafka and all were stored.
- The database now contains 2,000 transactions and 37 review cases, with no analyst
  labels yet at verification time.
- API and dashboard health checks passed; the evaluated champion is version 1 in
  this cluster's registry.
- Scheduled graph inference completed, and a fresh monitoring report completed.
- Seven deployment/replay checks passed, including dependency recovery and a
  one-row replay with missing device/email fields.

These counts are a snapshot, not live counters. Their saved source is
`data/processed/integration/resume_20261101.json`; test evidence is
`data/processed/integration/restart_tests.xml`. Earlier project measurements in
`results.md` are preserved as historical evidence.

## Do this in the dashboard

1. Open http://127.0.0.1:8502 and select **Operations**. Check that the scored count
   is 2,000, then return to **Review queue**.
2. Under **Open transaction**, search for and select **2988069**.
3. Inspect its risk estimate (about 95.8%), threshold (about 59.3%), model reasons,
   transaction fields, and linked accounts. The displayed estimate is not proof
   of fraud.
4. Change **Case status** to `reviewing`.
5. Enter a note such as: `Demo review: inspected model reasons and linked accounts;
   checked the original dataset label.` Press **Save case**.
6. For this historical demonstration, the original IEEE-CIS row for 2988069 has
   `isFraud=1`, which was checked directly from the source CSV. You can therefore
   press **Confirm fraud** to record the known demonstration outcome. Do not use
   the risk score alone as a confirmed label for other cases.
7. Select `closed` or `all` under **Show cases** to find the case again. Confirm that
   its label persisted. The labeled count in **Operations** should increase.
8. Open **Drift reports** and choose the latest report. Two of the four monitored
   columns shifted relative to the reference sample. This is a historical replay
   distribution check, not observed fraud or a live bank incident.

## What remains outside this exercise

Full CPU retraining overloaded the 4 GiB cluster and was stopped. Its Job
`drift-resume-20261101` is suspended; do not resume it beside the interactive
services. The hourly schedule remains enabled for reports without full training.
The existing champion remains available. The previously completed GPU challenger
and its rejection are documented in `results.md`; this interrupted attempt is not
a new completed training result. A new training run needs a separate configured
host/GPU environment or enough additional resources.

## Later, if you want another transaction batch

The next unused dataset offset is 2000 as of this verification. From PowerShell:

```powershell
kubectl --context fraud-demo -n fraud exec deployment/api -- python -m src.streaming.producer --offset 2000 --limit 100 --speed 1000
```

After that batch, use offset 2100 for the next 100. Keep events ordered for each
account and do not modify a payload under an already used transaction ID.

The dashboard connection depends on Docker, Minikube, and the port-forward
processes remaining running. If they are closed or the laptop restarts, follow
`use_and_handover.md` to restore the connections. This remains a local prototype;
public sharing and real payment integration have not been configured.
