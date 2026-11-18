# Ten-minute presentation outline

## 0:00–1:00 — Problem and objective
Fraud is rare, so accuracy alone is misleading. The system ranks incoming
transactions, preserves investigation evidence and accepts analyst corrections.
State the dataset is IEEE-CIS and pseudo-accounts are inferred, not real identities.

## 1:00–2:00 — Architecture
Show the diagram in architecture.md. Follow one transaction through Kafka,
scoring, Redis, the database and the review queue. Distinguish the fast scoring
path from scheduled graph training and expensive case explanations.

## 2:00–3:00 — Leakage prevention
Explain why data is split by time. Use an example of two purchases at the same
second: neither can see the other in its historical features. Show the 6,000-row
parity evidence and the graph's training cutoff.

## 3:00–4:00 — XGBoost results
Open results.md and use its generated table. Define PR-AUC and compare with the
fraud-rate random baseline. Describe the engineered-feature improvement as small.
Do not claim statistical significance or business savings that were not measured.

## 4:00–5:00 — Graph model and the negative result
Show accounts linked through identifiers and explain two-hop message passing.
Explain why common hubs are excluded. Show XGBoost, GNN and blend test PR-AUC.
Validation selected zero GNN scoring weight; retaining XGBoost is an evidence-based
decision, while graph links remain available during investigation.

## 5:00–6:00 — Live demonstration
Send a short replay, show the new cases, and show the saved 1,000-transaction proof.
Explain duplicate caching, the dead-letter topic and the latency measurement's scope.
State any SLA that remains unmet on the tested hardware.

## 6:00–7:00 — Analyst workflow
Open an alert, show the exact-version SHAP reasons and linked accounts. Save a
case status and label. Refresh to demonstrate that the feedback persisted.

## 7:00–8:00 — Drift and safe retraining
Show unchanged versus controlled tenfold amount-shift reports. Explain that this
is an injected demonstration. Show the actual challenger decision: rejection
preserved the stronger champion. Unlabeled transactions never become fraud labels.

## 8:00–9:00 — Deployment and proof
Show the Minikube pods, bootstrap Job and scheduled Jobs. Show the unit/parity/API
test report and saved run IDs. Explain that the deployment is local and single-node.

## 9:00–10:00 — Limits and next steps
Discuss pseudo-account uncertainty, graph coverage, no GNN ranking gain, fixed
retraining holdout, account ordering and the lack of production authentication/HA.
Conclude with what the measurements support, not what a production deployment
would require you to assume.

## Questions to be ready for
- Why PR-AUC instead of accuracy? Fraud is rare; ranking positives is the task.
- Why did the GNN not help? Train-only graph coverage and weak identifiers limit it.
- Why not force a 70/30 blend? The validation results would be worse.
- Can the same transaction be scored twice? It returns the cached prediction;
  downstream database writes remain idempotent.
- Does drift mean the model is wrong? No; it triggers evaluation, not automatic promotion.
- Can this serve a bank today? No; this is a measured local prototype with documented limits.
