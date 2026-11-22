# Autonomous engine & enterprise operations

This page covers the parts of MLPilot that run a model programme in production without someone
watching every step:
- the autonomous execution engine;
- data observability;
- incident handling;
- backups and disaster recovery;
- supply-chain security;
- serving limits;
- model governance;
- active learning;
- privacy testing;
- multi-tenancy.

What each one covers, and what is still missing, is tracked in the
[enterprise gap audit](../enterprise-gap-audit.md). To run everything on this page end to end,
use the executed notebook [examples/enterprise_walkthrough.ipynb](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/examples/enterprise_walkthrough.ipynb).

## Autonomous execution engine

Describe the goal; the engine plans the work, runs it with every gate, and stops where a person
has to decide.

```python
from mlpilot import orchestration as orc

goal = orc.goal_from_text("Predict churned from data/customers.csv with AUC above 0.85, "
                          "latency below 50 ms; check fairness across gender; deploy to production")
print(orc.plan(goal).describe())
res = orc.run(goal, "runs/churn")             # stops at the production approval
orc.approve("runs/churn", "promote", by="priya")
orc.resume("runs/churn")
```

```bash
mlpilot plan --text "predict churned from customers.csv with AUC above 0.85" --save goal.yml
mlpilot run --goal goal.yml --out runs/churn          # exit 0 done, 3 waiting for approval, 1 failed
mlpilot approve runs/churn --step promote --by priya
mlpilot run --out runs/churn --resume
mlpilot status runs/churn
```

**The plan:**
1. ingest
2. data-quality score and data contract
3. leak-checked split
4. poisoning check
5. AutoML
6. held-out evaluation against the goal's metric floors
7. latency and size limits
8. SHAP
9. fairness
10. robustness
11. membership-inference privacy check
12. model-file security scan
13. model card
14. register
15. promote
16. monitor baseline

**Rules enforced in code:**
- **Registered tools only.** A plan can use only registered tools with typed parameters. A
  planner, even an LLM, can't run arbitrary code or skip a gate.
- **The LLM only drafts.** Goals written from text use deterministic rules, plus Claude when
  `ANTHROPIC_API_KEY` is set. Either way the draft is validated like a hand-written goal, and the
  LLM never executes anything.
- **Gates are decisions.** A failed gate isn't retried. Unexpected errors are retried with
  back-off.
- **Promotion needs other people.** It waits for the goal's number of approvers, none of whom may
  be the requester. Approvals are tied to the step's exact inputs.
- **Re-runs are cheap.** Each step's inputs are hashed, so re-running reuses everything that
  didn't change. A killed run is taken over and resumed.
- **Cost-aware AutoML.** With latency or size limits, AutoML searches for the best model that
  meets them (dropping the ensemble, then the offending model families), rather than failing on
  the most accurate one.
- **Everything is recorded** in the run directory:
  - `events.jsonl`: an append-only log;
  - `state.json`: each step's timings and results;
  - the cost per step, priced as CPU time times a rate, with a budget stop;
  - `manifest.json`: packages, Python, OS, git commit, data fingerprint and seed, for
    reproduction;
  - `report.md`.

## Data observability & reconciliation

```python
from mlpilot.quality import DataMonitor, reconcile

mon = DataMonitor("orders", history="obs/orders.jsonl", timestamp="created_at",
                  sla={"max_staleness_hours": 6, "min_rows": 10_000, "max_null_rate": {"country": 0.01}})
rep = mon.observe(todays_batch)        # freshness, volume, schema, nulls, distribution (PSI)
rec = reconcile(source, warehouse, key="order_id")   # missing / extra / duplicate keys, changed rows, sums
```

The history keeps statistics only, never rows. Integer ID columns are excluded from
distribution checks.

## Incidents: diagnose, act, verify

```python
from mlpilot.monitoring import IncidentLog, diagnose

diag = diagnose(monitor=monitor.check(batch), data=mon.observe(batch),
                service={"latency_ms_p95": 180}, slo={"latency_ms_p95": 100}, released_hours_ago=5)
log = IncidentLog("incidents.jsonl", registry=registry)
inc = log.remediate(log.open(diag, model="churn"), handlers={"notify": page_on_call})
log.approve(inc, "rollback", by="priya")          # high-risk actions wait for a person
log.verify(inc, monitor.check(next_batch))        # closed only when a fresh check is healthy
```

| Cause | Typical evidence | Playbook |
|---|---|---|
| `upstream_data` | stale data, missing or retyped columns, volume collapse, null spikes | pause retraining, notify the data owner |
| `infrastructure` | latency or error-rate SLO breached; model and data normal | scale out, restart workers |
| `bad_release` | performance fell soon after a new version went live | roll back (approval) |
| `concept_drift` | accuracy fell, inputs unchanged | retrain on recent labelled data |
| `covariate_shift` | inputs and predictions shifted | retrain; review features |

Retraining never means automatic promotion. A retrained model goes through the same gates as any
other.

## Backups & disaster recovery

```bash
mlpilot backup --source registry=registry --source runs=runs --sqlite hub=hub.db --out backups --keep-last 14
mlpilot restore backups/backup-20261121T120000000000Z --target registry=/srv/registry
mlpilot dr-drill --source registry=registry --workdir dr --rto-limit 120   # exit 1 if it fails
```

**Backups.** Each source becomes a checksummed `.tar.gz` with a manifest. SQLite databases are
copied with SQLite's online backup API, so the Hub can stay up.

**Restores.** Every archive is verified before extraction, and unsafe paths and links are
refused.

**Drills.** A drill backs up, restores into a scratch area and compares every file. It reports
RTO (how long the restore took) and RPO (how old the restored data is).

For PostgreSQL, take a `pg_dump` or a cloud snapshot and add the dump file as a source.

## Supply-chain security

```bash
mlpilot scan-model models/churn        # exit 1 if a model file would run code when loaded
mlpilot sbom --out sbom.cdx.json       # CycloneDX 1.5: MLPilot and its installed dependencies
```

**The scanner.** It replays pickle and joblib files with a restricted unpickler:
- non-allowlisted globals are replaced by inert stubs, so nothing from the file executes;
- dangerous ones (`os.system`, `eval`, `subprocess`, network modules, nested loaders) are
  critical findings;
- ONNX files are checked for custom operator domains and external data.

The engine runs the scan before registering any model.

## Serving limits

```python
app = ac.make_app(model, api_key=..., rate_limit="120/minute", validate="strict")
```

- **Rate limits.** Each client is a token bucket, keyed by the hashed API key or by address
  (`trust_proxy=True` uses `X-Forwarded-For`). Requests over the limit get `429` with a
  `Retry-After` header.
- **Validation.** It checks for required input columns and non-numeric values in numeric
  columns:
  - `strict`: such records get `422`;
  - `warn` (the default): they are scored, and the problems come back in `warnings`.
- **Production configuration.** Set `AC_RATE_LIMIT`, `AC_TRUST_PROXY=1` and `AC_VALIDATE`.

## Model governance: risk & retirement

```python
reg.set_risk("credit", "high", reason="credit decisions about people", owner="risk-team", review_by="2027-06-30")
reg.promote("credit", 3, "production", approved_by=["priya", "omar"])   # high risk: two approvers, review current
reg.retire("credit", reason="replaced by the v2 scorecard", replacement="credit-v2")
```

| Risk level | Effect |
|---|---|
| `unacceptable` | no version can reach staging or production |
| `high` | production needs two approvers other than the promoter, and a risk review that isn't overdue |
| `minimal` / `limited` | no extra rules; recorded for the inventory |

A retired model has every version archived. It refuses promotion, and loading it raises
`ModelRetiredError` naming the replacement.

## Active learning & labeling

```python
from mlpilot.labeling import LabelQueue, select_for_labeling, simulate_active_learning

picks = select_for_labeling(model, unlabeled, n=50, id_col="ticket_id")   # diverse, most uncertain
q = LabelQueue("labels/")
q.add(picks, id_col="ticket_id")
q.record("T-102", "refund", by="asha")         # or q.import_labels(export_from_your_tool)
q.labels(), q.agreement()                      # majority vote, agreement between labelers
new_rows = q.training_rows(label_name="intent", min_agreement=0.66)
```

`simulate_active_learning(df, target)` replays labeling on data you already have. It shows
whether uncertainty sampling beats random labeling for your problem before you spend money on
labels.

## Privacy attacks on your own models

```python
from mlpilot.model_privacy import extraction_test, membership_inference_test

membership_inference_test(model, train, holdout).summary()   # attack AUC 0.5 = nothing leaks
extraction_test(model, holdout, queries=(100, 1000)).table   # how fast a copy matches your model
```

To make privacy a release gate, set `max_membership_auc` in the engine's goal (e.g. `0.6`).

## Multi-tenancy in the Hub

A platform admin creates organisations (tenants) with quotas, assigns users to them, and creates
workspaces inside them:

```bash
curl -X POST $HUB/api/orgs -d '{"name": "acme", "quotas": {"max_workspaces": 5, "max_model_versions": 200,
     "max_storage_bytes": 50000000000, "max_runs_per_day": 500}}'
curl -X POST $HUB/api/orgs/acme/users -d '{"user": "ana@acme.com"}'
curl -X POST $HUB/api/workspaces -d '{"name": "acme-churn", "org": "acme"}'
curl $HUB/api/orgs/acme/usage?days=30            # api_calls, runs, model_versions, storage_bytes per day
```

- **Isolation.** Adding a user from another tenant to a workspace is refused. Other tenants'
  workspaces aren't even visible.
- **Quotas.** Exceeding one returns `429` with what is in use.
- **Metering.** Successful requests are counted per tenant per day.
- **Existing Hub databases** are migrated in place when the Hub starts. Workspaces without an
  organisation keep working as before.

## Release certification

The `certify` job in CI installs the built package and runs the production-guarantee test suites
(fault injection, DR, tenancy, supply chain …). It then runs the certification sequence:
- an autonomous run;
- a model scan;
- a DR drill;
- an SBOM.

It uploads the evidence, and a release can't be published unless it passes.
