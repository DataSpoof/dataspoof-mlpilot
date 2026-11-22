# MLPilot Enterprise gap audit

This audit compares MLPilot against the 20-domain enterprise target architecture and its
readiness checklist. Each capability gets one of four statuses:

| Status | Meaning |
|---|---|
| ✅ **Implemented** | exists in the code, with tests |
| 🟡 **Needs enhancement** | exists, but enterprise use needs more |
| ❌ **Missing** | to be built |
| ⚪ **Optional** | only some customers need it; build on demand |

Statuses describe the code, not production proof. Cloud deployment, load, penetration and
disaster-recovery evidence come from running them, not from this table. It is the source of
truth for the roadmap. Update a row only with a linked change and its tests.

*Last updated: 2.3.0 development (autonomous engine, operations, governance, tenancy).*

## 1 · Interfaces & developer experience — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Python SDK, one fit / transform / save contract | ✅ | `mlpilot.core` |
| CLI for every stage | ✅ | `mlpilot …` (serve, score, train, gate, plan, run, approve, …) |
| Notebooks | 🟡 | Titanic walkthrough; image / audio / text notebooks in progress |
| REST API | ✅ | `deploy.server` (WSGI / ASGI) |
| Web workspace (EDA studio, pipeline builder, experiment studio …) | 🟡 | Hub UI covers runs, models, approvals and audit. The full studio suite is a frontend project |
| AI copilot (permission-aware) | ❌ | Builds on the execution engine's tool registry |
| IDE integrations | ⚪ | |

## 2 · Data connectors & ingestion — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Files, cloud storage, SQL, Snowflake, BigQuery, REST, Kafka | ✅ | `ingestion`, `integrations.kafka` |
| Incremental loads (watermarks) | ✅ | `IncrementalLoader` |
| CDC, late-arriving data, backfills | ❌ | |
| Kinesis, Flink | ⚪ | |
| Lakehouse tables (Iceberg, Delta) | ❌ | Polars can read Delta; no managed writer or versioned table support |
| Connector health checks | ❌ | |

## 3 · Data engineering & quality — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Cleaning pipelines learned once, replayed exactly | ✅ | `cleaners`, `steps` |
| Profiling, quality score, contracts, schema drift | ✅ | `quality` |
| Data observability (freshness, volume, schema, nulls, distribution over time; SLAs) | ✅ | `quality.DataMonitor` |
| Source-to-destination reconciliation | ✅ | `quality.reconcile` (keys, changed rows, sums, schema) |
| Distributed ETL | 🟡 | `scale` (chunked files, Spark / Ray / Dask scoring, SQL pushdown) |
| SQL and partition optimisation advice | ❌ | |

## 4 · Data governance & lifecycle — P1

| Capability | Status | Where / what's left |
|---|---|---|
| PII detection and masking | ✅ | Presidio-based cleaning step |
| Data versioning | ✅ | `versioning.DataRepository` |
| Lineage (job and dataset level) | ✅ | `lineage` (OpenLineage), Glue, ADF |
| Column-level lineage | ❌ | |
| Catalog, business glossary, ownership | 🟡 | Glue catalog export; DataHub via OpenLineage. No native glossary |
| Retention, deletion propagation, consent | ❌ | |
| Domain-driven data products (owners, contracts, SLAs) | ❌ | |

## 5 · Feature engineering & feature store — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Feature pipelines and leakage detection | ✅ | `features` |
| Point-in-time-correct offline and online store | ✅ | `feature_store`, Feast export |
| Autonomous feature discovery | ❌ | |

## 6 · AutoML & deep learning — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Tabular AutoML: CV, ensembles, Optuna, imbalance, conformal | ✅ | `automl` |
| Forecasting, causal, anomaly, clustering | ✅ | |
| Cost-aware AutoML (best model under latency, size and budget) | ✅ | Engine `automl` step searches for the best model within the limits; budget stop |
| Data labeling and active learning | ✅ | `labeling` (uncertainty / diverse selection, label queue, agreement, simulation) |
| Deep learning engine (PyTorch / transformers), GPU | ❌ | |
| LLM fine-tuning, RAG, LLM evaluation | ❌ | |
| Recommendation systems | ⚪ | |
| Distillation and pruning | ❌ | Quantization (int8 / fp16) exists |
| Online, continual and federated learning, NAS | ⚪ | |

## 7 · Experimentation & research — P1

| Capability | Status | Where / what's left |
|---|---|---|
| Runs, params, metrics, artifacts, dataset fingerprints | ✅ | `registry.Tracker` |
| Environment snapshot per run | ✅ | Execution engine `manifest.json` (packages, Python, OS, git commit, seed) |
| Statistical comparison of experiments | 🟡 | `stats` has the tests; a "run A vs run B is significant?" helper is missing |
| One-command reproduction of a run | 🟡 | Engine runs are resumable and record everything. Re-run-and-compare is missing |
| Next-experiment recommendation | ❌ | |

## 8 · Model evaluation & explainability — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Metrics with CIs, calibration, SHAP, error slices, fairness, model cards | ✅ | `evaluation` |
| Robustness and poisoning checks | ✅ | `robustness`, `poisoning`; enforced as gates by the engine |
| Membership-inference and model-extraction tests | ✅ | `model_privacy`; engine gate `max_membership_auc` |

## 9 · Model registry & versioning — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Versions, stages, gates, rollback, signed artifacts | ✅ | `registry`, `security` |
| Model risk classification, retirement, decommissioning | ✅ | `ModelRegistry.set_risk / retire`; high risk needs two approvers |
| Dependency compatibility check at load | 🟡 | Versions are recorded in the bundle; no enforcement |
| Prediction-level lineage | 🟡 | Each logged prediction has a request ID, prediction ID and model version; the link to the training run and data fingerprint is missing |

## 10 · Deployment & inference — P0

| Capability | Status | Where / what's left |
|---|---|---|
| REST, batch, streaming (Kafka), canary, shadow, hot reload | ✅ | `deploy`, `integrations.kafka` |
| Docker; SageMaker, Vertex AI, Azure ML targets; blue/green on AWS | ✅ | Not yet validated against a real cloud account |
| API keys, request size and row caps, JSON validation | ✅ | |
| Per-column request schema validation against the model contract | ✅ | `make_app(validate="strict" / "warn")` |
| Per-client rate limiting | ✅ | `make_app(rate_limit="120/minute")`, `AC_RATE_LIMIT` |
| Dynamic batching | ❌ | |
| A/B testing with statistical readout | 🟡 | Canary routing exists; the significance readout is missing |
| KServe, Triton, vLLM, BentoML backends | ❌ | |
| GPU serving, autoscaling (KEDA / HPA) | ❌ | The Helm chart has no autoscaler yet |
| Edge (ONNX, quantized) | ✅ | `export_model(format="onnx", quantize=…)` |

## 11 · Monitoring & observability — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Data and prediction drift, performance estimated without labels, delayed labels | ✅ | `monitoring` |
| Alerts (webhook, PagerDuty, Opsgenie) and retrain policy | ✅ | |
| Prometheus, Grafana, OpenTelemetry | ✅ | |
| Fairness degradation over time | ❌ | |
| Business KPI monitoring | ❌ | |
| Infra and cost metrics per model | 🟡 | Latency and throughput metrics exist; cost per prediction is missing |

## 12 · Autonomous AI orchestration — P1

| Capability | Status | Where / what's left |
|---|---|---|
| Goal → plan → typed workflow over registered tools | ✅ | `orchestration` (2.3) |
| Checkpointed, resumable, idempotent execution with retries | ✅ | `orchestration.run / resume` |
| Human approval checkpoints (four-eyes, tied to inputs) | ✅ | `orchestration.approve` |
| Run log, cost per step, reproducibility manifest | ✅ | `events.jsonl`, `state.json`, `manifest.json` |
| Natural-language goals (rules + optional Claude, validated) | ✅ | `goal_from_text` |
| Durable distributed execution (Temporal, Argo, KEDA) | ❌ | The engine runs in-process; a backend interface is next |
| Specialised agents (data, research, training, ops) | ❌ | They plug in as planners over the same tool registry |

## 13 · Autonomous incident management — P1

| Capability | Status | Where / what's left |
|---|---|---|
| Detection (drift, decay, latency) | ✅ | `monitoring` |
| Root-cause analysis, remediation playbooks, recovery checks, incident records | ✅ | `monitoring.diagnose`, `IncidentLog` (rule-based causes; high-risk actions need approval) |

## 14 · Agent security & governance — P1

| Capability | Status | Where / what's left |
|---|---|---|
| Tool allow-list, typed params, high-risk tools need approval | ✅ | `orchestration.TOOLS`, `Workflow.validate` |
| LLM token accounting | 🟡 | Recorded for goal drafting; per-agent budgets come with agents |
| Agent identity, sandboxed execution, prompt registry, LLM regression tests | ❌ | |

## 15 · Distributed compute & GPU scheduling — P1

| Capability | Status | Where / what's left |
|---|---|---|
| Rust engine, Polars | ✅ | |
| Spark / Ray / Dask scoring | ✅ | `scale` |
| Compute router (pick the engine by data size and memory) | ❌ | |
| GPU workers, spot instances, scale to zero | ❌ | |

## 16 · Enterprise security & compliance — P0

| Capability | Status | Where / what's left |
|---|---|---|
| OIDC SSO, RBAC, API tokens, OPA policies, hash-chained audit log | ✅ | `hub` |
| Secrets from Vault / AWS Secrets Manager, redaction | ✅ | `secretstore` |
| Signed models; skops export without pickle | ✅ | `security`, `deploy.export` |
| SAML, SCIM, ABAC | ❌ | |
| SBOM, signed release artifacts, dependency scanning in CI | ✅ | `mlpilot sbom`; CI: pip-audit, cargo audit, build-provenance attestations, SBOM on releases |
| Malicious model-file scanning | ✅ | `supply_chain.scan_model`, `mlpilot scan-model`; engine gate before registration |
| GDPR / SOC 2 / ISO 27001 evidence export | ❌ | |

## 17 · Multi-tenancy & identity — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Workspaces with roles | ✅ | `hub` |
| Tenant (organisation) isolation, quotas, usage metering | ✅ | Hub `/api/orgs`; cross-tenant membership refused; quotas return 429 |
| API versioning and deprecation policy | 🟡 | `docs/policies.md` covers the library; HTTP API versioning is missing |

## 18 · FinOps & resource optimisation — P1

| Capability | Status | Where / what's left |
|---|---|---|
| Cost per run and per step (CPU time × rate), budget stop | ✅ | `orchestration` |
| Cost per prediction, per tenant; chargeback | ❌ | |
| Cloud billing integration, forecasting | ⚪ | |

## 19 · High availability & disaster recovery — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Helm PDB, probes, stateless API | ✅ | `deploy/helm` |
| Backup and restore of registry, Hub DB and artifacts | ✅ | `backup` / `mlpilot backup` (checksummed; consistent SQLite snapshots) |
| RPO / RTO targets, DR drills | ✅ | `mlpilot dr-drill --rto-limit` (measured, run in CI) |

## 20 · Testing, certification & release engineering — P0

| Capability | Status | Where / what's left |
|---|---|---|
| Unit and integration tests (about 390), mypy, multi-OS wheel CI | ✅ | `tests/`, `.github/workflows/CI.yml` |
| End-to-end lifecycle test | ✅ | `tests/test_orchestration.py` |
| Fault injection (killed runs, tampered artifacts, concurrent promotions, failing models) | ✅ | `tests/test_fault_injection.py` |
| Load tests at scale | ❌ | |
| Release certification pipeline | ✅ | CI `certify` job (gates every release; uploads the evidence) |

## Not code: evidence that has to be earned

These can't be closed by writing code in this repository:

- real cloud deployment runs;
- penetration tests;
- load tests at customer scale;
- SOC 2 / ISO audits;
- SLA history;
- licensing, billing and support processes.

Track them separately, with links to the evidence.
