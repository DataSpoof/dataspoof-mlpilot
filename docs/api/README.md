# API reference

Every public class and function in `mlpilot`, generated from the source code's docstrings (`python scripts/gen_api_ref.py`). Start with the [documentation](../../README.md#-documentation) for task-oriented examples.

| Area | Public items | Summary |
|---|---:|---|
| [Pipeline core](core.md) | 18 | Shared pipeline core for the DataSpoof data-science modules. |
| [Ingestion & connectors](ingestion.md) | 19 | Ingestion & connectors (module 1 of the platform). |
| [Data cleaning](cleaning.md) | 86 | Stateful cleaning steps built on the pipeline core. |
| [Profiling, data quality & observability](quality.md) | 26 | Profiling & data quality: module 2 of the platform, built on the shared pipeline core. |
| [Data splitting](splitting.md) | 5 | Data splitting: train / validation / test splits and cross-validation folds that don't leak. |
| [Feature engineering](features.md) | 15 | Feature engineering: module 4 of the platform, built on the shared pipeline core. |
| [Feature store](feature_store.md) | 3 | Feature store: define features once, reuse them for training and serving without leakage. |
| [Data versioning](versioning.md) | 3 | Data version control: commit, tag, diff and restore datasets — like git, for DataFrames. |
| [Statistical testing](stats.md) | 16 | Statistical testing toolkit: the right test for the data, with effect sizes, confidence intervals, assumption checks and a plain-English reading — not just… |
| [AutoML](automl.md) | 13 | AutoML: module 5 of the platform. |
| [Evaluation & explainability](evaluation.md) | 18 | Evaluation & explainability: metrics, SHAP explanations, error analysis, fairness, model cards. |
| [Model security & privacy](model_security.md) | 8 | Privacy attacks against your own model: does it leak who was in the training data, and how easily can someone copy it through its API? |
| [Tracking, registry & model governance](registry.md) | 6 | Experiment tracking & model registry (module 7). |
| [Autonomous execution engine](orchestration.md) | 16 | Autonomous execution engine: from a goal to a registered, monitored model, step by step. |
| [Environments & CI/CD](environments.md) | 8 | Model CI/CD: the steps a pipeline runs, and ready-made pipelines that run them. |
| [Deployment](deploy.md) | 29 | Deployment (module 8): scoring, a REST API, batch jobs, exports and Docker images. |
| [Quantization & compression](compression.md) | 6 | Data compression: smaller column types without changing values, and quantized vectors. |
| [Monitoring & incidents](monitoring.md) | 31 | Monitoring (module 9): data & prediction drift, accuracy decay, alerts, retraining triggers. |
| [Operations: backups, labeling, supply chain](operations.md) | 14 | Backup, restore and disaster-recovery drills for everything MLPilot keeps on disk. |
| [Insights & reporting](insights.md) | 6 | Insights & reporting (module 10): plain-English findings and executive reports. |
| [Hub (platform server)](hub.md) | 48 | MLPilot Hub: the multi-user platform (server + client). |
| [Secrets & model signing](secrets_security.md) | 11 | Secrets from a secret manager instead of plain environment variables or config files. |
| [Lineage](lineage.md) | 5 | Data lineage with OpenLineage: which data went into which job, and what it produced. |
| [Forecasting](forecast.md) | 5 | Time-series forecasting: multi-step horizons, many series at once, backtests, intervals. |
| [Causal inference & uplift](causal.md) | 4 | Causal effects and uplift: what a treatment (discount, campaign, feature) actually *causes*. |
| [Anomaly detection](anomaly.md) | 1 | Unsupervised anomaly detection with explanations. |
| [Image & audio data](media.md) | 19 | Image and audio data: quality audits, consistent preprocessing, and features for AutoML. |
| [Weather, faces & plates](vision.md) | 34 | Robustness benchmark for image models: weather, low light, blur, noise and covered faces. |
| [Scale-out & SQL pushdown](scale.md) | 6 | Scaling out: transform files of any size, score on Spark / Ray / Dask, and push cleaning pipelines down into the warehouse as SQL. |
| [Integrations](integrations.md) | 38 | Integrations with the rest of the data stack. |
