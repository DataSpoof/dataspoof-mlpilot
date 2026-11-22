# Operations: backups, labeling, supply chain: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Operations: backups, labeling, supply chain guide](../guide/enterprise-operations.md).

**On this page:**

- [`mlpilot.backup`](#mlpilotbackup): `BackupManifest`, `backup`, `verify_backup`, `restore`, `DrillReport`, `dr_drill`, `prune_backups`
- [`mlpilot.labeling`](#mlpilotlabeling): `select_for_labeling`, `LabelQueue`, `simulate_active_learning`
- [`mlpilot.supply_chain`](#mlpilotsupply_chain): `Finding`, `ScanReport`, `scan_model`, `generate_sbom`

---

## `mlpilot.backup`

<sub>Source: [`mlpilot/backup.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/backup.py)</sub>

Backup, restore and disaster-recovery drills for everything MLPilot keeps on disk.

```python
m = backup({"registry": "registry", "runs": "runs", "monitors": "monitoring"}, "backups",
           sqlite={"hub": "hub.db"})
verify_backup(m.path)                      # every archive re-hashed against the manifest
restore(m.path, {"registry": "restored/registry"})
report = dr_drill({"registry": "registry"}, "dr-workdir")   # backup -> restore -> compare
report.passed, report.rto_seconds, report.rpo_seconds
prune_backups("backups", keep_last=7)
```

**Backups.** Each named source (a directory or a file) becomes one `.tar.gz`. The manifest
records each archive's SHA-256 and every file's hash and size. SQLite databases, such as the Hub's
default store, are copied with SQLite's online backup API, so the copy is consistent while the
Hub is running. For PostgreSQL, run `pg_dump` or use your cloud's snapshots, then add the dump
file as a source.

**Restores.** Checksums are verified before anything is extracted. Extraction refuses absolute
paths, `..` and links. A target that isn't empty is left alone unless `overwrite=True`.

**Drills.** A drill backs up, restores into a scratch directory and compares every file's hash.
It reports RTO (how long the restore took) and RPO (how old the restored data is).

### class `BackupManifest`

```python
class BackupManifest:
```

| Field | Type | Default |
|---|---|---|
| `path` | `Path` | required |
| `created` | `str` | required |
| `sources` | `dict` | required |

#### `BackupManifest.to_dict()`

```python
to_dict() -> dict
```

### `backup()`

```python
def backup(sources: dict, dest, *, sqlite: dict | None=None, label: str | None=None) -> BackupManifest
```

Back up `sources` ({name: directory or file}) and `sqlite` ({name: database file})
into a new timestamped folder under `dest`.

### `verify_backup()`

```python
def verify_backup(backup_path) -> dict
```

`{source: "ok" | "missing" | "corrupt"}` after re-hashing every archive.

### `restore()`

```python
def restore(backup_path, targets: dict, *, overwrite: bool=False) -> dict
```

Restore the named sources into `targets` ({name: destination}). A `file` / `sqlite`
source is restored to exactly the destination path. Returns `{name: files restored}`.

### class `DrillReport`

```python
class DrillReport:
```

| Field | Type | Default |
|---|---|---|
| `passed` | `bool` | required |
| `rto_seconds` | `float` | required |
| `rpo_seconds` | `float` | required |
| `sources` | `dict` | required |
| `backup_path` | `Path` | required |

#### `DrillReport.summary()`

```python
summary() -> str
```

### `dr_drill()`

```python
def dr_drill(sources: dict, workdir, *, sqlite: dict | None=None, rto_limit: float | None=None) -> DrillReport
```

Back up, restore into a scratch area under `workdir` and prove the copy is identical.
`rto_limit` (seconds) fails the drill when the restore is slower.

### `prune_backups()`

```python
def prune_backups(dest, *, keep_last: int=7) -> list
```

Delete all but the newest `keep_last` backups under `dest`; returns what was removed.

---

## `mlpilot.labeling`

<sub>Source: [`mlpilot/labeling.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/labeling.py)</sub>

Active learning and labeling: spend the labeling budget where the model learns most.

```python
picks = select_for_labeling(model, unlabeled, n=50, id_col="ticket_id")      # most uncertain first
queue = LabelQueue("labels/")
queue.add(picks, id_col="ticket_id")
queue.record("T-102", "refund", by="asha")         # or queue.import_labels(csv_from_your_tool)
queue.record("T-102", "refund", by="ben")
queue.labels()                                      # majority label, votes, agreement per item
queue.agreement()                                   # how consistent are the labelers?
simulate_active_learning(df, "label")              # does it beat random labeling on your data?
```

Strategies for `select_for_labeling`:

* `least_confident`: the lowest top-class probability.
* `margin`: the smallest gap between the two most likely classes.
* `entropy`: the most spread-out probabilities.
* `diverse` (default): margin, but spread across clusters of the candidates, so a batch isn't
  50 near-copies of one confusing case.

### `select_for_labeling()`

```python
def select_for_labeling(model, pool: pl.DataFrame, n: int=50, *, strategy: str='diverse', id_col: str | None=None, seed: int=0) -> pl.DataFrame
```

The `n` rows of `pool` most worth labeling for a classifier `model` (anything with
`predict_proba` returning one column per class), with `uncertainty` and `predicted`.

### class `LabelQueue`

```python
class LabelQueue:
```

A durable labeling queue in a directory: `items.parquet` (what to label) and
`labels.jsonl` (every label given, by whom and when; append-only).

| Field | Type | Default |
|---|---|---|
| `path` | `str \| Path` | required |

#### `LabelQueue.items()`

```python
items() -> pl.DataFrame
```

#### `LabelQueue.add()`

```python
add(rows: pl.DataFrame, *, id_col: str | None=None) -> int
```

Queue `rows` (ids from `id_col`, else generated). Already-queued ids are skipped.
Returns how many were added.

#### `LabelQueue.record()`

```python
record(item_id, label, *, by: str, note: str='') -> dict
```

#### `LabelQueue.import_labels()`

```python
import_labels(df, *, id_col: str='item_id', label_col: str='label', by_col: str | None='by', default_by: str='import') -> int
```

Bulk-load labels exported from a labeling tool (a DataFrame or a CSV / Parquet path).

#### `LabelQueue.pending()`

```python
pending() -> pl.DataFrame
```

Queued items with no label yet.

#### `LabelQueue.labels()`

```python
labels(*, min_votes: int=1) -> pl.DataFrame
```

One row per labelled item: the majority `label`, `votes`, and `agreement`
(share of votes for the majority). Ties are left unresolved (`label` null).

#### `LabelQueue.agreement()`

```python
agreement() -> dict
```

Label quality across items with at least two votes: mean agreement and the share of
unanimous items.

#### `LabelQueue.training_rows()`

```python
training_rows(*, label_name: str='label', min_agreement: float=0.0) -> pl.DataFrame
```

Queued items joined with their agreed labels, ready to add to the training set.

### `simulate_active_learning()`

```python
def simulate_active_learning(df: pl.DataFrame, target: str, *, initial: int=50, batch: int=25, rounds: int=6, strategies=('margin', 'random'), test_size: float=0.3, seed: int=0) -> pl.DataFrame
```

Replay labeling on data you already have labels for: start from `initial` labelled
rows, add `batch` rows per round chosen by each strategy, and record test accuracy. Uses
logistic regression on the numeric columns, so it is quick.

---

## `mlpilot.supply_chain`

<sub>Source: [`mlpilot/supply_chain.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/supply_chain.py)</sub>

Software supply-chain security: scan model files before trusting them, and list what you ship.

```python
rep = scan_model("models/churn")        # a saved model directory or a single file
rep.passed, rep.findings                # critical: code that would run on load (os.system, eval …)
generate_sbom("sbom.cdx.json")          # CycloneDX 1.5 bill of materials of the environment
```

**How the scan works.** Pickle and joblib files can run arbitrary code when loaded. The scanner
replays the file with a restricted unpickler:
- Every global the file imports is recorded.
- Only allowlisted library classes (numpy, scipy, scikit-learn, polars, LightGBM, XGBoost,
  MLPilot, safe builtins) are resolved.
- Everything else is replaced by an inert stub, so nothing from the file can execute.
- Known-dangerous callables (`os.system`, `subprocess`, `eval`, `exec`,
  `__import__`, socket and network modules, nested `pickle` or `load` calls) are
  **critical**. Other unknown globals are **warnings**.

ONNX files are checked for custom operator domains and external data files, which load native
code or read outside the model.

### class `Finding`

```python
class Finding:
```

| Field | Type | Default |
|---|---|---|
| `file` | `str` | required |
| `severity` | `str` | required |
| `kind` | `str` | required |
| `detail` | `str` | required |

### class `ScanReport`

```python
class ScanReport:
```

| Field | Type | Default |
|---|---|---|
| `path` | `str` | required |
| `files` | `list` | required |
| `findings` | `list` | required |
| `globals_seen` | `dict` | required |

#### `ScanReport.critical` *(property)*

```python
critical: list
```

#### `ScanReport.passed` *(property)*

```python
passed: bool
```

#### `ScanReport.summary()`

```python
summary() -> str
```

### `scan_model()`

```python
def scan_model(path, *, include_unknown: bool=False) -> ScanReport
```

Scan a model file or directory without loading it the unsafe way. `include_unknown`
also scans files with unrecognised extensions as pickles.

### `generate_sbom()`

```python
def generate_sbom(path=None, *, root: str | None='dataspoof-mlpilot', all_packages: bool=False) -> dict
```

A CycloneDX 1.5 SBOM (JSON) of the current Python environment.

By default it covers `root` and everything it depends on, as installed (`all_packages`:
every installed distribution). Each component has a name, version, purl and licence, and the
dependency graph links them. Written to `path` when given; the dict is returned either way.
