# Autonomous execution engine: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Autonomous execution engine guide](../guide/enterprise-operations.md).

**On this page:**

- [`mlpilot.orchestration`](#mlpilotorchestration)
- [`mlpilot.orchestration.engine`](#mlpilotorchestrationengine): `StepSpec`, `Workflow`, `plan`, `RunResult`, `run`, `resume`, `approve`, `status`
- [`mlpilot.orchestration.goal`](#mlpilotorchestrationgoal): `GoalError`, `Goal`, `goal_from_text`
- [`mlpilot.orchestration.tools`](#mlpilotorchestrationtools): `GateFailed`, `ToolSpec`, `register_tool`, `frame_fingerprint`, `RunContext`

---

## `mlpilot.orchestration`

<sub>Source: [`mlpilot/orchestration/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/orchestration/__init__.py)</sub>

Autonomous execution engine: from a goal to a registered, monitored model, step by step.

```python
from mlpilot import orchestration as orc
goal = orc.goal_from_text("Predict churned from data/customers.csv with AUC above 0.85, "
                          "latency below 50 ms; deploy to production")
print(orc.plan(goal).describe())
res = orc.run(goal, "runs/churn")          # stops at the production approval
orc.approve("runs/churn", "promote", by="priya")
orc.resume("runs/churn")
```

Design rules (enforced in code, not by convention):

* **Plans use registered tools only** (`TOOLS`): a planner, rule-based or an LLM, chooses steps
  and typed parameters; it can't run arbitrary code.
* **Gates are deterministic** (data quality and contracts, poisoning, metric floors, latency and
  size limits, fairness, robustness) and are never retried away.
* **High-risk steps need people.** Promotion waits for the goal's number of approvers, none of
  whom may be the requester, and approvals are tied to the exact inputs they approved.
* **Runs are resumable and idempotent.** Each step's inputs are hashed, unchanged steps are
  reused, and every event, cost and decision is recorded in the run directory.

CLI: `mlpilot plan`, `mlpilot run`, `mlpilot approve`, `mlpilot status`.

---

## `mlpilot.orchestration.engine`

<sub>Source: [`mlpilot/orchestration/engine.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/orchestration/engine.py)</sub>

The execution engine: plan a goal into a typed workflow, run it with checkpoints, approvals,
retries, budgets and a complete record.

```python
wf = plan(goal)                    # a validated DAG of registered tools
res = run(goal, "runs/churn")     # or run(wf, ...); resumable, idempotent
print(res.summary())
approve("runs/churn", "promote", by="priya")      # four-eyes: not the person who started the run
resume("runs/churn")
```

Everything a run does is in its directory:

* `workflow.json`  the goal and the plan;  `state.json`  each step's status, timings and result
* `events.jsonl`   append-only log of every start, finish, retry, gate, approval and error
* `approvals.json` who approved what (tied to the exact inputs of the step)
* `manifest.json`  reproducibility: library versions, Python, OS, git commit, data fingerprint, seed
* `artifacts/`     data splits, the model, metrics, model card, monitor baseline;  `report.md`

A step is skipped on resume only when its inputs (tool, parameters, goal and upstream results)
are unchanged, so re-running is safe and cheap; change the goal and exactly the affected steps
run again. Low-risk steps run automatically; a high-risk step (promotion) waits until the goal's
required number of *other* people have approved it.

### class `StepSpec`

```python
class StepSpec:
```

| Field | Type | Default |
|---|---|---|
| `id` | `str` | required |
| `tool` | `str` | required |
| `params` | `dict` | `dict()` |
| `needs` | `list` | `list()` |
| `approval_stage` | `str \| None` | `None` |
| `retries` | `int` | `1` |
| `optional` | `bool` | `False` |

### class `Workflow`

```python
class Workflow:
```

| Field | Type | Default |
|---|---|---|
| `goal` | `Goal` | required |
| `steps` | `list` | required |

#### `Workflow.validate()`

```python
validate() -> 'Workflow'
```

#### `Workflow.order()`

```python
order() -> list
```

Steps in dependency order (Kahn's algorithm; keeps the plan's order among ready steps).

#### `Workflow.to_dict()`

```python
to_dict() -> dict
```

#### `Workflow.from_dict()` *(classmethod)*

```python
from_dict(d: dict) -> 'Workflow'
```

#### `Workflow.describe()`

```python
describe() -> str
```

### `plan()`

```python
def plan(goal: Goal) -> Workflow
```

Turn a goal into a workflow of registered tools (deterministic rules; no LLM).

### class `RunResult`

```python
class RunResult:
```

| Field | Type | Default |
|---|---|---|
| `run_dir` | `Path` | required |
| `status` | `str` | required |
| `steps` | `dict` | required |
| `cost` | `dict` | required |

#### `RunResult.step()`

```python
step(step_id: str) -> dict
```

#### `RunResult.failed_step` *(property)*

```python
failed_step
```

#### `RunResult.summary()`

```python
summary() -> str
```

### `run()`

```python
def run(goal_or_workflow, run_dir, *, data=None, requested_by: str | None=None) -> RunResult
```

Execute (or continue) a run. `data`: a DataFrame to use instead of `goal.data`.

Returns when every step is done, a gate fails, an approval is needed, or the budget runs out;
call `resume` (or `run` again) to continue.

### `resume()`

```python
def resume(run_dir, *, data=None, requested_by: str | None=None) -> RunResult
```

Continue a run from its directory (after an approval, a fix, a crash or a goal edit).

### `approve()`

```python
def approve(run_dir, step: str, *, by: str, note: str='') -> dict
```

Record an approval for the *current* inputs of `step` (a changed plan needs new approvals).
The person who started the run can't approve it (four-eyes).

### `status()`

```python
def status(run_dir) -> RunResult
```

---

## `mlpilot.orchestration.goal`

<sub>Source: [`mlpilot/orchestration/goal.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/orchestration/goal.py)</sub>

The goal of an autonomous run: what to build, on which data, under which limits.

A goal is plain data (YAML / JSON / dict), validated before anything runs. It can be written by
hand or drafted from a sentence by `goal_from_text`; either way the same deterministic
engine executes it, with the same quality, fairness and approval gates.

```python
goal = Goal(name="churn", data="data/customers.csv", target="churned",
            min_metrics={"roc_auc": 0.85}, max_latency_ms=50, deploy="staging")
goal = Goal.from_file("goal.yml")
goal = goal_from_text("Predict churned from data/customers.csv with AUC above 0.85, deploy to staging")
```

### class `GoalError`

```python
class GoalError(ValueError):
```

The goal is incomplete or inconsistent; the message lists every problem.

### class `Goal`

```python
class Goal:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `data` | `str \| None` | required |
| `target` | `str` | required |
| `task` | `str` | `'auto'` |
| `metric` | `str \| None` | `None` |
| `min_metrics` | `dict` | `dict()` |
| `max_metrics` | `dict` | `dict()` |
| `time_budget` | `float` | `120.0` |
| `include` | `list \| None` | `None` |
| `max_latency_ms` | `float \| None` | `None` |
| `max_model_mb` | `float \| None` | `None` |
| `budget_usd` | `float \| None` | `None` |
| `cpu_hour_usd` | `float` | `0.05` |
| `test_size` | `float` | `0.2` |
| `group` | `str \| None` | `None` |
| `time` | `str \| None` | `None` |
| `contract` | `str \| None` | `None` |
| `min_quality` | `float \| None` | `50.0` |
| `max_suspicious_share` | `float \| None` | `None` |
| `sensitive` | `list` | `list()` |
| `fairness_limits` | `dict \| None` | `None` |
| `robustness` | `bool` | `True` |
| `max_membership_auc` | `float \| None` | `None` |
| `explain` | `bool` | `True` |
| `deploy` | `str` | `'staging'` |
| `approvals` | `dict` | `lambda: {'production': 1}()` |
| `registry` | `str` | `'registry'` |
| `monitor` | `bool` | `True` |
| `seed` | `int` | `0` |
| `description` | `str` | `''` |

#### `Goal.from_dict()` *(classmethod)*

```python
from_dict(data: dict) -> 'Goal'
```

#### `Goal.from_file()` *(classmethod)*

```python
from_file(path) -> 'Goal'
```

#### `Goal.to_dict()`

```python
to_dict() -> dict
```

#### `Goal.to_yaml()`

```python
to_yaml(path=None) -> str
```

#### `Goal.validate()`

```python
validate() -> 'Goal'
```

#### `Goal.approvals_for()`

```python
approvals_for(stage: str) -> int
```

### `goal_from_text()`

```python
def goal_from_text(text: str, *, data: str | None=None, name: str | None=None, use_llm: bool | str='auto', model: str='claude-opus-5-5', **overrides) -> Goal
```

Draft a `Goal` from a sentence such as *"Predict churned from customers.csv with AUC
above 0.85 and latency below 50 ms; deploy to staging"*.

Common phrasings are understood by deterministic rules. With `use_llm` (default: when
`ANTHROPIC_API_KEY` is set), Claude drafts the remaining fields; whatever it returns is
validated like a hand-written goal, and rule-based values win where both exist. The LLM only
proposes a goal: it never runs anything. `overrides` set fields explicitly.

---

## `mlpilot.orchestration.tools`

<sub>Source: [`mlpilot/orchestration/tools.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/orchestration/tools.py)</sub>

Tools: the only operations a workflow can perform.

Every step of a plan names a registered tool. A planner (rules or an LLM) can only choose among
these, with typed parameters; it cannot run arbitrary code, skip a gate or touch production
without the approval the goal requires. Each tool wraps an existing MLPilot function, reads and
writes artifacts through the `RunContext`, and returns a small JSON summary.

Gates raise `GateFailed`: the run stops there with the reason, and is not retried
(a failed gate is a decision, not a transient error).

### class `GateFailed`

```python
class GateFailed(Exception):
```

A quality / fairness / performance gate rejected the run (not retried).

**Constructor**

```python
GateFailed(message: str, details: dict | None=None)
```

### class `ToolSpec`

```python
class ToolSpec:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `fn` | `Callable` | required |
| `description` | `str` | required |
| `risk` | `str` | `'low'` |
| `always_run` | `bool` | `False` |
| `goal_fields` | `tuple \| None` | `None` |

### `register_tool()`

```python
def register_tool(name: str, description: str, *, risk: str='low', always_run: bool=False, goal_fields: tuple | None=None)
```

Decorator adding a tool. `fn(ctx, **params) -> dict`. `goal_fields` lists the goal
fields the tool reads, so editing any other field doesn't re-run it on resume.

### `frame_fingerprint()`

```python
def frame_fingerprint(df: pl.DataFrame) -> str
```

Content hash of a DataFrame (schema + every value).

### class `RunContext`

```python
class RunContext:
```

Artifacts of a run: DataFrames (Parquet), models (saved directories) and JSON, kept on
disk under `<run>/artifacts` so a resumed run picks up where it stopped.

**Constructor**

```python
RunContext(run_dir, goal, *, data: pl.DataFrame | None=None, run_id: str='', log=None)
```

#### `RunContext.log()`

```python
log(message: str, **details)
```

#### `RunContext.put()`

```python
put(name: str, obj)
```

#### `RunContext.get()`

```python
get(name: str)
```

#### `RunContext.path()`

```python
path(name: str) -> Path
```
