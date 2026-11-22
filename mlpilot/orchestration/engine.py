"""The execution engine: plan a goal into a typed workflow, run it with checkpoints, approvals,
retries, budgets and a complete record.

>>> wf = plan(goal)                    # a validated DAG of registered tools
>>> res = run(goal, "runs/churn")     # or run(wf, ...); resumable, idempotent
>>> print(res.summary())
>>> approve("runs/churn", "promote", by="priya")      # four-eyes: not the person who started the run
>>> resume("runs/churn")

Everything a run does is in its directory:

* ``workflow.json``  the goal and the plan;  ``state.json``  each step's status, timings and result
* ``events.jsonl``   append-only log of every start, finish, retry, gate, approval and error
* ``approvals.json`` who approved what (tied to the exact inputs of the step)
* ``manifest.json``  reproducibility: library versions, Python, OS, git commit, data fingerprint, seed
* ``artifacts/``     data splits, the model, metrics, model card, monitor baseline;  ``report.md``

A step is skipped on resume only when its inputs (tool, parameters, goal and upstream results)
are unchanged, so re-running is safe and cheap; change the goal and exactly the affected steps
run again. Low-risk steps run automatically; a high-risk step (promotion) waits until the goal's
required number of *other* people have approved it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from .goal import Goal
from .tools import TOOLS, GateFailed, RunContext

DONE, FAILED, WAITING, SKIPPED, PENDING, BLOCKED = "done", "failed", "waiting_approval", "skipped", "pending", "blocked"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclasses.dataclass
class StepSpec:
    id: str
    tool: str
    params: dict = dataclasses.field(default_factory=dict)
    needs: list = dataclasses.field(default_factory=list)
    approval_stage: str | None = None        # approvals required = goal.approvals_for(stage)
    retries: int = 1                         # extra attempts after an unexpected error (gates never retry)
    optional: bool = False                   # a failure is recorded but does not fail the run


@dataclasses.dataclass
class Workflow:
    goal: Goal
    steps: list

    def validate(self) -> "Workflow":
        ids = [s.id for s in self.steps]
        problems = []
        if len(ids) != len(set(ids)):
            problems.append("step ids must be unique")
        for s in self.steps:
            if s.tool not in TOOLS:
                problems.append(f"step {s.id!r}: unknown tool {s.tool!r} (registered: {sorted(TOOLS)})")
            for n in s.needs:
                if n not in ids:
                    problems.append(f"step {s.id!r} needs unknown step {n!r}")
            if TOOLS.get(s.tool) and TOOLS[s.tool].risk == "high" and s.approval_stage is None:
                problems.append(f"step {s.id!r} uses high-risk tool {s.tool!r} without an approval stage")
        if not problems:
            self.order()                                     # raises on cycles
        if problems:
            raise ValueError("Invalid workflow:\n  - " + "\n  - ".join(problems))
        return self

    def order(self) -> list:
        """Steps in dependency order (Kahn's algorithm; keeps the plan's order among ready steps)."""
        remaining = {s.id: set(s.needs) for s in self.steps}
        by_id = {s.id: s for s in self.steps}
        out = []
        while remaining:
            ready = [i for i in (s.id for s in self.steps) if i in remaining and not remaining[i]]
            if not ready:
                raise ValueError(f"workflow has a cycle among {sorted(remaining)}")
            for i in ready:
                out.append(by_id[i])
                del remaining[i]
                for deps in remaining.values():
                    deps.discard(i)
        return out

    def to_dict(self) -> dict:
        return {"goal": self.goal.to_dict(), "steps": [dataclasses.asdict(s) for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        return cls(Goal.from_dict(d["goal"]), [StepSpec(**s) for s in d["steps"]]).validate()

    def describe(self) -> str:
        lines = [f"Plan for {self.goal.name!r}: predict {self.goal.target!r}"]
        for i, s in enumerate(self.order(), 1):
            spec = TOOLS[s.tool]
            gate = f"  [needs {self.goal.approvals_for(s.approval_stage)} approval(s)]" if s.approval_stage and \
                self.goal.approvals_for(s.approval_stage) else ""
            lines.append(f"  {i:>2}. {s.id:<20} {spec.description}{gate}")
        return "\n".join(lines)


def plan(goal: Goal) -> Workflow:
    """Turn a goal into a workflow of registered tools (deterministic rules; no LLM)."""
    goal.validate()
    steps = [
        StepSpec("ingest", "ingest"),
        StepSpec("quality_gate", "quality_gate", needs=["ingest"]),
        StepSpec("split", "split", needs=["quality_gate"]),
        StepSpec("poisoning_check", "poisoning_check", needs=["split"]),
        StepSpec("automl", "automl", needs=["poisoning_check"], retries=0),
        StepSpec("evaluate", "evaluate", needs=["automl"]),
        StepSpec("performance_profile", "performance_profile", needs=["automl"]),
    ]
    checks = ["evaluate", "performance_profile"]
    if goal.explain:
        steps.append(StepSpec("explain", "explain", needs=["automl"], optional=True))
    if goal.sensitive:
        steps.append(StepSpec("fairness", "fairness", needs=["automl"]))
        checks.append("fairness")
    if goal.robustness:
        steps.append(StepSpec("robustness", "robustness", needs=["automl"]))
        checks.append("robustness")
    if goal.max_membership_auc is not None:
        steps.append(StepSpec("privacy", "privacy", needs=["automl"]))
        checks.append("privacy")
    steps.append(StepSpec("model_card", "model_card", needs=["evaluate"], optional=True))
    steps.append(StepSpec("artifact_scan", "artifact_scan", needs=["automl"]))
    checks.append("artifact_scan")
    if goal.deploy != "none":
        steps.append(StepSpec("register", "register", needs=checks))
        if goal.deploy in ("staging", "production"):
            steps.append(StepSpec("promote", "promote", params={"stage": goal.deploy}, needs=["register"],
                                  approval_stage=goal.deploy, retries=0))
    if goal.monitor:
        steps.append(StepSpec("monitor_setup", "monitor_setup", needs=checks))
    return Workflow(goal, steps).validate()


# ── run records ─────────────────────────────────────────────────────────────
@dataclasses.dataclass
class RunResult:
    run_dir: Path
    status: str                          # succeeded | failed | waiting_approval | budget_exceeded | running
    steps: dict
    cost: dict

    def step(self, step_id: str) -> dict:
        return self.steps.get(step_id, {})

    @property
    def failed_step(self):
        return next((k for k, v in self.steps.items() if v["status"] == FAILED and not v.get("optional")), None)

    def summary(self) -> str:
        icon = {DONE: "ok ", FAILED: "ERR", WAITING: "...", SKIPPED: "-- ", PENDING: "   ", BLOCKED: "   "}
        lines = [f"Run {self.run_dir.name}: {self.status.upper()}  "
                 f"({self.cost.get('wall_seconds', 0):.0f}s, {self.cost.get('cpu_seconds', 0):.0f} CPU-s, "
                 f"${self.cost.get('usd', 0):.4f})"]
        for k, v in self.steps.items():
            note = v.get("error") or v.get("note") or ""
            reused = " (reused)" if v.get("reused") else ""
            lines.append(f"  [{icon.get(v['status'], '?  ')}] {k:<20} {v['status']}{reused}"
                         + (f": {note[:160]}" if note else ""))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RunResult(status={self.status!r}, run_dir={str(self.run_dir)!r})"


def _pid_alive(pid: int) -> bool:
    """Whether process ``pid`` is running on this machine (never sends it a signal on Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32                      # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)      # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _RunLock:
    """One active process per run directory. The lock records host and pid: a lock left by a
    process that no longer runs on this host (a killed run) is taken over at once; a lock from
    another host is trusted until it is ``stale`` seconds old."""

    def __init__(self, path: Path, stale: float = 6 * 3600):
        self.path, self.stale = path, stale

    def _abandoned(self) -> bool:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            info = {}
        if info.get("host") == platform.node() and isinstance(info.get("pid"), int):
            return not _pid_alive(info["pid"])
        return time.time() - self.path.stat().st_mtime >= self.stale

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not self._abandoned():
                raise RuntimeError(f"{self.path.parent} is being run by another process "
                                   f"({self.path.read_text(errors='ignore')}); delete {self.path.name} if it crashed")
            self.path.unlink(missing_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node(), "since": _now()}).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _manifest(goal: Goal, data_fingerprint: str | None) -> dict:
    from importlib import metadata
    versions = {}
    for pkg in ("dataspoof-mlpilot", "polars", "numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost",
                "pyarrow", "shap", "onnxruntime"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    commit = None
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass
    return {"created": _now(), "python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": versions, "git_commit": commit,
            "data_fingerprint": data_fingerprint, "seed": goal.seed, "goal_hash": _hash(goal.to_dict())}


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def run(goal_or_workflow, run_dir, *, data=None, requested_by: str | None = None) -> RunResult:
    """Execute (or continue) a run. ``data``: a DataFrame to use instead of ``goal.data``.

    Returns when every step is done, a gate fails, an approval is needed, or the budget runs out;
    call :func:`resume` (or ``run`` again) to continue."""
    wf = plan(goal_or_workflow) if isinstance(goal_or_workflow, Goal) else goal_or_workflow.validate()
    goal = wf.goal
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with _RunLock(run_dir / "run.lock"):
        from ..registry._store import user as _user          # AC_USER, else the OS user
        return _execute(wf, run_dir, data, requested_by or _user())


def _execute(wf: Workflow, run_dir: Path, data, user: str) -> RunResult:
    goal = wf.goal
    state = _load_json(run_dir / "state.json", {})
    state.setdefault("run_id", run_dir.name)
    state.setdefault("requested_by", user)
    state.setdefault("created", _now())
    state.setdefault("cost", {"wall_seconds": 0.0, "cpu_seconds": 0.0, "usd": 0.0, "llm_tokens": {}})
    tokens = goal.__dict__.get("llm_tokens")
    if tokens:
        state["cost"]["llm_tokens"] = tokens
    old_steps = state.get("steps", {})
    (run_dir / "workflow.json").write_text(json.dumps(wf.to_dict(), indent=2, default=str), encoding="utf-8")
    events = (run_dir / "events.jsonl").open("a", encoding="utf-8")

    def log(level, message, **details):
        events.write(json.dumps({"ts": _now(), "level": level, "message": message, **details}, default=str) + "\n")
        events.flush()

    def save():
        state["updated"] = _now()
        (run_dir / "state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    ctx = RunContext(run_dir, goal, data=data, run_id=state["run_id"], log=log)
    steps: dict = {}
    keys: dict = {}                       # step id -> output key (what downstream input hashes depend on)
    state["steps"] = steps
    state["status"] = "running"
    log("info", "run started", by=user, goal=goal.name)
    stop_reason = None
    for spec in wf.order():
        tool = TOOLS[spec.tool]
        prev = old_steps.get(spec.id, {})
        rec = {"tool": spec.tool, "status": PENDING, "optional": spec.optional, "attempts": 0}
        steps[spec.id] = rec
        if stop_reason:
            rec["status"], rec["note"] = SKIPPED, stop_reason
            continue
        blocked = [n for n in spec.needs if steps.get(n, {}).get("status") != DONE]
        if blocked:
            rec["status"], rec["note"] = BLOCKED, f"waiting for {', '.join(blocked)}"
            continue
        goal_part = goal.to_dict() if tool.goal_fields is None else {f: getattr(goal, f) for f in tool.goal_fields}
        input_hash = _hash({"tool": spec.tool, "params": spec.params, "goal": goal_part,
                            "upstream": [keys[n] for n in spec.needs]})
        rec["input_hash"] = input_hash
        if prev.get("status") == DONE and prev.get("input_hash") == input_hash and not tool.always_run:
            rec.update({k: prev[k] for k in ("summary", "started", "ended", "seconds", "cpu_seconds", "attempts")
                        if k in prev}, status=DONE, reused=True)
            keys[spec.id] = prev.get("output_key", input_hash)
            log("info", "step reused (inputs unchanged)", step=spec.id)
            continue
        if spec.approval_stage:
            need = goal.approvals_for(spec.approval_stage)
            approvals = [a for a in _load_json(run_dir / "approvals.json", [])
                         if a["step"] == spec.id and a["input_hash"] == input_hash]
            approvers = sorted({a["by"] for a in approvals if a["by"] != state["requested_by"]})
            rec["approvals"] = {"required": need, "approved_by": approvers}
            ctx.approvers = approvers                       # handed to the registry (high-risk models need two)
            if len(approvers) < need:
                rec["status"] = WAITING
                rec["note"] = (f"needs {need - len(approvers)} more approval(s) for {spec.approval_stage}: "
                               f"mlpilot approve {run_dir} --step {spec.id} --by <name>")
                log("info", "waiting for approval", step=spec.id, required=need, approved_by=approvers)
                continue
        budget = goal.budget_usd
        if budget is not None and state["cost"]["usd"] >= budget:
            stop_reason = f"budget ${budget} used (${state['cost']['usd']:.4f})"
            rec["status"], rec["note"] = SKIPPED, stop_reason
            log("warning", "budget exceeded", step=spec.id, spent=state["cost"]["usd"], budget=budget)
            continue
        rec["started"] = _now()
        log("info", "step started", step=spec.id, tool=spec.tool, params=spec.params)
        save()
        for attempt in range(1, spec.retries + 2):
            rec["attempts"] = attempt
            t0, c0 = time.perf_counter(), time.process_time()
            try:
                summary = tool.fn(ctx, **spec.params) or {}
                status, error = DONE, None
            except GateFailed as exc:
                summary, status, error = {"gate": str(exc), **exc.details}, FAILED, f"gate: {exc}"
            except Exception as exc:          # noqa: BLE001 - recorded, retried, then reported
                summary, status = {}, FAILED
                error = f"{type(exc).__name__}: {exc}"
                log("error", "step error", step=spec.id, attempt=attempt, error=error,
                    traceback=traceback.format_exc(limit=8))
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            state["cost"]["wall_seconds"] += wall
            state["cost"]["cpu_seconds"] += cpu
            state["cost"]["usd"] = round(state["cost"]["cpu_seconds"] / 3600 * goal.cpu_hour_usd, 6)
            rec["seconds"] = round(rec.get("seconds", 0) + wall, 3)
            rec["cpu_seconds"] = round(rec.get("cpu_seconds", 0) + cpu, 3)
            if status == DONE or (error or "").startswith("gate:") or attempt > spec.retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
            log("warning", "retrying", step=spec.id, attempt=attempt + 1)
        rec.update(status=status, ended=_now(), summary=summary)
        if error:
            rec["error"] = error
        fingerprint = summary.pop("_fingerprint", None) if isinstance(summary, dict) else None
        if fingerprint:
            rec["data_fingerprint"] = fingerprint
        rec["output_key"] = keys[spec.id] = fingerprint or input_hash
        log("info" if status == DONE else "error", f"step {status}", step=spec.id,
            seconds=rec["seconds"], error=error)
        if status == FAILED and not spec.optional:
            stop_reason = f"stopped after {spec.id} failed"
        elif status == FAILED:
            rec["status"] = DONE                    # optional: downstream may continue; error kept in the record
            rec["note"] = f"optional step failed: {error}"
        save()
    statuses = [v["status"] for v in steps.values()]
    if any(v["status"] == FAILED for v in steps.values()):
        state["status"] = "failed"
    elif stop_reason and "budget" in stop_reason:
        state["status"] = "budget_exceeded"
    elif WAITING in statuses:
        state["status"] = WAITING
    else:
        state["status"] = "succeeded"
    ingest = steps.get("ingest", {})
    if not (run_dir / "manifest.json").exists() or ingest.get("summary"):
        (run_dir / "manifest.json").write_text(json.dumps(_manifest(goal, ingest.get("data_fingerprint")), indent=2),
                                               encoding="utf-8")
    log("info", "run finished", status=state["status"])
    save()
    events.close()
    result = RunResult(run_dir, state["status"], steps, state["cost"])
    _write_report(result, wf, state)
    return result


def resume(run_dir, *, data=None, requested_by: str | None = None) -> RunResult:
    """Continue a run from its directory (after an approval, a fix, a crash or a goal edit)."""
    run_dir = Path(run_dir)
    wf = Workflow.from_dict(_load_json(run_dir / "workflow.json", None) or {})
    state = _load_json(run_dir / "state.json", {})
    return run(wf, run_dir, data=data, requested_by=requested_by or state.get("requested_by"))


def approve(run_dir, step: str, *, by: str, note: str = "") -> dict:
    """Record an approval for the *current* inputs of ``step`` (a changed plan needs new approvals).
    The person who started the run can't approve it (four-eyes)."""
    run_dir = Path(run_dir)
    state = _load_json(run_dir / "state.json", None)
    if state is None:
        raise FileNotFoundError(f"{run_dir} is not a run directory")
    rec = state.get("steps", {}).get(step)
    if not rec or rec.get("status") != WAITING:
        raise ValueError(f"step {step!r} is not waiting for approval (status: {rec and rec.get('status')})")
    if by == state.get("requested_by"):
        raise PermissionError("The person who started the run can't approve it (four-eyes rule)")
    entry = {"step": step, "input_hash": rec["input_hash"], "by": by, "note": note, "at": _now()}
    approvals = _load_json(run_dir / "approvals.json", [])
    approvals.append(entry)
    (run_dir / "approvals.json").write_text(json.dumps(approvals, indent=2), encoding="utf-8")
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": entry["at"], "level": "info", "message": "approved", **entry}) + "\n")
    return entry


def status(run_dir) -> RunResult:
    run_dir = Path(run_dir)
    state = _load_json(run_dir / "state.json", None)
    if state is None:
        raise FileNotFoundError(f"{run_dir} is not a run directory")
    return RunResult(run_dir, state.get("status", "unknown"), state.get("steps", {}), state.get("cost", {}))


def _write_report(result: RunResult, wf: Workflow, state: dict):
    g = wf.goal
    lines = [f"# Run `{result.run_dir.name}`: {result.status}", "",
             f"**Goal:** {g.description or f'predict {g.target}'}  ", f"**Model name:** `{g.name}` · "
             f"**target:** `{g.target}` · **requested by:** {state.get('requested_by')}  ",
             f"**Cost:** {result.cost.get('wall_seconds', 0):.0f} s wall, {result.cost.get('cpu_seconds', 0):.0f} CPU-s, "
             f"${result.cost.get('usd', 0):.4f} at ${g.cpu_hour_usd}/CPU-hour", ""]
    if g.min_metrics or g.max_metrics or g.max_latency_ms or g.max_model_mb or g.budget_usd:
        lines += ["## Requirements", ""]
        lines += [f"- {k} ≥ {v}" for k, v in (g.min_metrics or {}).items()]
        lines += [f"- {k} ≤ {v}" for k, v in (g.max_metrics or {}).items()]
        if g.max_latency_ms:
            lines.append(f"- p95 latency ≤ {g.max_latency_ms} ms")
        if g.max_model_mb:
            lines.append(f"- model size ≤ {g.max_model_mb} MB")
        if g.budget_usd:
            lines.append(f"- compute budget ≤ ${g.budget_usd}")
        lines.append("")
    lines += ["## Steps", "", "| Step | Status | Seconds | Result |", "|---|---|---|---|"]
    for k, v in result.steps.items():
        summary = v.get("error") or v.get("note") or json.dumps(v.get("summary", {}), default=str)
        summary = summary.replace("|", "/").replace("\n", " ")
        lines.append(f"| {k} | {v['status']}{' (reused)' if v.get('reused') else ''} | {v.get('seconds', '')} | "
                     f"{summary[:300]} |")
    lines += ["", "Files: `workflow.json`, `state.json`, `events.jsonl`, `approvals.json`, `manifest.json`, "
              "`artifacts/` (splits, model, metrics, model card, monitor baseline)."]
    (result.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
