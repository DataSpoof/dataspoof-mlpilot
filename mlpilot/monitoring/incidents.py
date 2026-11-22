"""Incident management: diagnose a production problem, choose a playbook, act safely, verify recovery.

>>> diag = diagnose(monitor=monitor.check(batch), data=data_monitor.observe(batch),
...                 service={"latency_ms_p95": 180, "error_rate": 0.002}, slo={"latency_ms_p95": 100})
>>> print(diag.summary())          # root cause, confidence, evidence, recommended actions
>>> log = IncidentLog("incidents.jsonl")
>>> inc = log.open(diag, model="churn")
>>> log.remediate(inc, handlers={"retrain": start_retrain_run, "notify": page_on_call})
>>> log.approve(inc, "rollback", by="priya")          # high-risk actions wait for a person
>>> log.verify(inc, monitor.check(next_batch))        # resolved only when the next check is healthy

**Root causes**, checked in this order, because the first one that fits decides the playbook:

| Cause | Evidence | Playbook |
|---|---|---|
| `upstream_data` | stale data, missing or retyped columns, a volume collapse, null spikes | pause retraining, notify the data owner |
| `infrastructure` | latency or error-rate SLO breached, model and data look normal | scale out (automatic), restart workers |
| `bad_release` | performance fell right after a new version went live | roll back (needs approval) |
| `concept_drift` | measured accuracy fell, inputs did not shift | retrain on recent labelled data |
| `covariate_shift` | inputs and predictions shifted (with or without labels) | retrain; review features |

**Safety rules:**
- Low-risk actions (notify, pause, scale, start a retraining *run*) can run automatically.
- High-risk actions (rollback, promoting a retrained model) need an approver.
- Retraining never means automatic promotion: a retrained model goes through the same gates as
  any other.
- An incident closes only when a later check is healthy.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import uuid
from pathlib import Path
from typing import Callable

ACTIONS = {   # name: (risk, description)
    "notify": ("low", "alert the owners (on-call, data owner)"),
    "pause_retraining": ("low", "stop scheduled retraining until the data is fixed (don't learn from broken data)"),
    "scale_out": ("low", "add serving capacity"),
    "restart_workers": ("low", "restart unhealthy serving workers"),
    "retrain": ("low", "start a retraining run (its promotion still needs the usual gates and approvals)"),
    "review_features": ("low", "ask the owners to review the drifted features"),
    "rollback": ("high", "move production back to the previous version"),
}


@dataclasses.dataclass
class Action:
    name: str
    risk: str
    reason: str
    status: str = "proposed"          # proposed | done | failed | awaiting_approval | no_handler
    result: str | None = None
    approved_by: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Diagnosis:
    cause: str                        # upstream_data | infrastructure | bad_release | concept_drift | covariate_shift | healthy
    confidence: str                   # high | medium | low
    evidence: list
    actions: list

    @property
    def healthy(self) -> bool:
        return self.cause == "healthy"

    def summary(self) -> str:
        lines = [f"Diagnosis: {self.cause} (confidence {self.confidence})"]
        lines += [f"  evidence: {e}" for e in self.evidence]
        lines += [f"  action [{a.risk}]: {a.name} - {a.reason}" for a in self.actions]
        return "\n".join(lines)


def _action(name: str, reason: str | None = None) -> Action:
    risk, desc = ACTIONS[name]
    return Action(name, risk, reason or desc)


def _alerts(monitor) -> list:
    if monitor is None:
        return []
    alerts = monitor.alerts if hasattr(monitor, "alerts") else monitor.get("alerts", [])
    return [a if isinstance(a, dict) else dataclasses.asdict(a) for a in alerts]


def diagnose(monitor=None, data=None, service: dict | None = None, slo: dict | None = None, *,
             released_hours_ago: float | None = None, release_window_hours: float = 48.0) -> Diagnosis:
    """Find the most likely root cause from the signals you have (any may be missing).

    ``monitor``: a ``MonitorReport`` (or its ``to_dict()``). ``data``: a ``DataMonitor``
    ``ObservationReport``. ``service``: current service metrics (``latency_ms_p95``,
    ``error_rate``, ``availability``). ``slo``: limits for those metrics (availability is a
    floor). ``released_hours_ago``: how long the current model version has been live."""
    alerts = _alerts(monitor)
    kinds = {a["kind"]: a for a in alerts}
    evidence: list = []

    # 1. upstream data: the inputs themselves are broken
    data_errors = [i for i in (data.issues if data is not None else []) if i.severity == "error"]
    null_spikes = [i for i in (data.issues if data is not None else []) if i.check == "nulls"]
    upstream = [str(i) for i in data_errors + null_spikes]
    if "schema" in kinds and kinds["schema"]["severity"] == "critical":
        upstream.append(f"monitor: {kinds['schema']['message']}")
    if "volume" in kinds:
        upstream.append(f"monitor: {kinds['volume']['message']}")
    if upstream:
        return Diagnosis("upstream_data", "high" if data_errors or "schema" in kinds else "medium", upstream,
                         [_action("pause_retraining"), _action("notify", "tell the data owner what broke")])

    # 2. infrastructure: the service breaches its SLOs while model and data look normal
    slo, service = slo or {}, service or {}
    breaches = []
    for k, limit in slo.items():
        v = service.get(k)
        if v is None:
            continue
        if (k == "availability" and v < limit) or (k != "availability" and v > limit):
            breaches.append(f"{k} = {v} (SLO {limit})")
    model_signals = [a for a in alerts if a["kind"] in ("performance", "estimated_performance", "prediction_drift",
                                                        "data_drift")]
    if breaches and not model_signals:
        return Diagnosis("infrastructure", "high", breaches,
                         [_action("scale_out"), _action("restart_workers"), _action("notify")])

    perf = kinds.get("performance") or kinds.get("estimated_performance")
    drifted = "data_drift" in kinds or "prediction_drift" in kinds or (
        monitor is not None and getattr(monitor, "prediction_drift", None)
        and monitor.prediction_drift.get("severity") == "major")
    # 3. a bad release: quality fell right after a new version went live
    if perf and released_hours_ago is not None and released_hours_ago <= release_window_hours:
        evidence = [f"{perf['kind']}: {perf['message']}", f"current version live for {released_hours_ago:g} h"]
        return Diagnosis("bad_release", "high" if perf["kind"] == "performance" else "medium", evidence + breaches,
                         [_action("rollback", "restore the previous production version"), _action("notify")])
    # 4. concept drift: measured accuracy fell although the inputs look the same
    if perf and perf["kind"] == "performance" and not drifted:
        return Diagnosis("concept_drift", "medium", [f"performance: {perf['message']}", "inputs did not drift"] + breaches,
                         [_action("retrain", "retrain on recent labelled data"), _action("notify")])
    # 5. covariate shift: inputs / predictions moved (with or without measured decay)
    if drifted or perf:
        ev = [f"{a['kind']}: {a['message']}" for a in model_signals]
        confidence = "high" if perf and drifted else "medium"
        return Diagnosis("covariate_shift", confidence, ev + breaches,
                         [_action("retrain", "retrain including the recent data"), _action("review_features"),
                          _action("notify")])
    if breaches:
        return Diagnosis("infrastructure", "medium", breaches, [_action("scale_out"), _action("notify")])
    return Diagnosis("healthy", "high", ["no alerts, no SLO breaches"], [])


# ── incidents ───────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Incident:
    id: str
    model: str | None
    opened: str
    cause: str
    confidence: str
    evidence: list
    actions: list
    status: str = "open"              # open | mitigating | awaiting_approval | resolved
    resolved: str | None = None
    history: list = dataclasses.field(default_factory=list)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class IncidentLog:
    """Incidents and everything done about them, in an append-only JSONL file (the latest record
    of an incident wins)."""

    def __init__(self, path="incidents.jsonl", *, registry=None):
        self.path = Path(path)
        self.registry = registry                       # a ModelRegistry (or path) for the built-in rollback

    def _write(self, inc: Incident):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(inc), default=str) + "\n")

    def get(self, incident_id: str) -> Incident:
        latest = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["id"] == incident_id:
                latest = rec
        if latest is None:
            raise KeyError(incident_id)
        latest["actions"] = [Action(**a) for a in latest["actions"]]
        return Incident(**latest)

    def incidents(self, status: str | None = None) -> list:
        ids: dict = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                ids[json.loads(line)["id"]] = None
        out = [self.get(i) for i in ids]
        return [i for i in out if status is None or i.status == status]

    def open(self, diagnosis: Diagnosis, *, model: str | None = None) -> Incident:
        if diagnosis.healthy:
            raise ValueError("nothing to open: the diagnosis is healthy")
        inc = Incident(f"INC-{uuid.uuid4().hex[:8]}", model, _now(), diagnosis.cause, diagnosis.confidence,
                       list(diagnosis.evidence), [dataclasses.replace(a) for a in diagnosis.actions])
        inc.history.append({"at": inc.opened, "event": "opened", "cause": inc.cause})
        self._write(inc)
        return inc

    def _builtin(self, name: str, inc: Incident):
        if name == "rollback" and self.registry is not None and inc.model:
            from ..registry import ModelRegistry
            reg = self.registry if isinstance(self.registry, ModelRegistry) else ModelRegistry(self.registry)
            return lambda incident: f"rolled back to version {reg.rollback(incident.model, reason=incident.id).version}"
        if name == "pause_retraining":
            flag = self.path.parent / f"retraining_paused_{inc.model or 'all'}.flag"
            return lambda incident: (flag.write_text(incident.id, encoding="utf-8"), f"wrote {flag.name}")[1]
        return None

    def remediate(self, inc: Incident, handlers: dict | None = None) -> Incident:
        """Run the incident's actions. ``handlers`` maps action names to ``fn(incident) -> str``.
        Low-risk actions run now; high-risk ones run only once approved (:meth:`approve`)."""
        handlers = handlers or {}
        for a in inc.actions:
            if a.status in ("done",):
                continue
            if a.risk == "high" and not a.approved_by:
                a.status = "awaiting_approval"
                continue
            fn: Callable | None = handlers.get(a.name) or self._builtin(a.name, inc)
            if fn is None:
                a.status, a.result = "no_handler", "do this manually, or pass a handler"
                continue
            try:
                a.result, a.status = str(fn(inc)), "done"
            except Exception as exc:                  # noqa: BLE001 - a failed action is recorded, not fatal
                a.result, a.status = f"{type(exc).__name__}: {exc}", "failed"
            inc.history.append({"at": _now(), "event": f"action {a.name}", "status": a.status, "result": a.result})
        inc.status = "awaiting_approval" if any(a.status == "awaiting_approval" for a in inc.actions) else "mitigating"
        self._write(inc)
        return inc

    def approve(self, inc: Incident, action: str, *, by: str, handlers: dict | None = None) -> Incident:
        """Approve a high-risk action and run it."""
        a = next((x for x in inc.actions if x.name == action), None)
        if a is None:
            raise KeyError(f"incident {inc.id} has no action {action!r}")
        a.approved_by.append(by)
        inc.history.append({"at": _now(), "event": f"approved {action}", "by": by})
        return self.remediate(inc, handlers)

    def verify(self, inc: Incident, monitor=None, data=None, service=None, slo=None) -> Incident:
        """Close the incident when a fresh check is healthy; otherwise keep it open with the new evidence."""
        check = diagnose(monitor, data, service, slo)
        inc.history.append({"at": _now(), "event": "verification", "cause": check.cause, "evidence": check.evidence})
        if check.healthy:
            inc.status, inc.resolved = "resolved", _now()
        else:
            inc.status = "open"
        self._write(inc)
        return inc
