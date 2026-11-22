"""The goal of an autonomous run: what to build, on which data, under which limits.

A goal is plain data (YAML / JSON / dict), validated before anything runs. It can be written by
hand or drafted from a sentence by :func:`goal_from_text`; either way the same deterministic
engine executes it, with the same quality, fairness and approval gates.

>>> goal = Goal(name="churn", data="data/customers.csv", target="churned",
...             min_metrics={"roc_auc": 0.85}, max_latency_ms=50, deploy="staging")
>>> goal = Goal.from_file("goal.yml")
>>> goal = goal_from_text("Predict churned from data/customers.csv with AUC above 0.85, deploy to staging")
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

STAGES = ("none", "dev", "staging", "production")
LOWER_IS_BETTER = {"rmse", "mae", "mape", "smape", "log_loss", "brier", "ece", "max_error", "median_ae"}
_METRIC_WORDS = {"auc": "roc_auc", "roc auc": "roc_auc", "roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1",
                 "f1 score": "f1", "precision": "precision", "recall": "recall", "r2": "r2", "r²": "r2",
                 "rmse": "rmse", "mae": "mae", "mape": "mape", "log loss": "log_loss"}


class GoalError(ValueError):
    """The goal is incomplete or inconsistent; the message lists every problem."""


@dataclasses.dataclass
class Goal:
    name: str                                  # model / project name (registry name)
    data: str | None                           # path, glob, URL or cloud path readable by mlpilot.read
    target: str
    task: str = "auto"                         # auto | binary | multiclass | regression | classification
    metric: str | None = None                  # metric AutoML optimises (default: chosen for the task)
    min_metrics: dict = dataclasses.field(default_factory=dict)    # test metric floors, e.g. {"roc_auc": 0.85}
    max_metrics: dict = dataclasses.field(default_factory=dict)    # test metric ceilings, e.g. {"rmse": 4.0}
    time_budget: float = 120.0                 # AutoML search seconds
    include: list | None = None                # AutoML model families (default: all available)
    max_latency_ms: float | None = None        # per-row prediction latency limit
    max_model_mb: float | None = None          # saved model size limit
    budget_usd: float | None = None            # compute budget for the whole run
    cpu_hour_usd: float = 0.05                 # rate used to price CPU time
    test_size: float = 0.2
    group: str | None = None                   # keep groups (customers, patients …) on one side of the split
    time: str | None = None                    # split by time instead of at random
    contract: str | None = None                # data contract (YAML / JSON) the data must pass
    min_quality: float | None = 50.0           # data-quality score floor (0–100)
    max_suspicious_share: float | None = None  # poisoning / label-error limit for the training rows
    sensitive: list = dataclasses.field(default_factory=list)      # columns checked for fairness
    fairness_limits: dict | None = None
    robustness: bool = True
    max_membership_auc: float | None = None    # privacy gate: membership-inference attack AUC limit (e.g. 0.6)
    explain: bool = True
    deploy: str = "staging"                    # none | dev | staging | production
    approvals: dict = dataclasses.field(default_factory=lambda: {"production": 1})   # approvers per stage
    registry: str = "registry"
    monitor: bool = True
    seed: int = 0
    description: str = ""

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - fields)
        if unknown:
            raise GoalError(f"Unknown goal field(s) {unknown}; valid fields: {sorted(fields)}")
        goal = cls(**data)
        goal.validate()
        return goal

    @classmethod
    def from_file(cls, path) -> "Goal":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yml", ".yaml"):
            import yaml
            return cls.from_dict(yaml.safe_load(text) or {})
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_yaml(self, path=None) -> str:
        import yaml
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text

    # ── validation ──────────────────────────────────────────────────────────
    def validate(self) -> "Goal":
        problems = []
        if not self.name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", str(self.name)):
            problems.append("name must be 1-64 letters, digits, '.', '_' or '-'")
        if not self.target:
            problems.append("target is required")
        if self.task not in ("auto", "binary", "multiclass", "regression", "classification"):
            problems.append("task must be auto, binary, multiclass, regression or classification")
        if self.deploy not in STAGES:
            problems.append(f"deploy must be one of {STAGES}")
        if not 0 < float(self.test_size) < 1:
            problems.append("test_size must be between 0 and 1")
        if self.group and self.time:
            problems.append("use either group or time to split, not both")
        if float(self.time_budget) <= 0:
            problems.append("time_budget must be positive")
        for label, metrics in (("min_metrics", self.min_metrics), ("max_metrics", self.max_metrics)):
            if not isinstance(metrics, dict) or not all(isinstance(v, (int, float)) for v in metrics.values()):
                problems.append(f"{label} must map metric names to numbers")
        for k in self.min_metrics or {}:
            if k in LOWER_IS_BETTER:
                problems.append(f"{k} is lower-is-better: put it in max_metrics, not min_metrics")
        for limit in ("max_latency_ms", "max_model_mb", "budget_usd", "max_suspicious_share"):
            v = getattr(self, limit)
            if v is not None and float(v) <= 0:
                problems.append(f"{limit} must be positive")
        if self.max_membership_auc is not None and not 0.5 <= float(self.max_membership_auc) <= 1:
            problems.append("max_membership_auc is an attack AUC between 0.5 (no leakage) and 1")
        if self.min_quality is not None and not 0 <= float(self.min_quality) <= 100:
            problems.append("min_quality is a 0-100 score")
        if not isinstance(self.approvals, dict) or any(int(n) < 0 for n in self.approvals.values()):
            problems.append("approvals maps stage -> number of approvers (>= 0)")
        if problems:
            raise GoalError("Invalid goal:\n  - " + "\n  - ".join(problems))
        return self

    def approvals_for(self, stage: str) -> int:
        return int((self.approvals or {}).get(stage, 0))


# ── drafting a goal from a sentence ─────────────────────────────────────────
_NUM = r"(\d+(?:\.\d+)?)"


def _rules(text: str, data: str | None) -> dict:
    """Deterministic extraction of the common phrasings (no LLM)."""
    t = " " + text.strip() + " "
    low = t.lower()
    out: dict = {}
    m = (re.search(r"\b(?:predict|predicting|forecast|classify|estimate)\s+(?:the\s+|whether\s+)?[`'\"]?([A-Za-z_][\w.]*)", t, re.I)
         or re.search(r"\btarget(?:\s+column)?\s*(?:is|=|:)?\s*[`'\"]?([A-Za-z_][\w.]*)", t, re.I))
    if m:
        out["target"] = m.group(1).strip(".,")
    m = re.search(r"\b(?:from|using|in|on)\s+(?:my\s+|the\s+)?[`'\"]?((?:[a-z]+://|[\w./\\-]+[/\\])?[\w./\\-]*\.(?:csv|parquet|json|ndjson|xlsx|xls))", t, re.I)
    if m and not data:
        out["data"] = m.group(1)
    for word, metric in sorted(_METRIC_WORDS.items(), key=lambda kv: -len(kv[0])):
        m = re.search(rf"\b{re.escape(word)}\b[^.\d]{{0,30}}?(above|over|greater than|at least|>=|>|of|below|under|less than|at most|<=|<)\s*{_NUM}", low)
        if m:
            value, direction = float(m.group(2)), m.group(1)
            lower = metric in LOWER_IS_BETTER or direction in ("below", "under", "less than", "at most", "<=", "<")
            out.setdefault("max_metrics" if lower else "min_metrics", {})[metric] = value
            if not lower:
                out.setdefault("metric", metric)
            break
    m = re.search(rf"latency[^.\d]{{0,25}}?(?:below|under|less than|<|<=|of|within)?\s*{_NUM}\s*ms", low)
    if m:
        out["max_latency_ms"] = float(m.group(1))
    m = re.search(rf"(?:budget|cost)[^.\d$]{{0,20}}?\$\s*{_NUM}|\$\s*{_NUM}\s*(?:budget|max)", low)
    if m:
        out["budget_usd"] = float(m.group(1) or m.group(2))
    m = re.search(rf"(?:within|in|under|budget of)\s*{_NUM}\s*(minutes?|mins?|hours?|h|seconds?|s)\b", low)
    if m:
        v, unit = float(m.group(1)), m.group(2)
        out["time_budget"] = v * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)
    if re.search(r"\b(production|prod|real-?time api|live)\b", low):
        out["deploy"] = "production"
    elif "staging" in low:
        out["deploy"] = "staging"
    elif re.search(r"\b(don'?t|do not|no)\s+deploy", low):
        out["deploy"] = "none"
    m = re.search(r"\bfair(?:ness|ly)?\b[^.]{0,40}?\b(?:across|by|for|on)\s+([A-Za-z_][\w]*(?:\s*(?:,|and)\s*[A-Za-z_][\w]*)*)", t, re.I)
    if m:
        out["sensitive"] = [s.strip() for s in re.split(r",|\band\b", m.group(1)) if s.strip()]
    m = re.search(r"\b(?:grouped|group(?:ed)?\s+by|per)\s+([A-Za-z_]\w*)", t)
    if m and "fair" not in m.group(0).lower():
        out["group"] = m.group(1)
    if re.search(r"\b(classif|churn|fraud|whether|yes/no|binary)\w*", low):
        out.setdefault("task", "auto")
    return out


def _llm_draft(text: str, model: str) -> dict:
    import anthropic
    schema = {f.name: str(f.type) for f in dataclasses.fields(Goal)}
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model, max_tokens=1500,
        system=("You turn a data scientist's request into a JSON goal for the MLPilot engine. Reply with one JSON "
                "object and nothing else. Use only these fields: " + json.dumps(schema) + ". Metric names: roc_auc, "
                "accuracy, f1, precision, recall, r2 (min_metrics) and rmse, mae, mape, log_loss (max_metrics). "
                "Leave out anything the request doesn't say."),
        messages=[{"role": "user", "content": text}])
    raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON in the reply")
    usage = getattr(response, "usage", None)
    draft = json.loads(m.group(0))
    draft["_tokens"] = {"input": getattr(usage, "input_tokens", 0), "output": getattr(usage, "output_tokens", 0)}
    return draft


def goal_from_text(text: str, *, data: str | None = None, name: str | None = None, use_llm: bool | str = "auto",
                   model: str = "claude-opus-5-5", **overrides) -> Goal:
    """Draft a :class:`Goal` from a sentence such as *"Predict churned from customers.csv with AUC
    above 0.85 and latency below 50 ms; deploy to staging"*.

    Common phrasings are understood by deterministic rules. With ``use_llm`` (default: when
    ``ANTHROPIC_API_KEY`` is set), Claude drafts the remaining fields; whatever it returns is
    validated like a hand-written goal, and rule-based values win where both exist. The LLM only
    proposes a goal: it never runs anything. ``overrides`` set fields explicitly."""
    import os
    draft: dict = {}
    if use_llm is True or (use_llm == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        try:
            draft = _llm_draft(text, model)
        except Exception:                      # no key, no network, odd reply: the rules still apply
            draft = {}
    tokens = draft.pop("_tokens", None)
    fields = {f.name for f in dataclasses.fields(Goal)}
    draft = {k: v for k, v in draft.items() if k in fields and v is not None}
    draft.update(_rules(text, data))
    if data:
        draft["data"] = data
    draft.update(overrides)
    draft.setdefault("name", name or re.sub(r"[^A-Za-z0-9_.-]+", "-", str(draft.get("target", "model"))).strip("-") or "model")
    draft.setdefault("data", None)
    draft.setdefault("target", "")
    draft.setdefault("description", text.strip())
    if not draft["target"]:
        raise GoalError("Couldn't find the target in the request: say e.g. 'predict churned' or pass target=")
    goal = Goal.from_dict(draft)
    if tokens:
        goal.__dict__["llm_tokens"] = tokens            # reported by the engine's cost accounting
    return goal
