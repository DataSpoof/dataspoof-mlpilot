"""Tools: the only operations a workflow can perform.

Every step of a plan names a registered tool. A planner (rules or an LLM) can only choose among
these, with typed parameters; it cannot run arbitrary code, skip a gate or touch production
without the approval the goal requires. Each tool wraps an existing MLPilot function, reads and
writes artifacts through the :class:`RunContext`, and returns a small JSON summary.

Gates raise :class:`GateFailed`: the run stops there with the reason, and is not retried
(a failed gate is a decision, not a transient error).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl


class GateFailed(Exception):
    """A quality / fairness / performance gate rejected the run (not retried)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclasses.dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str
    risk: str = "low"                 # "high": changes shared state others rely on (registry stages)
    always_run: bool = False          # re-run on every resume (e.g. reading data that may have changed)
    goal_fields: tuple | None = None  # goal fields the tool reads (None: all); others don't trigger a re-run


TOOLS: dict = {}


def register_tool(name: str, description: str, *, risk: str = "low", always_run: bool = False,
                  goal_fields: tuple | None = None):
    """Decorator adding a tool. ``fn(ctx, **params) -> dict``. ``goal_fields`` lists the goal
    fields the tool reads, so editing any other field doesn't re-run it on resume."""
    if risk not in ("low", "high"):
        raise ValueError("risk must be 'low' or 'high'")

    def deco(fn):
        TOOLS[name] = ToolSpec(name, fn, description, risk, always_run, tuple(goal_fields) if goal_fields else None)
        return fn
    return deco


def frame_fingerprint(df: pl.DataFrame) -> str:
    """Content hash of a DataFrame (schema + every value)."""
    h = hashlib.sha256(json.dumps({c: str(t) for c, t in df.schema.items()}).encode())
    if df.height:
        h.update(df.hash_rows(seed=0).to_numpy().tobytes())
    return h.hexdigest()[:16]


class RunContext:
    """Artifacts of a run: DataFrames (Parquet), models (saved directories) and JSON, kept on
    disk under ``<run>/artifacts`` so a resumed run picks up where it stopped."""

    def __init__(self, run_dir, goal, *, data: pl.DataFrame | None = None, run_id: str = "", log=None):
        self.dir = Path(run_dir)
        self.art = self.dir / "artifacts"
        self.art.mkdir(parents=True, exist_ok=True)
        self.goal, self.input_frame, self.run_id = goal, data, run_id
        self._cache: dict = {}
        self._log = log or (lambda *a, **k: None)
        self.approvers: list = []          # people who approved the step being run (set by the engine)

    def log(self, message: str, **details):
        self._log("info", message, **details)

    def put(self, name: str, obj):
        if isinstance(obj, pl.DataFrame):
            obj.write_parquet(self.art / f"{name}.parquet")
        elif hasattr(obj, "save") and hasattr(obj, "predict"):
            target = self.art / name
            if target.exists():
                shutil.rmtree(target)
            obj.save(target)
        else:
            (self.art / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        self._cache[name] = obj
        return obj

    def get(self, name: str):
        if name in self._cache:
            return self._cache[name]
        if (self.art / f"{name}.parquet").exists():
            obj = pl.read_parquet(self.art / f"{name}.parquet")
        elif (self.art / name / "manifest.json").exists():
            from ..automl import load_model
            obj = load_model(self.art / name)
        elif (self.art / f"{name}.json").exists():
            obj = json.loads((self.art / f"{name}.json").read_text(encoding="utf-8"))
        else:
            raise KeyError(f"artifact {name!r} not found (has the step that makes it run?)")
        self._cache[name] = obj
        return obj

    def path(self, name: str) -> Path:
        return self.art / name


# ── built-in tools ──────────────────────────────────────────────────────────
@register_tool("ingest", "Read the goal's data source (or the DataFrame passed to the run) and check the target exists",
               always_run=True, goal_fields=("data", "target"))
def _ingest(ctx: RunContext) -> dict:
    g = ctx.goal
    if ctx.input_frame is not None:
        df, source = ctx.input_frame, "<DataFrame passed to run()>"
    elif not g.data and (ctx.art / "data.parquet").exists():
        # resumed without a source: continue with the exact snapshot this run already read
        df, source = pl.read_parquet(ctx.art / "data.parquet"), "<snapshot stored by this run>"
    else:
        if not g.data:
            raise GateFailed("The goal has no data source: set goal.data or pass data= to run()")
        from ..ingestion import read
        df = read(str(g.data))
        df = df.collect() if isinstance(df, pl.LazyFrame) else df
        source = str(g.data)
    if g.target not in df.columns:
        raise GateFailed(f"Target column {g.target!r} is not in the data (columns: {df.columns[:20]})")
    if df[g.target].null_count() == df.height:
        raise GateFailed(f"Target column {g.target!r} is empty")
    ctx.put("data", df)
    return {"source": source, "rows": df.height, "columns": df.width, "_fingerprint": frame_fingerprint(df),
            "target_missing": int(df[g.target].null_count())}


@register_tool("quality_gate", "Score data quality (0-100) and validate the data contract; stop below the goal's floor", goal_fields=("contract", "min_quality"))
def _quality(ctx: RunContext) -> dict:
    from ..quality import check_quality
    g, df = ctx.goal, ctx.get("data")
    rep = check_quality(df)
    (ctx.art / "quality.txt").write_text(rep.summary(), encoding="utf-8")
    out = {"score": round(float(rep.score), 1), "grade": rep.grade}
    if g.contract:
        from ..quality import Contract
        res = Contract.from_file(g.contract).validate(df)
        out["contract"] = {"passed": res.passed, "errors": len(res.errors), "warnings": len(res.warnings)}
        if not res.passed:
            raise GateFailed(f"Data contract failed with {len(res.errors)} error(s)", out)
    if g.min_quality is not None and rep.score < g.min_quality:
        raise GateFailed(f"Data quality {rep.score:.1f} is below the goal's floor {g.min_quality}", out)
    return out


@register_tool("split", "Leak-checked train/test split (stratified, grouped or by time)", goal_fields=("target", "test_size", "group", "time", "seed"))
def _split(ctx: RunContext) -> dict:
    from ..splitting import split_data
    g = ctx.goal
    df = ctx.get("data").filter(pl.col(g.target).is_not_null())
    s = split_data(df, g.target, test_size=g.test_size, group=g.group, time=g.time, seed=g.seed)
    ctx.put("train", s.train)
    ctx.put("test", s.test)
    return {"strategy": s.strategy, "train_rows": s.train.height, "test_rows": s.test.height,
            "issues": [str(i) for i in s.issues]}


@register_tool("poisoning_check", "Look for label errors, conflicting duplicates and backdoor triggers in the training rows", goal_fields=("target", "max_suspicious_share", "seed"))
def _poisoning(ctx: RunContext) -> dict:
    from ..poisoning import poisoning_report
    g, train = ctx.goal, ctx.get("train")
    rep = poisoning_report(train, g.target, seed=g.seed)
    share = rep.suspicious.height / max(train.height, 1)
    out = {"suspicious_rows": rep.suspicious.height, "suspicious_share": round(share, 4)}
    if g.max_suspicious_share is not None and share > g.max_suspicious_share:
        raise GateFailed(f"{share:.1%} of training rows look mislabelled or poisoned "
                         f"(limit {g.max_suspicious_share:.1%})", out)
    return out


def _measure(model, X: pl.DataFrame, repeats: int = 30) -> dict:
    """Single-row latency percentiles (ms) and saved size (MB) of a fitted model."""
    import tempfile
    model.predict(X.head(1))                            # warm-up
    times = []
    for i in range(repeats):
        r = X.slice(i % max(X.height, 1), 1)
        t = time.perf_counter()
        model.predict(r)
        times.append((time.perf_counter() - t) * 1000)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(Path(tmp) / "m")
        size = _dir_size(Path(tmp) / "m") / 1e6
    return {"latency_ms_p50": round(float(np.percentile(times, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(times, 95)), 2), "model_mb": round(size, 3)}


def _fits(g, m: dict) -> list:
    problems = []
    if g.max_latency_ms is not None and m["latency_ms_p95"] > g.max_latency_ms:
        problems.append(f"p95 latency {m['latency_ms_p95']} ms > {g.max_latency_ms} ms")
    if g.max_model_mb is not None and m["model_mb"] > g.max_model_mb:
        problems.append(f"size {m['model_mb']:.2f} MB > {g.max_model_mb} MB")
    return problems


@register_tool("automl", "Train models with AutoML within the goal's time budget; with latency or size limits, "
               "search for the best model that meets them (cost-aware)",
               goal_fields=("target", "task", "metric", "time_budget", "include", "seed", "max_latency_ms", "max_model_mb"))
def _automl(ctx: RunContext, max_attempts: int = 4) -> dict:
    from ..automl import AutoML
    g = ctx.goal
    train = ctx.get("train")
    constrained = g.max_latency_ms is not None or g.max_model_mb is not None
    families = list(g.include) if g.include else None
    excluded: list = []
    ensemble = True
    attempts = []
    model = None
    for attempt in range(1, (max_attempts if constrained else 1) + 1):
        kw: dict = {"task": g.task, "seed": g.seed, "verbose": False, "ensemble": ensemble,
                    "time_budget": float(g.time_budget) * (1.0 if attempt == 1 else 0.5)}
        if families:
            kw["include"] = [f for f in families if f not in excluded]
        elif excluded:
            kw["exclude"] = excluded
        if g.metric:
            kw["metric"] = g.metric
        if "include" in kw and not kw["include"]:
            break
        model = AutoML(g.target, **kw).fit(train)
        rec = {"attempt": attempt, "best": model.best_["model"], "cv_score": round(float(model.best_["score"]), 4),
               "ensemble": ensemble, "excluded": list(excluded)}
        if not constrained:
            attempts.append(rec)
            break
        rec.update(_measure(model, train.drop(g.target)))
        problems = _fits(g, rec)
        rec["meets_limits"] = not problems
        attempts.append(rec)
        ctx.log("automl attempt", **rec)
        if not problems:
            break
        best = model.best_["model"]                     # too slow / too big: drop what made it so and search again
        if best == "ensemble" and ensemble:
            ensemble = False
        else:
            excluded.append(best)
    ctx.put("model", model)
    board = model.leaderboard_
    return {"best": model.best_["model"], "cv_score": float(model.best_["score"]), "metric": str(model.metric_),
            "trials": len(model.trials_), "attempts": attempts,
            "leaderboard": board.select("model", "score").head(5).to_dicts() if "score" in board.columns else []}


def _metric_value(metrics: dict, name: str):
    for key in (name, f"{name}_macro", f"{name}_ovr", f"{name}_weighted"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v == v:
            return key, float(v)
    return None, None


@register_tool("evaluate", "Held-out metrics with confidence intervals; stop if a metric misses the goal", goal_fields=("min_metrics", "max_metrics"))
def _evaluate(ctx: RunContext) -> dict:
    from ..evaluation import evaluate_model
    g = ctx.goal
    rep = evaluate_model(ctx.get("model"), ctx.get("test"))
    metrics = {k: v for k, v in rep.metrics.items() if isinstance(v, (int, float)) and v == v}
    ctx.put("metrics", metrics)
    failures, checked = [], {}
    for name, floor in (g.min_metrics or {}).items():
        key, v = _metric_value(metrics, name)
        checked[name] = v
        if v is None:
            failures.append(f"{name}: not computed for this task")
        elif v < floor:
            failures.append(f"{key} = {v:.4f} is below the goal {floor}")
    for name, ceiling in (g.max_metrics or {}).items():
        key, v = _metric_value(metrics, name)
        checked[name] = v
        if v is None:
            failures.append(f"{name}: not computed for this task")
        elif v > ceiling:
            failures.append(f"{key} = {v:.4f} is above the goal {ceiling}")
    out = {"metrics": {k: round(v, 4) for k, v in metrics.items()}, "checked": checked,
           "confidence_intervals": {k: [round(a, 4), round(b, 4)] for k, (a, b) in (rep.confidence_intervals or {}).items()}}
    if failures:
        raise GateFailed("The model misses the goal: " + "; ".join(failures), out)
    return out


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


@register_tool("performance_profile", "Measure single-row latency (p50/p95), batch throughput and model size; "
               "stop if a limit in the goal is exceeded", goal_fields=("target", "max_latency_ms", "max_model_mb"))
def _performance(ctx: RunContext, repeats: int = 30) -> dict:
    g, model, test = ctx.goal, ctx.get("model"), ctx.get("test")
    X = test.drop(g.target)
    out = _measure(model, X, repeats)
    batch = X.head(min(X.height, 5000))
    t = time.perf_counter()
    model.predict(batch)
    out["throughput_rows_per_s"] = round(batch.height / max(time.perf_counter() - t, 1e-9))
    problems = _fits(g, out)
    if problems:
        raise GateFailed("; ".join(problems), out)
    return out


@register_tool("explain", "SHAP global importance of the features", goal_fields=("target",))
def _explain(ctx: RunContext, rows: int = 150) -> dict:
    g, model, test = ctx.goal, ctx.get("model"), ctx.get("test")
    imp = model.explain(test.head(rows), rows=rows).global_importance().head(10)
    return {"top_features": imp.select(imp.columns[:3]).to_dicts()}


@register_tool("fairness", "Bias checks across the goal's sensitive columns; stop when a limit is broken", goal_fields=("sensitive", "fairness_limits"))
def _fairness(ctx: RunContext) -> dict:
    g = ctx.goal
    kw = {"limits": g.fairness_limits} if g.fairness_limits else {}
    rep = ctx.get("model").fairness(ctx.get("test"), sensitive=list(g.sensitive), **kw)
    flags = [str(f) for f in rep.flags]
    out = {"passed": bool(rep.passed), "flags": flags}
    if not rep.passed:
        raise GateFailed("Fairness limits broken: " + "; ".join(flags[:5]), out)
    return out


@register_tool("robustness", "Noise, adversarial and extreme-input tests; stop if extreme inputs break the model", goal_fields=("target",))
def _robustness(ctx: RunContext, rows: int = 150) -> dict:
    from ..robustness import robustness_report
    test = ctx.get("test")
    rep = robustness_report(ctx.get("model"), test, n=min(rows, test.height))
    out = {"noise_flip_5pct": rep.noise.get(0.05), "median_adversarial_distance": rep.attack.get("median_distance"),
           "extreme": rep.extreme}
    if rep.extreme.get("crashed") or rep.extreme.get("non_finite"):
        raise GateFailed(f"Extreme inputs break the model: {rep.extreme}", out)
    return out


@register_tool("privacy", "Membership-inference attack on the model; stop if it can tell training rows apart",
               goal_fields=("target", "max_membership_auc"))
def _privacy(ctx: RunContext) -> dict:
    from ..model_privacy import membership_inference_test
    g = ctx.goal
    rep = membership_inference_test(ctx.get("model"), ctx.get("train"), ctx.get("test"), target=g.target)
    out = {"attack_auc": rep.auc, "advantage": rep.advantage, "risk": rep.risk, "mean_confidence": rep.mean_confidence}
    if g.max_membership_auc is not None and rep.auc > g.max_membership_auc:
        raise GateFailed(f"Membership-inference attack AUC {rep.auc:.3f} exceeds {g.max_membership_auc}: "
                         "the model memorises its training rows (regularise, or train with more data)", out)
    return out


@register_tool("model_card", "Write the HTML model card", goal_fields=("sensitive",))
def _card(ctx: RunContext) -> dict:
    g = ctx.goal
    kw = {"sensitive": list(g.sensitive)} if g.sensitive else {}
    path = ctx.art / "model_card.html"
    ctx.get("model").model_card(ctx.get("test"), **kw).save(path)
    return {"path": str(path)}


@register_tool("artifact_scan", "Scan the saved model files for code that would run on load (supply-chain gate)",
               goal_fields=())
def _artifact_scan(ctx: RunContext) -> dict:
    from ..supply_chain import scan_model
    ctx.get("model")                                   # make sure it is saved
    rep = scan_model(ctx.path("model"))
    out = {"files": len(rep.files), "critical": [str(f) for f in rep.critical],
           "warnings": [str(f) for f in rep.findings if f.severity != "critical"][:20]}
    if not rep.passed:
        raise GateFailed("The model files contain code that would run on load", out)
    return out


@register_tool("register", "Add the model to the registry (stage dev) with its test metrics and run id", goal_fields=("registry", "name", "description"))
def _register(ctx: RunContext) -> dict:
    from ..registry import ModelRegistry
    g = ctx.goal
    metrics = {f"test_{k}": v for k, v in ctx.get("metrics").items()}
    mv = ModelRegistry(g.registry).register(ctx.get("model"), g.name, stage="dev", metrics=metrics,
                                            tags={"run_id": ctx.run_id, "goal": g.description[:200]},
                                            description=g.description or None, dataset=ctx.get("train"))
    ctx.put("registered", {"name": g.name, "version": mv.version})
    return {"name": g.name, "version": mv.version, "stage": mv.stage}


@register_tool("promote", "Move the registered version to the goal's stage (needs the goal's approvals)", risk="high", goal_fields=("registry", "name", "deploy", "metric", "approvals"))
def _promote(ctx: RunContext, stage: str = "staging") -> dict:
    from ..registry import ModelRegistry
    g, reg = ctx.goal, ctx.get("registered")
    must_beat = None
    if stage == "production" and g.metric:
        must_beat = ("-" if g.metric in ("rmse", "mae", "mape", "log_loss") else "") + f"test_{g.metric}"
    mv = ModelRegistry(g.registry).promote(reg["name"], reg["version"], stage, reason=f"autonomous run {ctx.run_id}",
                                           must_beat_production=must_beat, approved_by=getattr(ctx, "approvers", []))
    return {"name": reg["name"], "version": mv.version, "stage": mv.stage}


@register_tool("monitor_setup", "Create the drift / performance monitor baseline from the test data", goal_fields=("name",))
def _monitor(ctx: RunContext) -> dict:
    from ..monitoring import Monitor
    g = ctx.goal
    path = ctx.art / "monitor.json"
    Monitor(ctx.get("model"), ctx.get("test"), name=g.name).save(path)
    return {"path": str(path)}
