"""Incident diagnosis, playbooks, approvals and verified recovery."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.monitoring import IncidentLog, diagnose
from mlpilot.quality import DataMonitor


def _report(*alerts):
    return {"alerts": [{"severity": s, "kind": k, "message": m} for s, k, m in alerts]}


def test_upstream_data_comes_first_and_pauses_retraining(tmp_path):
    now = dt.datetime(2026, 11, 21, tzinfo=dt.timezone.utc)
    stale = pl.DataFrame({"ts": [now - dt.timedelta(hours=40)] * 50, "x": range(50)})
    obs = DataMonitor("orders", history=tmp_path / "h.jsonl", timestamp="ts",
                      sla={"max_staleness_hours": 6}).observe(stale, at=now)
    diag = diagnose(monitor=_report(("critical", "performance", "accuracy fell 12%")), data=obs)
    assert diag.cause == "upstream_data" and diag.confidence == "high"
    assert [a.name for a in diag.actions] == ["pause_retraining", "notify"]       # never retrain on broken data
    log = IncidentLog(tmp_path / "incidents.jsonl")
    inc = log.open(diag, model="churn")
    sent = []
    log.remediate(inc, handlers={"notify": lambda i: sent.append(i.id) or "paged data owner"})
    assert {a.name: a.status for a in inc.actions} == {"pause_retraining": "done", "notify": "done"}
    assert (tmp_path / "retraining_paused_churn.flag").read_text() == inc.id and sent == [inc.id]


def test_infrastructure_when_only_slos_break(tmp_path):
    diag = diagnose(service={"latency_ms_p95": 240, "error_rate": 0.001, "availability": 0.98},
                    slo={"latency_ms_p95": 100, "error_rate": 0.01, "availability": 0.999})
    assert diag.cause == "infrastructure" and len(diag.evidence) == 2
    log = IncidentLog(tmp_path / "i.jsonl")
    inc = log.remediate(log.open(diag), handlers={"scale_out": lambda i: "replicas 2 -> 4"})
    status = {a.name: a.status for a in inc.actions}
    assert status["scale_out"] == "done" and status["restart_workers"] == "no_handler"


def test_cause_ordering_for_model_problems():
    perf = ("critical", "performance", "roc_auc fell from 0.91 to 0.80")
    assert diagnose(_report(perf), released_hours_ago=6).cause == "bad_release"
    assert diagnose(_report(perf), released_hours_ago=500).cause == "concept_drift"
    shifted = diagnose(_report(perf, ("critical", "data_drift", "60% of columns drifted")))
    assert shifted.cause == "covariate_shift" and shifted.confidence == "high"
    unlabelled = diagnose(_report(("warning", "prediction_drift", "PSI 0.4")))
    assert unlabelled.cause == "covariate_shift" and unlabelled.confidence == "medium"
    assert diagnose(_report()).healthy


def _registered(tmp_path):
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)}).with_columns(
        y=(pl.col("a") > 0).cast(pl.Int8))
    reg = ac.ModelRegistry(tmp_path / "registry")
    for seed in (0, 1):
        m = ac.AutoML("y", time_budget=4, include=["logistic_regression"], verbose=False, seed=seed).fit(df)
        v = reg.register(m, "churn")
        reg.promote("churn", v.version, "production")
    return reg


def test_rollback_needs_approval_then_verified_recovery(tmp_path):
    reg = _registered(tmp_path)
    assert reg.get("churn", stage="production").version == 2
    log = IncidentLog(tmp_path / "i.jsonl", registry=reg)
    diag = diagnose(_report(("critical", "performance", "accuracy fell 15%")), released_hours_ago=3)
    inc = log.remediate(log.open(diag, model="churn"))
    assert inc.status == "awaiting_approval" and reg.get("churn", stage="production").version == 2
    inc = log.approve(inc, "rollback", by="priya")
    rollback = next(a for a in inc.actions if a.name == "rollback")
    assert rollback.status == "done" and rollback.approved_by == ["priya"]
    assert reg.get("churn", stage="production").version == 1
    inc = log.verify(inc, _report(("warning", "prediction_drift", "still odd")))
    assert inc.status == "open"
    inc = log.verify(inc, _report())
    assert inc.status == "resolved" and inc.resolved
    assert [i.id for i in log.incidents("resolved")] == [inc.id]
    assert log.get(inc.id).status == "resolved"
    with pytest.raises(ValueError, match="healthy"):
        log.open(diagnose(_report()))
