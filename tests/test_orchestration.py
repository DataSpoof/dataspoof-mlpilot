"""The autonomous execution engine: goals, plans, gates, approvals, resume and the run record."""

import json

import numpy as np
import polars as pl
import pytest

from mlpilot import orchestration as orc
from mlpilot.orchestration import Goal, GoalError, StepSpec, Workflow


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = (x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2] + rng.normal(scale=0.6, size=n) > 0).astype(int)
    return pl.DataFrame({"tenure": x[:, 0] * 10 + 30, "spend": x[:, 1] * 50 + 200, "visits": x[:, 2],
                         "noise": x[:, 3], "plan": rng.choice(["basic", "pro", "team"], n).tolist(), "churned": y})


def _goal(tmp_path, **kw):
    base = dict(name="churn", data=None, target="churned", include=["logistic_regression"], time_budget=8,
                registry=str(tmp_path / "registry"), robustness=False, explain=False, min_quality=None)
    base.update(kw)
    return Goal.from_dict(base)


# ── goals ───────────────────────────────────────────────────────────────────
def test_goal_validation_lists_every_problem():
    with pytest.raises(GoalError) as e:
        Goal.from_dict({"name": "bad name!", "data": None, "target": "", "deploy": "moon", "min_metrics": {"rmse": 3}})
    msg = str(e.value)
    for part in ("name must be", "target is required", "deploy must be", "rmse is lower-is-better"):
        assert part in msg
    with pytest.raises(GoalError, match="Unknown goal field"):
        Goal.from_dict({"name": "x", "data": None, "target": "y", "colour": "blue"})


def test_goal_round_trips_through_yaml(tmp_path):
    g = Goal(name="churn", data="d.csv", target="churned", min_metrics={"roc_auc": 0.8}, sensitive=["gender"])
    path = tmp_path / "goal.yml"
    g.to_yaml(path)
    assert Goal.from_file(path) == g


def test_goal_from_text_rules():
    g = orc.goal_from_text("Predict churned from data/customers.csv with AUC above 0.85 and latency below 50 ms; "
                           "check fairness across gender and region; budget $20; deploy to production within 30 minutes",
                           use_llm=False)
    assert (g.target, g.data, g.deploy) == ("churned", "data/customers.csv", "production")
    assert g.min_metrics == {"roc_auc": 0.85} and g.metric == "roc_auc"
    assert g.max_latency_ms == 50 and g.budget_usd == 20 and g.time_budget == 1800
    assert g.sensitive == ["gender", "region"] and g.name == "churned"
    r = orc.goal_from_text("forecast sales with RMSE below 4.5, don't deploy", use_llm=False, data="s.parquet")
    assert r.max_metrics == {"rmse": 4.5} and r.deploy == "none" and r.data == "s.parquet"
    with pytest.raises(GoalError, match="target"):
        orc.goal_from_text("make me a good model", use_llm=False)


# ── plans ───────────────────────────────────────────────────────────────────
def test_plan_follows_the_goal(tmp_path):
    wf = orc.plan(_goal(tmp_path, deploy="production", sensitive=["plan"], robustness=True))
    ids = [s.id for s in wf.order()]
    assert ids[:5] == ["ingest", "quality_gate", "split", "poisoning_check", "automl"]
    assert {"fairness", "robustness", "register", "promote", "monitor_setup"} <= set(ids)
    promote = next(s for s in wf.steps if s.id == "promote")
    assert promote.approval_stage == "production" and promote.params == {"stage": "production"}
    assert "needs 1 approval" in wf.describe()
    assert "register" not in [s.id for s in orc.plan(_goal(tmp_path, deploy="none")).steps]


def test_workflows_only_use_registered_tools_safely(tmp_path):
    g = _goal(tmp_path)
    with pytest.raises(ValueError, match="unknown tool"):
        Workflow(g, [StepSpec("x", "rm_rf")]).validate()
    with pytest.raises(ValueError, match="cycle"):
        Workflow(g, [StepSpec("a", "ingest", needs=["b"]), StepSpec("b", "split", needs=["a"])]).validate()
    with pytest.raises(ValueError, match="high-risk"):
        Workflow(g, [StepSpec("p", "promote")]).validate()


# ── runs ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def prod_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("orc")
    goal = _goal(tmp, deploy="production", min_metrics={"roc_auc": 0.7}, max_latency_ms=5000, robustness=True,
                 explain=True)
    res = orc.run(goal, tmp / "run", data=_data(), requested_by="alice")
    return tmp, goal, res


def test_run_stops_for_production_approval(prod_run):
    tmp, goal, res = prod_run
    assert res.status == "waiting_approval", res.summary()
    for step in ("ingest", "quality_gate", "split", "poisoning_check", "automl", "evaluate", "performance_profile",
                 "explain", "robustness", "model_card", "register", "monitor_setup"):
        assert res.step(step)["status"] == "done", (step, res.step(step))
    assert res.step("promote")["status"] == "waiting_approval"
    assert res.step("evaluate")["summary"]["checked"]["roc_auc"] > 0.7
    assert res.step("performance_profile")["summary"]["latency_ms_p95"] > 0
    run = tmp / "run"
    for f in ("workflow.json", "state.json", "events.jsonl", "manifest.json", "report.md",
              "artifacts/model_card.html", "artifacts/monitor.json", "artifacts/train.parquet"):
        assert (run / f).exists(), f
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["data_fingerprint"] and manifest["seed"] == 0 and "polars" in manifest["packages"]
    assert res.cost["cpu_seconds"] > 0 and res.cost["usd"] > 0
    assert "WAITING_APPROVAL" in res.summary() and "mlpilot approve" in res.step("promote")["note"]


def test_four_eyes_approval_then_resume(prod_run):
    tmp, goal, _ = prod_run
    run = tmp / "run"
    with pytest.raises(PermissionError, match="four-eyes"):
        orc.approve(run, "promote", by="alice")
    with pytest.raises(ValueError, match="not waiting"):
        orc.approve(run, "automl", by="bob")
    orc.approve(run, "promote", by="bob", note="reviewed the model card")
    res = orc.resume(run)
    assert res.status == "succeeded", res.summary()
    assert res.step("automl")["reused"] and res.step("promote")["summary"]["stage"] == "production"
    from mlpilot.registry import ModelRegistry
    assert ModelRegistry(goal.registry).get("churn", stage="production").version == 1
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert any(e["message"] == "approved" and e["by"] == "bob" for e in events)
    assert orc.status(run).status == "succeeded"


def test_resume_reruns_only_what_changed(tmp_path):
    df = _data()
    goal = _goal(tmp_path, deploy="none", min_metrics={"roc_auc": 0.99})          # impossible floor
    res = orc.run(goal, tmp_path / "r", data=df)
    assert res.status == "failed" and res.failed_step == "evaluate"
    assert "below the goal" in res.step("evaluate")["error"]
    assert res.step("monitor_setup")["status"] in ("skipped", "blocked")
    fixed = Goal.from_dict({**goal.to_dict(), "min_metrics": {"roc_auc": 0.6}})
    res2 = orc.run(fixed, tmp_path / "r", data=df)
    assert res2.status == "succeeded", res2.summary()
    assert res2.step("automl").get("reused") and not res2.step("evaluate").get("reused")
    res3 = orc.run(fixed, tmp_path / "r", data=df.with_columns(pl.col("noise") * 2))   # new data: retrain
    assert not res3.step("split").get("reused") and not res3.step("automl").get("reused")


def test_quality_gate_and_budget(tmp_path):
    df = _data()
    bad = _goal(tmp_path, min_quality=99.9)
    res = orc.run(bad, tmp_path / "q", data=df.with_columns(pl.when(pl.col("visits") > 0).then(None)
                                                           .otherwise(pl.col("visits")).alias("visits")))
    assert res.status == "failed" and res.failed_step == "quality_gate"
    res = orc.run(_goal(tmp_path, deploy="none", budget_usd=1e-9), tmp_path / "b", data=df)
    assert res.status == "budget_exceeded" and res.step("automl")["status"] == "skipped"


def test_missing_target_is_a_gate(tmp_path):
    res = orc.run(_goal(tmp_path, target="nope"), tmp_path / "t", data=_data())
    assert res.status == "failed" and "not in the data" in res.step("ingest")["error"]


def test_custom_tools_and_optional_steps(tmp_path):
    calls = []

    @orc.register_tool("count_rows", "Count training rows")
    def count_rows(ctx, label="rows"):
        calls.append(label)
        return {label: ctx.get("train").height}

    @orc.register_tool("flaky", "Always fails")
    def flaky(ctx):
        raise RuntimeError("downstream service unavailable")

    goal = _goal(tmp_path, deploy="none", monitor=False)
    wf = orc.plan(goal)
    wf.steps += [StepSpec("count", "count_rows", params={"label": "n"}, needs=["split"]),
                 StepSpec("notify", "flaky", needs=["count"], optional=True, retries=1)]
    res = orc.run(wf, tmp_path / "c", data=_data())
    assert res.status == "succeeded", res.summary()
    assert res.step("count")["summary"]["n"] > 0 and calls == ["n"]
    assert res.step("notify")["attempts"] == 2 and "optional step failed" in res.step("notify")["note"]


def test_cli_plan_run_status_approve(tmp_path, capsys):
    from mlpilot.deploy.cli import main
    data = tmp_path / "customers.parquet"
    _data().write_parquet(data)
    goal = _goal(tmp_path, data=str(data), deploy="production")
    goal.to_yaml(tmp_path / "goal.yml")
    assert main(["plan", "--goal", str(tmp_path / "goal.yml")]) == 0
    assert "promote" in capsys.readouterr().out
    out = str(tmp_path / "cli_run")
    assert main(["run", "--goal", str(tmp_path / "goal.yml"), "--out", out, "--by", "alice"]) == 3   # waiting
    assert main(["approve", out, "--step", "promote", "--by", "bob"]) == 0
    assert main(["run", "--out", out, "--resume"]) == 0
    assert main(["status", out]) == 0
    assert "SUCCEEDED" in capsys.readouterr().out


def test_cost_aware_automl_finds_a_model_within_limits(tmp_path):
    goal = _goal(tmp_path, deploy="none", monitor=False, include=["random_forest", "logistic_regression"],
                 max_model_mb=0.5, time_budget=10)
    res = orc.run(goal, tmp_path / "cost", data=_data())
    assert res.status == "succeeded", res.summary()
    attempts = res.step("automl")["summary"]["attempts"]
    assert attempts[-1]["meets_limits"] and attempts[-1]["model_mb"] <= 0.5
    assert all(not a["meets_limits"] for a in attempts[:-1])                  # it stopped at the first fit
    assert res.step("performance_profile")["summary"]["model_mb"] <= 0.5
