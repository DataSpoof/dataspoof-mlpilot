"""Fault injection: killed runs, tampered artifacts, concurrent promotions, failing models in the API."""

import io
import json
import platform
import subprocess
import sys
import threading

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot import orchestration as orc
from mlpilot.orchestration.engine import _pid_alive


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    return pl.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2]}).with_columns(
        y=pl.Series(np.where(x[:, 0] + rng.normal(0, 0.7, n) > 0, 1, 0)))


def _goal(tmp_path, **kw):
    base = dict(name="m", data=None, target="y", include=["logistic_regression"], time_budget=5, deploy="none",
                robustness=False, explain=False, min_quality=None, registry=str(tmp_path / "reg"))
    base.update(kw)
    return orc.Goal.from_dict(base)


def test_a_killed_run_is_taken_over_and_resumed(tmp_path):
    goal = _goal(tmp_path)
    run_dir = tmp_path / "run"
    calls = {"n": 0}

    @orc.register_tool("crash_once", "Dies like a killed process the first time")
    def crash_once(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt                       # not an Exception: the engine can't record it
        return {"ok": True}

    wf = orc.plan(goal)
    wf.steps.append(orc.StepSpec("crash", "crash_once", needs=["split"]))
    with pytest.raises(KeyboardInterrupt):
        orc.run(wf, run_dir, data=_data())
    # simulate a hard kill: the lock is left behind by a process that no longer exists
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert not _pid_alive(dead.pid)
    (run_dir / "run.lock").write_text(json.dumps({"pid": dead.pid, "host": platform.node(), "since": "earlier"}))
    res = orc.run(wf, run_dir, data=_data())
    assert res.status == "succeeded", res.summary()
    assert res.step("split").get("reused") and res.step("crash")["summary"] == {"ok": True}
    assert not (run_dir / "run.lock").exists()


def test_a_live_lock_blocks_a_second_runner(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    import os
    (run_dir / "run.lock").write_text(json.dumps({"pid": os.getpid(), "host": platform.node(), "since": "now"}))
    with pytest.raises(RuntimeError, match="another process"):
        orc.run(_goal(tmp_path), run_dir, data=_data())


def test_tampered_model_is_refused_on_reuse(tmp_path):
    goal = _goal(tmp_path)
    run_dir = tmp_path / "run"
    assert orc.run(goal, run_dir, data=_data()).status == "succeeded"
    model_file = run_dir / "artifacts" / "model" / "model.joblib"
    model_file.write_bytes(model_file.read_bytes() + b"tampered")
    changed = orc.Goal.from_dict({**goal.to_dict(), "min_metrics": {"roc_auc": 0.6}})   # evaluate must re-run
    res = orc.run(changed, run_dir, data=_data())
    assert res.status == "failed" and res.failed_step == "evaluate"
    assert "checksum" in res.step("evaluate")["error"].lower()


def test_concurrent_promotions_leave_exactly_one_production_version(tmp_path):
    df = _data()
    reg = ac.ModelRegistry(tmp_path / "registry")
    m = ac.AutoML("y", time_budget=4, include=["logistic_regression"], verbose=False).fit(df)
    versions = [reg.register(m, "churn").version for _ in range(4)]
    errors = []

    def promote(v):
        try:
            reg.promote("churn", v, "production")
        except Exception as exc:                          # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=promote, args=(v,)) for v in versions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    stages = reg.versions("churn")["stage"].to_list()
    assert stages.count("production") == 1 and stages.count("archived") == 3


def test_api_survives_a_model_that_throws(tmp_path):
    df = _data()
    model = ac.AutoML("y", time_budget=4, include=["logistic_regression"], verbose=False).fit(df)
    app = ac.make_app(model)

    def call(body):
        data = json.dumps(body).encode()
        env = {"REQUEST_METHOD": "POST", "PATH_INFO": "/predict", "CONTENT_LENGTH": str(len(data)),
               "CONTENT_TYPE": "application/json", "wsgi.input": io.BytesIO(data)}
        out = {}
        raw = b"".join(app(env, lambda s, h: out.update(status=int(s.split()[0]))))
        return out["status"], json.loads(raw)
    ok = {"records": [{"a": 0.1, "b": 0.2, "c": 0.3}]}
    assert call(ok)[0] == 200
    original = model.model_.predict_proba
    model.model_.predict_proba = lambda X: (_ for _ in ()).throw(MemoryError("injected"))
    try:
        status, body = call(ok)
        assert status == 500 and body["error"] == "Internal error." and "injected" not in json.dumps(body)
    finally:
        model.model_.predict_proba = original
    assert call(ok)[0] == 200                             # recovers on the next request
