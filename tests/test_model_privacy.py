"""Membership inference and model extraction tests, and the engine's privacy gate."""

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot import orchestration as orc
from mlpilot.model_privacy import extraction_test, membership_inference_test


def _data(n, seed=0, noise=1.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    y = np.where(x[:, 0] + rng.normal(0, noise, n) > 0, "yes", "no")       # noisy labels: easy to memorise
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(6)}).with_columns(pl.Series("y", y))


def test_overfit_model_leaks_membership_regularised_one_does_not():
    train, holdout = _data(300, 1), _data(300, 2)
    leaky = ac.AutoML("y", time_budget=6, include=["random_forest"], ensemble=False, verbose=False).fit(train)
    safe = ac.AutoML("y", time_budget=6, include=["logistic_regression"], ensemble=False, verbose=False).fit(train)
    a, b = membership_inference_test(leaky, train, holdout), membership_inference_test(safe, train, holdout)
    assert a.auc > b.auc and a.risk in ("medium", "high") and b.risk == "low"
    assert a.mean_confidence["train"] > a.mean_confidence["holdout"]
    assert "attack AUC" in a.summary()
    reg = ac.AutoML("x0", time_budget=4, include=["ridge"], verbose=False).fit(train.drop("y"))
    with pytest.raises(ValueError, match="classifier"):
        membership_inference_test(reg, train.drop("y"), holdout.drop("y"))


def test_extraction_gets_easier_with_more_queries():
    train, holdout = _data(400, 3, noise=0.3), _data(400, 4, noise=0.3)
    model = ac.AutoML("y", time_budget=4, include=["logistic_regression"], verbose=False).fit(train)
    rep = extraction_test(model, holdout, queries=(20, 2000))
    few, many = rep.table["agreement"].to_list()
    assert many > few and many > 0.85
    assert "queries" in rep.summary()


def test_privacy_gate_in_the_engine(tmp_path):
    goal = orc.Goal(name="m", data=None, target="y", include=["random_forest"], time_budget=6, deploy="none",
                    robustness=False, explain=False, min_quality=None, monitor=False, max_membership_auc=0.55,
                    registry=str(tmp_path / "reg"))
    assert "privacy" in [s.id for s in orc.plan(goal).steps]
    res = orc.run(goal, tmp_path / "run", data=_data(500, 5))
    assert res.step("privacy")["status"] == "failed" and "memorises" in res.step("privacy")["error"]
    with pytest.raises(orc.GoalError, match="max_membership_auc"):
        orc.Goal.from_dict({**goal.to_dict(), "max_membership_auc": 0.3})
