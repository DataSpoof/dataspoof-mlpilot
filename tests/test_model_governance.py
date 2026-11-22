"""Model risk classification and retirement in the registry."""

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.registry import ModelRegistry, ModelRetiredError, PromotionError
from mlpilot.registry._store import user


@pytest.fixture()
def reg(tmp_path):
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150)}).with_columns(
        y=((pl.col("a") + rng.normal(0, 1, 150)) > 0).cast(pl.Int8))
    r = ModelRegistry(tmp_path / "registry")
    m = ac.AutoML("y", time_budget=4, include=["logistic_regression"], verbose=False).fit(df)
    r.register(m, "credit")
    r.register(m, "credit")
    return r


def test_high_risk_needs_two_other_approvers(reg):
    gov = reg.set_risk("credit", "high", reason="credit decisions about people", owner="risk-team",
                       review_by="2999-01-01")
    assert gov["risk"] == "high" and reg.governance("credit")["owner"] == "risk-team"
    reg.promote("credit", 1, "staging")                                  # only production is gated
    with pytest.raises(PromotionError, match="two approvers"):
        reg.promote("credit", 1, "production", approved_by=["priya"])
    with pytest.raises(PromotionError, match="two approvers"):
        reg.promote("credit", 1, "production", approved_by=["priya", user()])   # self-approval doesn't count
    assert reg.promote("credit", 1, "production", approved_by=["priya", "omar"]).stage == "production"
    assert any("risk set to high" in (r or "") for r in reg.history("credit")["reason"].to_list())


def test_overdue_review_and_unacceptable_risk(reg):
    reg.set_risk("credit", "high", reason="r", review_by="2000-01-01")
    with pytest.raises(PromotionError, match="review was due"):
        reg.promote("credit", 1, "production", approved_by=["a", "b"])
    reg.set_risk("credit", "unacceptable", reason="social scoring")
    with pytest.raises(PromotionError, match="unacceptable"):
        reg.promote("credit", 2, "staging")
    with pytest.raises(ValueError):
        reg.set_risk("credit", "spicy", reason="?")
    with pytest.raises(ValueError):
        reg.set_risk("credit", "high", reason="r", review_by="next tuesday")


def test_retirement(reg):
    reg.promote("credit", 2, "production")
    assert reg.load("credit") is not None
    reg.retire("credit", reason="replaced by credit-v2 scorecard", replacement="credit-v2")
    assert {v["stage"] for v in reg.versions("credit").to_dicts()} == {"archived"}
    with pytest.raises(ModelRetiredError, match="credit-v2"):
        reg.load("credit")
    assert reg.load("credit", version=2, allow_retired=True) is not None
    with pytest.raises(PromotionError, match="retired"):
        reg.promote("credit", 2, "production")
