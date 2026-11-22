"""Data observability (freshness, volume, schema, nulls, distribution) and reconciliation."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from mlpilot.quality import DataMonitor, reconcile

NOW = dt.datetime(2026, 11, 21, 12, 0, tzinfo=dt.timezone.utc)


def _batch(n=2000, seed=0, hours_old=1.0, shift=0.0):
    rng = np.random.default_rng(seed)
    latest = NOW - dt.timedelta(hours=hours_old)
    return pl.DataFrame({
        "order_id": np.arange(n) + seed * 100_000,
        "amount": rng.gamma(2.0, 30.0, n) + shift,
        "country": rng.choice(["IN", "US", "DE"], n).tolist(),
        "created_at": [latest - dt.timedelta(minutes=int(m)) for m in rng.integers(0, 600, n)],
    })


def test_clean_batches_pass_and_build_history(tmp_path):
    mon = DataMonitor("orders", history=tmp_path / "h.jsonl", timestamp="created_at",
                      sla={"max_staleness_hours": 6, "min_rows": 1000})
    for seed in range(3):
        rep = mon.observe(_batch(seed=seed), at=NOW)
        assert rep.passed and rep.issues == [], rep.summary()
    hist = mon.history()
    assert hist.height == 3 and hist["passed"].all() and hist["staleness_hours"][0] == pytest.approx(1.0)


def test_every_signal_is_detected(tmp_path):
    mon = DataMonitor("orders", history=tmp_path / "h.jsonl", timestamp="created_at",
                      sla={"max_staleness_hours": 6, "min_rows": 1000, "max_null_rate": {"country": 0.05}})
    for seed in range(3):
        mon.observe(_batch(seed=seed), at=NOW)
    bad = _batch(n=800, seed=9, hours_old=30, shift=200.0).drop("country").with_columns(
        pl.col("order_id").cast(pl.Utf8), coupon=pl.lit("X"))
    rep = mon.observe(bad, at=NOW)
    checks = {(i.check, i.column) for i in rep.issues}
    assert not rep.passed
    assert ("freshness", None) in checks
    assert ("volume", None) in checks                                   # 800 rows: below SLA and -60%
    assert ("schema", "country") in checks and ("schema", "coupon") in checks and ("schema", "order_id") in checks
    assert ("distribution", "amount") in checks
    severities = {i.check: i.severity for i in rep.issues if i.column == "coupon"}
    assert severities == {"schema": "warning"}                          # additive schema: a new column only warns


def test_null_jump_and_strict_schema(tmp_path):
    mon = DataMonitor("orders", history=tmp_path / "h.jsonl", sla={"schema": "strict"})
    mon.observe(_batch(seed=1), at=NOW)
    b = _batch(seed=2).with_columns(pl.when(pl.col("amount") > 40).then(None).otherwise(pl.col("amount")).alias("amount"),
                                    extra=pl.lit(1))
    rep = mon.observe(b, at=NOW, record=False)
    assert any(i.check == "nulls" and i.column == "amount" for i in rep.issues)
    assert any(i.check == "schema" and i.column == "extra" and i.severity == "error" for i in rep.issues)
    assert mon.history().height == 1                                    # record=False did not append
    with pytest.raises(ValueError, match="Unknown SLA"):
        DataMonitor("x", sla={"max_rows": 5})


def test_reconcile_identical_and_broken_copies():
    src = _batch(n=500, seed=3)
    assert reconcile(src, src.sample(fraction=1.0, shuffle=True, seed=1), key="order_id").passed
    dst = (src.filter(pl.col("order_id") != src["order_id"][0])                       # one row lost
           .with_columns(pl.when(pl.col("order_id") == src["order_id"][5]).then(pl.col("amount") + 1)
                         .otherwise(pl.col("amount")).alias("amount"))               # one value changed
           .vstack(src.tail(1)))                                                     # one duplicate
    rec = reconcile(src, dst, key="order_id")
    assert not rec.passed
    assert rec.n_missing == 1 and rec.n_extra == 0 and rec.duplicate_keys["destination"] == 1
    assert rec.changed_rows == 1 and rec.changed_sample == [{"order_id": int(src["order_id"][5])}]
    assert not rec.columns.filter(pl.col("column") == "amount")["match"][0]
    assert "1 missing in destination" in rec.summary()
    typed = reconcile(src, src.with_columns(pl.col("amount").cast(pl.Float32)))
    assert not typed.passed and typed.schema_differences
