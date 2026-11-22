"""Active learning selection, the labeling queue and label quality."""

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.labeling import LabelQueue, select_for_labeling, simulate_active_learning


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = np.where(x[:, 0] + 0.5 * x[:, 1] > 0, "yes", "no")
    return pl.DataFrame({"id": [f"r{i}" for i in range(n)], "a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": y})


@pytest.fixture(scope="module")
def model():
    return ac.AutoML("y", time_budget=4, include=["logistic_regression"], verbose=False).fit(_data().drop("id").head(200))


def test_uncertain_rows_come_first(model):
    pool = _data(seed=1).drop("y")
    for strategy in ("least_confident", "margin", "entropy", "diverse"):
        picks = select_for_labeling(model, pool, n=20, strategy=strategy, id_col="id")
        assert picks.height == 20 and {"uncertainty", "predicted", "id"} <= set(picks.columns)
        # the chosen rows sit near the decision boundary a + 0.5 b = 0
        assert picks.select((pl.col("a") + 0.5 * pl.col("b")).abs().median()).item() < \
            pool.select((pl.col("a") + 0.5 * pl.col("b")).abs().median()).item()
    margin = select_for_labeling(model, pool, n=20, strategy="margin", id_col="id")
    assert margin["uncertainty"].is_sorted(descending=True)
    assert select_for_labeling(model, pool.head(5), n=20, id_col="id").height == 5
    with pytest.raises(ValueError):
        select_for_labeling(model, pool, strategy="vibes")


def test_label_queue_votes_and_agreement(tmp_path, model):
    q = LabelQueue(tmp_path / "labels")
    picks = select_for_labeling(model, _data(seed=2).drop("y"), n=4, id_col="id")
    assert q.add(picks, id_col="id") == 4 and q.add(picks, id_col="id") == 0       # no duplicates
    ids = q.items()["item_id"].to_list()
    q.record(ids[0], "yes", by="asha")
    q.record(ids[0], "yes", by="ben")
    q.record(ids[1], "yes", by="asha")
    q.record(ids[1], "no", by="ben")                     # a tie
    q.record(ids[2], "no", by="asha")
    q.record(ids[2], "yes", by="asha")                   # asha changed her mind: latest wins
    assert q.pending()["item_id"].to_list() == [ids[3]]
    lab = {r["item_id"]: r for r in q.labels().iter_rows(named=True)}
    assert lab[ids[0]]["label"] == "yes" and lab[ids[0]]["agreement"] == 1.0
    assert lab[ids[1]]["label"] is None and lab[ids[1]]["votes"] == 2
    assert lab[ids[2]]["label"] == "yes" and lab[ids[2]]["votes"] == 1
    assert q.agreement() == {"items": 2, "mean_agreement": 0.75, "unanimous": 0.5}
    rows = q.training_rows(label_name="y")
    assert set(rows["item_id"]) == {ids[0], ids[2]} and "uncertainty" not in rows.columns
    with pytest.raises(KeyError):
        q.record("nope", "yes", by="asha")
    n = q.import_labels(pl.DataFrame({"item_id": [ids[3]], "label": ["no"], "by": ["tool"]}))
    assert n == 1 and q.pending().height == 0


def test_active_learning_beats_random_labeling():
    curve = simulate_active_learning(_data(n=1500, seed=3), "y", initial=20, batch=10, rounds=5)
    final = {r["strategy"]: r["accuracy"] for r in curve.filter(pl.col("round") == 5).iter_rows(named=True)}
    assert curve["labelled"].max() == 70
    assert final["margin"] >= final["random"]
