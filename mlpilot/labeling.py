"""Active learning and labeling: spend the labeling budget where the model learns most.

>>> picks = select_for_labeling(model, unlabeled, n=50, id_col="ticket_id")      # most uncertain first
>>> queue = LabelQueue("labels/")
>>> queue.add(picks, id_col="ticket_id")
>>> queue.record("T-102", "refund", by="asha")         # or queue.import_labels(csv_from_your_tool)
>>> queue.record("T-102", "refund", by="ben")
>>> queue.labels()                                      # majority label, votes, agreement per item
>>> queue.agreement()                                   # how consistent are the labelers?
>>> simulate_active_learning(df, "label")              # does it beat random labeling on your data?

Strategies for :func:`select_for_labeling`:

* ``least_confident``: the lowest top-class probability.
* ``margin``: the smallest gap between the two most likely classes.
* ``entropy``: the most spread-out probabilities.
* ``diverse`` (default): margin, but spread across clusters of the candidates, so a batch isn't
  50 near-copies of one confusing case.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import polars as pl

STRATEGIES = ("least_confident", "margin", "entropy", "diverse", "random")


def _uncertainty(proba: np.ndarray, strategy: str) -> np.ndarray:
    """Higher = more worth labeling."""
    p = np.clip(proba, 1e-12, 1.0)
    if strategy == "least_confident":
        return 1 - p.max(axis=1)
    if strategy == "entropy":
        return -(p * np.log(p)).sum(axis=1)
    top2 = np.sort(p, axis=1)[:, -2:] if p.shape[1] > 1 else np.hstack([np.zeros((len(p), 1)), p])
    return 1 - (top2[:, 1] - top2[:, 0])                # margin


def select_for_labeling(model, pool: pl.DataFrame, n: int = 50, *, strategy: str = "diverse",
                        id_col: str | None = None, seed: int = 0) -> pl.DataFrame:
    """The ``n`` rows of ``pool`` most worth labeling for a classifier ``model`` (anything with
    ``predict_proba`` returning one column per class), with ``uncertainty`` and ``predicted``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if pool.height == 0:
        return pool.with_columns(uncertainty=pl.lit(None, pl.Float64), predicted=pl.lit(None, pl.Utf8))
    n = min(n, pool.height)
    if hasattr(model, "preprocess_"):                    # an MLPilot model: exactly the columns it was trained on
        from .deploy.scoring import model_inputs
        X = model_inputs(model, pool)
    else:
        X = pool.drop(id_col) if id_col and id_col in pool.columns else pool
    proba_df = model.predict_proba(X)
    proba = proba_df.to_numpy() if hasattr(proba_df, "to_numpy") else np.asarray(proba_df)
    classes = [c.removeprefix("proba_") for c in proba_df.columns] if hasattr(proba_df, "columns") else \
        [str(i) for i in range(proba.shape[1])]
    rng = np.random.default_rng(seed)
    score = rng.random(pool.height) if strategy == "random" else \
        _uncertainty(proba, "margin" if strategy == "diverse" else strategy)
    order = np.argsort(-score, kind="stable")
    if strategy == "diverse" and pool.height > n:
        cand = order[: min(pool.height, 5 * n)]
        numeric = [c for c, t in X.schema.items() if t.is_numeric()]
        if numeric:
            from sklearn.cluster import KMeans
            feats = X[cand.tolist()].select(numeric).fill_null(0).to_numpy().astype(float)
            feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
            k = min(n, len(cand))
            labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(feats)
            chosen = []
            for cl in range(k):                         # the most uncertain member of every cluster
                members = cand[labels == cl]
                if members.size:
                    chosen.append(members[np.argmax(score[members])])
            rest = [i for i in cand if i not in set(chosen)]
            order = np.array((sorted(chosen, key=lambda i: -score[i]) + rest)[:n])
    picked = order[:n]
    return pool[picked.tolist()].with_columns(
        uncertainty=pl.Series(np.round(score[picked], 6)),
        predicted=pl.Series([classes[i] for i in proba[picked].argmax(axis=1)]))


# ── the labeling queue ──────────────────────────────────────────────────────
@dataclasses.dataclass
class LabelQueue:
    """A durable labeling queue in a directory: ``items.parquet`` (what to label) and
    ``labels.jsonl`` (every label given, by whom and when; append-only)."""

    path: str | Path

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def _items(self) -> Path:
        return Path(self.path) / "items.parquet"

    @property
    def _labels(self) -> Path:
        return Path(self.path) / "labels.jsonl"

    def items(self) -> pl.DataFrame:
        return pl.read_parquet(self._items) if self._items.exists() else pl.DataFrame({"item_id": []}, schema={"item_id": pl.Utf8})

    def add(self, rows: pl.DataFrame, *, id_col: str | None = None) -> int:
        """Queue ``rows`` (ids from ``id_col``, else generated). Already-queued ids are skipped.
        Returns how many were added."""
        rows = rows.with_columns(pl.col(id_col).cast(pl.Utf8).alias("item_id")) if id_col else \
            rows.with_columns(item_id=pl.Series([f"item-{_dt.datetime.now().strftime('%Y%m%d%H%M%S%f')}-{i}"
                                                 for i in range(rows.height)]))
        current = self.items()
        new = rows.filter(~pl.col("item_id").is_in(current["item_id"].to_list())) if current.height else rows
        new = new.with_columns(queued_at=pl.lit(_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")))
        out = pl.concat([current, new], how="diagonal_relaxed") if current.height else new
        out.write_parquet(self._items)
        return new.height

    def record(self, item_id, label, *, by: str, note: str = "") -> dict:
        ids = set(self.items()["item_id"].to_list())
        if str(item_id) not in ids:
            raise KeyError(f"{item_id!r} is not in the queue")
        entry = {"item_id": str(item_id), "label": label, "by": by, "note": note,
                 "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        with self._labels.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def import_labels(self, df, *, id_col: str = "item_id", label_col: str = "label", by_col: str | None = "by",
                      default_by: str = "import") -> int:
        """Bulk-load labels exported from a labeling tool (a DataFrame or a CSV / Parquet path)."""
        if not isinstance(df, pl.DataFrame):
            from .ingestion import read
            df = read(str(df))
            df = df.collect() if isinstance(df, pl.LazyFrame) else df
        n = 0
        for r in df.iter_rows(named=True):
            self.record(r[id_col], r[label_col], by=str(r.get(by_col) or default_by) if by_col else default_by)
            n += 1
        return n

    def _votes(self) -> pl.DataFrame:
        rows = [json.loads(line) for line in self._labels.read_text(encoding="utf-8").splitlines()] \
            if self._labels.exists() else []
        if not rows:
            return pl.DataFrame(schema={"item_id": pl.Utf8, "label": pl.Utf8, "by": pl.Utf8})
        # a labeler's latest answer for an item replaces their earlier ones
        return (pl.DataFrame(rows).with_columns(pl.col("label").cast(pl.Utf8))
                .unique(["item_id", "by"], keep="last", maintain_order=True))

    def pending(self) -> pl.DataFrame:
        """Queued items with no label yet."""
        done = set(self._votes()["item_id"].to_list())
        items = self.items()
        return items.filter(~pl.col("item_id").is_in(list(done))) if items.height else items

    def labels(self, *, min_votes: int = 1) -> pl.DataFrame:
        """One row per labelled item: the majority ``label``, ``votes``, and ``agreement``
        (share of votes for the majority). Ties are left unresolved (``label`` null)."""
        v = self._votes()
        if v.height == 0:
            return pl.DataFrame(schema={"item_id": pl.Utf8, "label": pl.Utf8, "votes": pl.Int64, "agreement": pl.Float64})
        counts = v.group_by(["item_id", "label"]).agg(pl.len().alias("n"))
        totals = v.group_by("item_id").agg(pl.len().alias("votes"))
        best = (counts.sort(["item_id", "n"], descending=[False, True])
                .group_by("item_id", maintain_order=True)
                .agg(pl.col("label").first().alias("label"), pl.col("n").first().alias("top"),
                     (pl.col("n") == pl.col("n").max()).sum().alias("tied")))
        out = best.join(totals, on="item_id").with_columns(
            agreement=(pl.col("top") / pl.col("votes")).round(4),
            label=pl.when(pl.col("tied") > 1).then(None).otherwise(pl.col("label")))
        return out.filter(pl.col("votes") >= min_votes).select("item_id", "label", "votes", "agreement")

    def agreement(self) -> dict:
        """Label quality across items with at least two votes: mean agreement and the share of
        unanimous items."""
        lab = self.labels(min_votes=2)
        if lab.height == 0:
            return {"items": 0, "mean_agreement": None, "unanimous": None}
        agree = lab["agreement"].to_numpy()
        return {"items": lab.height, "mean_agreement": round(float(agree.mean()), 4),
                "unanimous": round(float((agree == 1.0).mean()), 4)}

    def training_rows(self, *, label_name: str = "label", min_agreement: float = 0.0) -> pl.DataFrame:
        """Queued items joined with their agreed labels, ready to add to the training set."""
        lab = self.labels().filter(pl.col("label").is_not_null() & (pl.col("agreement") >= min_agreement))
        return self.items().join(lab.select("item_id", pl.col("label").alias(label_name)), on="item_id") \
            .drop([c for c in ("queued_at", "uncertainty", "predicted") if c in self.items().columns])


def simulate_active_learning(df: pl.DataFrame, target: str, *, initial: int = 50, batch: int = 25, rounds: int = 6,
                             strategies=("margin", "random"), test_size: float = 0.3, seed: int = 0) -> pl.DataFrame:
    """Replay labeling on data you already have labels for: start from ``initial`` labelled
    rows, add ``batch`` rows per round chosen by each strategy, and record test accuracy. Uses
    logistic regression on the numeric columns, so it is quick."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    num = [c for c, t in df.schema.items() if t.is_numeric() and c != target]
    X = df.select(num).fill_null(0).to_numpy().astype(float)
    y = df[target].cast(pl.Utf8).to_numpy()
    idx = rng.permutation(df.height)
    n_test = int(df.height * test_size)
    test, pool0 = idx[:n_test], idx[n_test:]
    rows = []
    for strategy in strategies:
        labelled, pool = list(pool0[:initial]), list(pool0[initial:])
        for r in range(rounds + 1):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[labelled], y[labelled])
            rows.append({"strategy": strategy, "round": r, "labelled": len(labelled),
                         "accuracy": round(float((m.predict(X[test]) == y[test]).mean()), 4)})
            if r == rounds or not pool:
                break
            if strategy == "random":
                pick = list(rng.choice(len(pool), min(batch, len(pool)), replace=False))
            else:
                score = _uncertainty(m.predict_proba(X[pool]), strategy)
                pick = list(np.argsort(-score)[:batch])
            chosen = {pool[i] for i in pick}
            labelled += list(chosen)
            pool = [p for p in pool if p not in chosen]
    return pl.DataFrame(rows)
