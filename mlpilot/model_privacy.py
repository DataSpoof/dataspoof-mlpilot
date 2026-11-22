"""Privacy attacks against your own model: does it leak who was in the training data, and how
easily can someone copy it through its API?

>>> mi = membership_inference_test(model, train_df, holdout_df)
>>> mi.auc, mi.advantage, mi.risk                 # 0.5 / 0.0 = no leakage
>>> ex = extraction_test(model, holdout_df, queries=(100, 500, 2000))
>>> ex.table                                      # surrogate agreement vs number of API queries

* **Membership inference** (the loss-threshold attack of Yeom et al.): overfitted models are
  more confident on rows they were trained on. The attack scores every row by the model's
  probability of its true label and tries to separate training rows from held-out rows.
  ``auc`` is the attack's ROC AUC: 0.5 means chance, so nothing leaks. ``advantage`` is the
  best TPR − FPR. Risk levels: low (AUC < 0.55), medium (< 0.65), high.
* **Model extraction**: an attacker sends synthetic queries, drawn from each column's observed
  values, and trains a decision tree on the answers. ``agreement`` is how often the copy matches
  your model on real held-out rows. High agreement after few queries means the API needs rate
  limits, query monitoring, and labels rather than probabilities.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl


@dataclasses.dataclass
class MembershipReport:
    auc: float
    advantage: float
    train_rows: int
    holdout_rows: int
    mean_confidence: dict            # {"train": …, "holdout": …}

    @property
    def risk(self) -> str:
        return "low" if self.auc < 0.55 else ("medium" if self.auc < 0.65 else "high")

    def summary(self) -> str:
        return (f"Membership inference: attack AUC {self.auc:.3f}, advantage {self.advantage:.3f} -> {self.risk} risk "
                f"(true-label confidence: train {self.mean_confidence['train']:.3f}, "
                f"held-out {self.mean_confidence['holdout']:.3f})")


def _true_label_confidence(model, df: pl.DataFrame, target: str) -> np.ndarray:
    proba = model.predict_proba(df.drop(target))
    classes = [c.removeprefix("proba_") for c in proba.columns]
    p = proba.to_numpy()
    y = df[target].cast(pl.Utf8).to_list()
    col = {c: i for i, c in enumerate(classes)}
    return np.array([p[i, col[v]] if v in col else 0.0 for i, v in enumerate(y)])


def membership_inference_test(model, train: pl.DataFrame, holdout: pl.DataFrame, *, target: str | None = None,
                              max_rows: int = 5000, seed: int = 0) -> MembershipReport:
    """Run the loss-threshold membership attack (classification models)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    target = target or getattr(model, "target", None)
    if target is None:
        raise ValueError("pass target= (the model doesn't say which column it predicts)")
    if getattr(model, "classes_", None) is None:
        raise ValueError("membership inference here needs a classifier (predict_proba)")
    rng = np.random.default_rng(seed)
    n = min(train.height, holdout.height, max_rows)
    tr = train[rng.choice(train.height, n, replace=False).tolist()]
    ho = holdout[rng.choice(holdout.height, n, replace=False).tolist()]
    s_in, s_out = _true_label_confidence(model, tr, target), _true_label_confidence(model, ho, target)
    scores = np.concatenate([s_in, s_out])
    member = np.concatenate([np.ones(n), np.zeros(n)])
    auc = float(roc_auc_score(member, scores))
    fpr, tpr, _ = roc_curve(member, scores)
    return MembershipReport(round(auc, 4), round(float(np.max(tpr - fpr)), 4), n, n,
                            {"train": round(float(s_in.mean()), 4), "holdout": round(float(s_out.mean()), 4)})


@dataclasses.dataclass
class ExtractionReport:
    table: pl.DataFrame              # queries, agreement
    target: str

    def summary(self) -> str:
        rows = ", ".join(f"{r['queries']} queries -> {r['agreement']:.0%}" for r in self.table.iter_rows(named=True))
        return f"Model extraction (decision-tree copy agreeing with your model on real rows): {rows}"


def extraction_test(model, holdout: pl.DataFrame, *, target: str | None = None, queries=(100, 500, 2000),
                    seed: int = 0) -> ExtractionReport:
    """Simulate stealing the model through its prediction API with synthetic queries."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    target = target or getattr(model, "target", None)
    X_real = holdout.drop(target) if target and target in holdout.columns else holdout
    rng = np.random.default_rng(seed)
    numeric = [c for c, t in X_real.schema.items() if t.is_numeric()]
    classify = getattr(model, "classes_", None) is not None

    def synth(n):
        cols = {}
        for c in X_real.columns:
            vals = X_real[c].drop_nulls()
            if vals.len() == 0:
                cols[c] = pl.Series(c, [None] * n, dtype=X_real.schema[c])
            elif c in numeric:
                cols[c] = pl.Series(c, rng.uniform(float(vals.min()), float(vals.max()), n)).cast(X_real.schema[c])
            else:
                cols[c] = pl.Series(c, vals.sample(n, with_replacement=True, seed=int(rng.integers(1 << 31))).to_list(),
                                    dtype=X_real.schema[c])
        return pl.DataFrame(cols)

    def encode(df):
        out = df.select(numeric).fill_null(0).to_numpy().astype(float) if numeric else np.zeros((df.height, 0))
        cats = [c for c in df.columns if c not in numeric]
        if cats:
            codes = np.column_stack([np.array([hash(str(v)) % 997 for v in df[c].to_list()], float) for c in cats])
            out = np.hstack([out, codes])
        return out

    truth = np.asarray(model.predict(X_real).to_list())
    rows = []
    for q in queries:
        Q = synth(int(q))
        answers = np.asarray(model.predict(Q).to_list())
        surrogate = (DecisionTreeClassifier if classify else DecisionTreeRegressor)(max_depth=12, random_state=seed)
        surrogate.fit(encode(Q), answers)
        copy = surrogate.predict(encode(X_real))
        if classify:
            agreement = float(np.mean(copy == truth))
        else:
            scale = float(np.std(truth)) or 1.0
            agreement = float(np.mean(np.abs(copy.astype(float) - truth.astype(float)) <= 0.1 * scale))
        rows.append({"queries": int(q), "agreement": round(agreement, 4)})
    return ExtractionReport(pl.DataFrame(rows), str(target))
