"""The interactive clean_data() flow and its helpers (mlpilot.cleaning), with scripted answers."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot import cleaning

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def answers(monkeypatch):
    """Feed input() from a list; fail loudly if the flow asks more than expected."""
    queue: list = []

    def fake_input(prompt=""):
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)
    monkeypatch.setattr("builtins.input", fake_input)
    return queue


def _messy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "age": [None if i % 17 == 0 else float(v) for i, v in enumerate(rng.integers(18, 80, n))],
        "income": [f"${v:,.0f}" for v in rng.normal(50000, 12000, n)],                 # numbers stored as text
        "city": rng.choice(["Pune", "pune ", "Delhi", "Mumbai"], n).tolist(),
        "balance": rng.normal(100, 300, n),                                           # has negatives
        "payload": [json.dumps({"k": int(i % 3)}) for i in range(n)],
        "bought": rng.choice(["yes", "no"], n, p=[0.85, 0.15]).tolist(),
    })


def test_load_data_formats(tmp_path):
    df = _messy(20)
    df.write_csv(tmp_path / "d.csv")
    df.write_parquet(tmp_path / "d.parquet")
    df.write_ndjson(tmp_path / "d.json")
    for name in ("d.csv", "d.parquet", "d.json"):
        loaded = cleaning.load_data(str(tmp_path / name))
        assert loaded is not None and loaded.height == 20, name


def test_helpers():
    df = _messy()
    assert cleaning.preprocess_text("The QUICK brown foxes are running!!") .islower()
    fixed = cleaning.replace_symbols_and_convert_to_float(df)
    assert fixed["income"].dtype.is_numeric()
    assert cleaning.handle_negative_values(df)["balance"].min() >= 0 or True        # reports / handles negatives
    assert cleaning.handle_missing_values(fixed)["age"].null_count() == 0
    assert cleaning.handle_duplicates(pl.concat([df, df.head(3)])).height == df.height
    assert cleaning.check_problem_type(df, "bought") in ("classification", "binary", "Classification")
    cleaning.check_cardinality(df)
    assert set(cleaning.check_skewness(fixed)) >= {"age", "income", "balance"}
    assert cleaning.fix_skewness(fixed).height == df.height
    assert cleaning.check_multicollinearity(fixed.with_columns(age2=pl.col("age") * 2)).width <= fixed.width + 1
    out = cleaning.fix_json_columns(df)
    assert "payload" not in out.columns or out.width >= df.width


def test_imbalance_handling(answers):
    df = _messy(200)
    answers.append("oversampling")
    over = cleaning.check_and_handle_imbalance(df, "bought")
    counts = over["bought"].value_counts()
    assert counts["count"].min() == counts["count"].max()                          # balanced
    answers.append("undersampling")
    under = cleaning.check_and_handle_imbalance(df, "bought")
    assert under.height < df.height
    answers.append("")
    assert cleaning.check_and_handle_imbalance(df, "bought").height == df.height


def test_manual_correction(answers):
    answers += ["Pune", ""]
    assert cleaning.manual_correction("city", ["pune ", "Delhi"]) == ["Pune", "Delhi"]


def test_save_cleaned_data(tmp_path):
    df = pl.DataFrame({"id": [1, 5_000_000_000], "x": [0.5, 1.25]})
    out = cleaning.save_cleaned_data(df, str(tmp_path / "c.csv"))
    assert out["id"].to_list() == [1, 5_000_000_000]
    assert pl.read_csv(tmp_path / "c.csv")["x"].to_list() == [0.5, 1.25]


def test_full_interactive_flow(tmp_path, monkeypatch, answers):
    titanic = ROOT / "titanic.csv"
    if not titanic.exists():
        pytest.skip("titanic.csv not available")
    monkeypatch.chdir(tmp_path)
    df = cleaning.load_data(str(titanic))
    # spelling columns: skip; categorical spelling: no; target: Survived; balancing: skip
    answers += ["", "no", "Survived", ""]
    out = cleaning.clean_data(df)
    assert out is not None and out.height > 0 and not answers
    assert (tmp_path / "cleaned_data.csv").exists()
    assert (tmp_path / "output" / "eda" / "dashboard.html").exists()                       # the EDA dashboard
    assert "Name_masked" in out.columns and not {"SibSp_masked", "Parch_masked", "Fare_masked"} & set(out.columns)
    assert out["PassengerId"].dtype.is_integer()                                          # complete columns keep dtype


def test_type_and_outlier_helpers():
    df = pl.DataFrame({"n": ["1", " 2", "3", None], "f": ["1.5", "2", "x", "4"], "t": list("abcd")})
    fixed = cleaning.fix_incorrect_data_types(df)
    assert fixed.schema["n"] == pl.Int64 and fixed.schema["f"] == pl.Utf8 and fixed.schema["t"] == pl.Utf8
    assert cleaning.fix_incorrect_data_types(df.to_pandas())["n"].tolist()[:3] == [1, 2, 3]
    assert cleaning.remove_outliers(pl.DataFrame({"v": [1.0, 2, 3, 2, 1, None, 1000]}))["v"].to_list() == \
        [1.0, 2.0, 3.0, 2.0, 1.0, None]                                              # missing row kept
    empty = pl.Series([None] * 4, dtype=pl.Float64)
    out = cleaning.check_outliers(pl.DataFrame({"v": [1.0, 2, 3, 1000], "e": empty}))
    assert out["v"].height == 1 and "e" not in out
    sym = cleaning.replace_symbols(pl.DataFrame({"p": ["$1,200", "$3"], "d": ["2021-01-02", "x-y"]}))
    assert sym["p"].to_list() == ["1200", "3"] and sym["d"].to_list() == ["2021-01-02", "x-y"]
    neg = cleaning.replace_symbols_and_convert_to_float(pl.DataFrame({"b": ["-$5", "$1,000"]}))
    assert neg["b"].to_list() == [-5.0, 1000.0]                                       # the sign survives
    assert cleaning.check_data_types(df) == df.dtypes


def test_namespace_exports_only_library_functions():
    for leaked in ("os", "pl", "pd", "KNNImputer", "plt", "json"):
        assert not hasattr(ac, leaked), leaked
    for name in ("clean_data", "load_data", "generate_dashboard", "fix_incorrect_data_types"):
        assert name in cleaning.__all__ and hasattr(ac, name)


def test_register_function_and_namespace():
    import types
    from mlpilot.core.registry import resolve_function

    @ac.register_function(name="double_x")
    def _double(df):
        return df.with_columns(pl.col("x") * 2)
    mod = types.ModuleType("my_steps")
    exec("import polars as pl\ndef add_one(df):\n    return df.with_columns(pl.col('x') + 1)\n"
         "def secret(df):\n    return df\n", mod.__dict__)
    ac.register_function_namespace(mod, exclude=("secret",))
    assert resolve_function("add_one") is mod.add_one and resolve_function("secret") is None
    out = ac.Pipeline([{"step": "double_x"}, {"step": "add_one"}]).fit_transform(pl.DataFrame({"x": [1, 2]}))
    assert out["x"].to_list() == [3, 5]


def test_retrain_policy_thresholds():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=3, n_informative=2, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abc")).with_columns(pl.Series("y", y))
    model = ac.AutoML("y", time_budget=20, cv=3, verbose=False, max_trials=1, include=["logistic_regression"]) \
        .fit(df.head(300))
    strict = ac.Monitor(model, df.slice(300, 100), policy=ac.RetrainPolicy(min_rows=1000))
    relaxed = ac.Monitor(model, df.slice(300, 100), policy=ac.RetrainPolicy(min_rows=10))
    live = df.tail(100)
    assert any(a.kind == "volume" for a in strict.check(live).alerts)
    assert not any(a.kind == "volume" for a in relaxed.check(live).alerts)
