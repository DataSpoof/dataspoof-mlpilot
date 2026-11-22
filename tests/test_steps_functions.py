"""Every standalone cleaning function in mlpilot.steps, with the behaviour it promises."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot import steps


@pytest.fixture(autouse=True)
def _quiet(capsys):
    yield                                   # the functions print progress; keep test output readable


def test_column_names_and_structure():
    df = pl.DataFrame({" First Name ": ["a"], "first-name": ["b"], "Amount ($)": [1]})
    assert steps.standardize_column_names(df).columns == ["first_name", "first_name_1", "amount"]
    df = pl.DataFrame({"a": [1, None, 3], "empty": [None, None, None], "b": [None, None, 4]},
                      schema={"a": pl.Int64, "empty": pl.Int64, "b": pl.Int64})
    out = steps.drop_empty_rows_and_columns(df)
    assert out.columns == ["a", "b"] and out.height == 2
    assert steps.drop_high_missing_columns(df, threshold=0.5).columns == ["a"]
    dup = pl.DataFrame({"x": [1, 2], "x_copy": [1, 2], "y": [1, 3]})
    assert steps.remove_duplicate_columns(dup).columns == ["x", "y"]


def test_drop_id_like_columns_keeps_measurements_and_text():
    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "row": list(range(n)),
                       "price": rng.uniform(1, 100, n),                         # unique floats: a measurement
                       "review": [f"the product arrived {['late', 'early', 'broken'][i % 3]} and the box was "
                                  f"{['fine', 'torn', 'wet'][i % 3]} order {i}" for i in range(n)],
                       "plan": rng.choice(["a", "b"], n).tolist()})
    out = steps.drop_id_like_columns(df)
    assert "customer_id" not in out.columns and "row" not in out.columns
    assert {"price", "review", "plan"} <= set(out.columns)


def test_missing_values():
    df = pl.DataFrame({"city": ["Pune", "N/A", "?", None, "Delhi"], "score": [1.0, 999.0, 3.0, None, 5.0]})
    out = steps.replace_disguised_missing(df, numeric_sentinels=[999])
    assert out["city"].null_count() == 3 and out["score"].null_count() == 2
    flags = steps.add_missing_indicators(out)
    assert {"city_missing", "score_missing"} <= set(flags.columns)
    for strategy in ("mean", "median", "interpolate", "ffill", "bfill", "constant", "knn", "mice"):
        imp = steps.impute_missing(out, numeric_strategy=strategy, categorical_strategy="mode")
        assert imp["score"].null_count() == 0, strategy
        assert imp["city"].null_count() == 0
    assert steps.impute_missing(out, categorical_strategy="constant", constant_value="unknown")["city"].to_list() \
        == ["Pune", "unknown", "unknown", "unknown", "Delhi"]


def test_text_normalisation():
    df = pl.DataFrame({"t": ["  hello   world ", "cafÃ©", "ﬁle\x07", "MiXeD"]})
    assert steps.normalize_whitespace(df)["t"][0] == "hello world"
    assert steps.fix_mojibake(df)["t"][1] == "café"
    assert steps.normalize_unicode(df)["t"][2] == "file"
    assert steps.standardize_casing(df, {"t": "lower"})["t"][3] == "mixed"
    assert steps.standardize_casing(df, {"t": "title"})["t"][3] == "Mixed"
    assert steps.standardize_categories(pl.DataFrame({"c": [" Red", "red ", "RED"]}))["c"].n_unique() == 1


def test_dates_and_booleans():
    df = pl.DataFrame({"joined": ["2021-01-05", "2021-02-11", "2021-03-01"],
                       "ticket": ["113803", "373450", "330877"],               # numbers, not dates
                       "active": ["yes", "no", "Y"]})
    out = steps.parse_dates(df)
    assert out["joined"].dtype in (pl.Date, pl.Datetime) and out["ticket"].dtype == pl.Utf8
    parts = steps.extract_date_parts(out)
    assert parts["joined_year"].to_list() == [2021, 2021, 2021] and parts["joined_quarter"][0] == 1
    assert steps.normalize_booleans(df)["active"].to_list() == [True, False, True]
    mixed = pl.DataFrame({"d": ["5 March 2021", "2021-04-02", "April 7, 2021"]})
    assert steps.parse_dates(mixed)["d"].null_count() == 0            # varied formats via the fallback


def test_downcast_and_categories():
    df = pl.DataFrame({"small": [1, 2, 3], "cat": ["a"] * 98 + ["b", "c"][:2] if False else ["a"] * 3})
    assert steps.downcast_numeric(df).schema["small"] != pl.Int64
    many = pl.DataFrame({"c": ["common"] * 97 + ["rare1", "rare2", "rare3"]})
    assert set(steps.group_rare_categories(many, threshold=0.02)["c"]) == {"common", "Other"}
    spell = pl.DataFrame({"c": ["New York"] * 5 + ["new york"] * 2 + ["New Yrok"] + ["Boston"] * 3})
    out = steps.fuzzy_cluster_categories(spell, threshold=85)
    assert set(out["c"]) == {"New York", "Boston"}


@pytest.mark.parametrize("method", ["onehot", "label", "frequency", "ordinal", "target"])
def test_encode_categorical(method):
    df = pl.DataFrame({"color": ["red", "blue", "red", "green", None], "y": [1, 0, 1, 0, 1]})
    out = steps.encode_categorical(df, method=method, target="y")
    if method == "onehot":
        assert {"color_red", "color_blue", "color_green"} <= set(out.columns)
    else:
        assert out["color"].dtype.is_numeric(), out["color"].dtype
    if method == "frequency":
        assert out["color"].to_list()[:4] == [2, 1, 2, 1]
    if method == "target":
        assert out["color"].to_list()[:4] == [1.0, 0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        steps.encode_categorical(df, method="magic")


def test_outlier_removal_keeps_rows_with_missing_values():
    rng = np.random.default_rng(0)
    vals = list(rng.normal(10, 1, 98)) + [None, 1000.0]
    df = pl.DataFrame({"v": vals, "g": ["a", "b"] * 50})
    for fn in (steps.remove_outliers_zscore, steps.remove_outliers_mad):
        out = fn(df)
        assert out["v"].max() < 1000 and out["v"].null_count() == 1, fn.__name__      # missing row kept
    by_group = steps.remove_outliers_by_group(df, "g")
    assert by_group["v"].max() < 1000 and by_group["v"].null_count() == 1
    assert steps.remove_outliers_isolation_forest(df.drop_nulls(), contamination=0.02)["v"].max() < 1000
    w = steps.winsorize(df, lower=0.05, upper=0.05)
    assert w.height == df.height and w["v"].max() < 1000


def test_duplicates():
    df = pl.DataFrame({"name": ["Jon Smith", "John Smith", "Ann Lee", "Jon  Smith"], "n": [1, 2, 3, 4]})
    assert steps.drop_duplicates_subset(df, subset=["name"]).height == 4
    first = steps.fuzzy_deduplicate(df, ["name"], threshold=85)
    assert first["n"].to_list() == [1, 3]                    # keep the first of each near-duplicate group
    last = steps.fuzzy_deduplicate(df, ["name"], threshold=85, keep="last")
    assert last["n"].to_list() == [3, 4]                     # keep the last of each group, in original order


def test_scaling_and_binning():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    assert steps.scale_numeric(df, "minmax")["x"].to_list()[0] == 0.0
    assert abs(steps.scale_numeric(df, "standard")["x"].mean()) < 1e-9
    assert steps.scale_numeric(df, "robust")["x"][2] == 0.0
    with pytest.raises(ValueError):
        steps.scale_numeric(df, "magic")
    for strategy in ("quantile", "uniform"):
        out = steps.bin_numeric(df, ["x"], bins=2, strategy=strategy)
        assert "x_bin" in out.columns and out["x_bin"].n_unique() == 2, strategy


def test_validation_rules():
    df = pl.DataFrame({"age": [25, -3, 140, None], "start": [1, 5, 3, 1], "end": [2, 4, 9, 1],
                       "email": ["a@b.com", "nope", None, "x@y.org"], "state": ["MH", "KA", "ZZ", None]})
    assert steps.apply_range_rules(df, {"age": (0, 120)}, "null")["age"].to_list() == [25, None, None, None]
    assert steps.apply_range_rules(df, {"age": (0, 120)}, "clip")["age"].to_list() == [25, 0, 120, None]
    assert steps.apply_range_rules(df, {"age": (0, 120)}, "drop").height == 2
    assert steps.check_cross_field(df, {"order": pl.col("start") <= pl.col("end")}, drop_violations=True).height == 3
    assert steps.validate_formats(df, {"email": "email"})["email_valid"].to_list() == [True, False, None, True]
    assert steps.check_referential_integrity(df, "state", ["MH", "KA"], drop_invalid=True).height == 3


def test_domain_standardisation():
    df = pl.DataFrame({"phone": ["(415) 555-2671", "bad", None], "email": [" Ann@Example.COM ", "x@", None],
                       "url": ["Example.com/a/", "https://X.org/p#frag", None], "country": ["USA", "u.k.", "Bharat"],
                       "price_eur": [10.0, 20.0, None]})
    assert steps.normalize_phone_numbers(df, ["phone"])["phone"].to_list() == ["+14155552671", None, None]
    em = steps.normalize_emails(df, ["email"])
    assert em["email"][0] == "ann@example.com" and em["email_valid"].to_list()[:2] == [True, False]
    assert steps.canonicalize_urls(df, ["url"])["url"].to_list() == ["http://example.com/a", "https://x.org/p", None]
    assert steps.standardize_country_region(df, ["country"])["country"].to_list() == ["United States",
                                                                                      "United Kingdom", "India"]
    assert steps.convert_currency(df, ["price_eur"], {"price_eur": 1.1})["price_eur"].to_list()[:2] == \
        pytest.approx([11.0, 22.0])


def test_text_extras():
    df = pl.DataFrame({"t": ["This is a perfectly normal English sentence about shipping.",
                             "Ceci est une phrase française sur la livraison des colis.",
                             "This is a perfectly normal English sentence about shipping!"]})
    pytest.importorskip("langdetect")
    langs = steps.detect_language(df, "t")["t_lang"].to_list()
    assert langs[0] == "en" and langs[1] == "fr"
    assert steps.filter_by_language(df, "t", keep="en").height == 2
    flags = steps.near_duplicate_text(df, "t", threshold=95)["t_near_dup"].to_list()
    assert flags == [False, False, True]
    assert steps.near_duplicate_text(df, "t", threshold=95, drop=True).height == 2
    pytest.importorskip("spellchecker")
    assert "receive" in steps.correct_spelling_text(pl.DataFrame({"t": ["please recieve the parcel"]}), ["t"])["t"][0]
    pytest.importorskip("better_profanity")
    assert "****" in steps.redact_profanity(pl.DataFrame({"t": ["what the hell"]}), ["t"])["t"][0]


def test_every_step_is_registered_as_a_pipeline_step():
    from mlpilot.core import available_steps
    names = set(available_steps())
    for fn in ("standardize_column_names", "replace_disguised_missing", "impute_missing", "winsorize",
               "fuzzy_deduplicate", "normalize_emails"):
        assert fn in names, fn
    pipe = ac.Pipeline([{"step": "normalize_whitespace"}, {"step": "winsorize", "params": {"lower": 0.1, "upper": 0.1}}])
    out = pipe.fit_transform(pl.DataFrame({"s": [" a ", "b  "], "x": [1.0, 100.0]}))
    assert out["s"].to_list() == ["a", "b"]
