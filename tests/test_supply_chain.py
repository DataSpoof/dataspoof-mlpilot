"""Model-file scanning (without executing anything) and the CycloneDX SBOM."""

import json
import os
import pickle

import joblib
import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.supply_chain import generate_sbom, scan_model


class _Evil:
    """Unpickling this calls os.system - the classic malicious model file."""

    def __reduce__(self):
        return (os.system, ("echo this-should-never-run",))


class _Sneaky:
    def __reduce__(self):
        return (eval, ("1 + 1",))


def test_clean_automl_model_passes(tmp_path):
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)}).with_columns(
        y=(pl.col("a") > 0).cast(pl.Int8))
    model = ac.AutoML("y", time_budget=5, include=["logistic_regression"], verbose=False).fit(df)
    model.save(tmp_path / "model")
    rep = scan_model(tmp_path / "model")
    assert rep.passed, rep.summary()
    assert any(f.endswith("model.joblib") for f in rep.files)
    seen = next(iter(rep.globals_seen.values()))
    assert any(g.startswith("sklearn.") for g in seen)


def test_malicious_pickles_are_caught_without_running(tmp_path, capfd):
    (tmp_path / "evil.pkl").write_bytes(pickle.dumps({"weights": [1, 2], "x": _Evil()}))
    joblib.dump({"model": _Sneaky()}, tmp_path / "sneaky.joblib", compress=3)
    rep = scan_model(tmp_path)
    assert not rep.passed
    details = {(os.path.basename(f.file), f.detail) for f in rep.critical}
    system = "nt.system" if os.name == "nt" else "posix.system"
    assert ("evil.pkl", system) in details
    assert ("sneaky.joblib", "builtins.eval") in details
    assert "this-should-never-run" not in capfd.readouterr().out          # nothing executed
    assert "FAILED" in rep.summary()


def test_unknown_globals_warn_and_unreadable_files_are_reported(tmp_path):
    import fractions
    import collections
    pickle.dump({"c": collections.Counter("aab"), "f": fractions.Fraction(1, 3)}, open(tmp_path / "ok.pkl", "wb"))
    pickle.dump(pytest.approx, open(tmp_path / "odd.pkl", "wb"))            # an unknown library function
    (tmp_path / "broken.pkl").write_bytes(b"\x80\x05garbage")
    rep = scan_model(tmp_path)
    assert rep.passed                                                       # nothing critical
    kinds = {(os.path.basename(f.file), f.kind, f.severity) for f in rep.findings}
    assert ("odd.pkl", "global", "warning") in kinds
    assert ("broken.pkl", "unreadable", "warning") in kinds
    assert not any(os.path.basename(f.file) == "ok.pkl" for f in rep.findings)


def test_sbom_is_cyclonedx_with_dependency_graph(tmp_path):
    sbom = generate_sbom(tmp_path / "sbom.json", root="scikit-learn")
    assert sbom["bomFormat"] == "CycloneDX" and sbom["specVersion"] == "1.5"
    names = {c["name"] for c in sbom["components"]}
    assert {"scikit-learn", "numpy", "scipy", "joblib"} <= names and "polars" not in names
    sk = next(d for d in sbom["dependencies"] if d["ref"].startswith("pkg:pypi/scikit-learn@"))
    assert any(r.startswith("pkg:pypi/numpy@") for r in sk["dependsOn"])
    assert sbom["metadata"]["component"]["name"] == "scikit-learn"
    assert json.loads((tmp_path / "sbom.json").read_text())["serialNumber"].startswith("urn:uuid:")
    assert len(generate_sbom(all_packages=True)["components"]) > len(names)
