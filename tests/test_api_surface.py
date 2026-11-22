"""The public surface: every module imports, every advertised name exists, every CLI command answers --help."""

import importlib
import pkgutil
import subprocess
import sys

import pytest

import mlpilot

# Modules whose import needs an optional extra; they must fail with an ImportError that names it.
OPTIONAL = {"mlpilot.hub.app": ("fastapi", "sqlalchemy", "jwt"), "mlpilot.deploy.asgi": ("starlette", "fastapi"),
            "mlpilot.integrations.airflow": ("airflow",), "mlpilot.integrations.dagster": ("dagster",),
            "mlpilot.integrations.prefect": ("prefect",)}
# Modules that do work at import time (entry points), not libraries to import blindly.
ENTRY_POINTS = {"mlpilot.deploy.wsgi", "mlpilot.deploy.asgi", "mlpilot.deploy.__main__"}


def _modules():
    for info in pkgutil.walk_packages(mlpilot.__path__, "mlpilot."):
        if info.name not in ENTRY_POINTS and "._rustcore" not in info.name:
            yield info.name


MODULES = sorted(_modules())


def test_there_are_many_modules():
    assert len(MODULES) > 80


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_and_exports_exist(name):
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        needs = OPTIONAL.get(name)
        # acceptable only when it tells the user what to install
        if "pip install" in str(exc) or (needs and any(n in str(exc).lower() for n in needs)):
            pytest.skip(f"{name} needs an optional extra: {exc}")
        raise
    missing = [n for n in getattr(mod, "__all__", []) if not hasattr(mod, n)]
    assert not missing, f"{name}.__all__ lists names that don't exist: {missing}"


def test_top_level_names():
    names = [n for n in dir(mlpilot) if not n.startswith("_")]
    assert len(names) > 150
    for n in ("AutoML", "ModelRegistry", "Monitor", "Pipeline", "read", "ingest", "evaluate_model", "score",
              "export_model", "quantize_model", "optimize_dtypes", "Goal", "goal_from_text", "scan_model",
              "generate_sbom", "split_data", "check_quality", "profile", "find_insights", "clean_data"):
        assert hasattr(mlpilot, n), n
    assert mlpilot.orchestration.run and mlpilot.media.conditions and mlpilot.supply_chain


def _cli_commands():
    from mlpilot.deploy.cli import main
    import argparse
    seen = {}
    orig = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                seen.update(action.choices)
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = capture
    try:
        try:
            main([])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    return sorted(seen)


CLI = _cli_commands()


def test_cli_has_every_command():
    for cmd in ("serve", "score", "docker", "export", "quantize", "monitor", "hub", "train", "gate", "register",
                "promote", "release", "latest", "deploy-sagemaker", "cicd", "plan", "run", "approve", "status",
                "scan-model", "sbom", "backup", "restore", "dr-drill"):
        assert cmd in CLI, cmd


@pytest.mark.parametrize("cmd", CLI)
def test_cli_help(cmd):
    res = subprocess.run([sys.executable, "-m", "mlpilot.deploy", cmd, "--help"], capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, res.stderr[-800:]
    assert "usage:" in res.stdout
