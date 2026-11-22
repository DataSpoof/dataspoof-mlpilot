"""Software supply-chain security: scan model files before trusting them, and list what you ship.

>>> rep = scan_model("models/churn")        # a saved model directory or a single file
>>> rep.passed, rep.findings                # critical: code that would run on load (os.system, eval …)
>>> generate_sbom("sbom.cdx.json")          # CycloneDX 1.5 bill of materials of the environment

**How the scan works.** Pickle and joblib files can run arbitrary code when loaded. The scanner
replays the file with a restricted unpickler:
- Every global the file imports is recorded.
- Only allowlisted library classes (numpy, scipy, scikit-learn, polars, LightGBM, XGBoost,
  MLPilot, safe builtins) are resolved.
- Everything else is replaced by an inert stub, so nothing from the file can execute.
- Known-dangerous callables (``os.system``, ``subprocess``, ``eval``, ``exec``,
  ``__import__``, socket and network modules, nested ``pickle`` or ``load`` calls) are
  **critical**. Other unknown globals are **warnings**.

ONNX files are checked for custom operator domains and external data files, which load native
code or read outside the model.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import pickle
import re
import uuid
from pathlib import Path

ALLOWED_ROOTS = {"numpy", "scipy", "sklearn", "polars", "pyarrow", "pandas", "lightgbm", "xgboost", "mlpilot",
                 "collections", "copyreg", "datetime", "decimal", "_codecs", "joblib", "functools", "re", "uuid",
                 "fractions", "numbers", "enum", "pathlib", "zoneinfo", "_operator", "operator", "array", "math"}
SAFE_BUILTINS = {"set", "frozenset", "list", "dict", "tuple", "slice", "complex", "bytearray", "bytes", "range",
                 "object", "int", "float", "str", "bool", "NoneType", "Ellipsis", "NotImplemented"}
DANGEROUS_ROOTS = {"os", "posix", "nt", "subprocess", "sys", "importlib", "runpy", "socket", "shutil", "pty",
                   "ctypes", "pickle", "_pickle", "marshal", "webbrowser", "code", "codeop", "requests", "urllib",
                   "http", "ftplib", "smtplib", "telnetlib", "asyncio", "multiprocessing", "threading", "signal",
                   "tempfile", "glob", "io", "builtins", "__builtin__", "platform", "commands", "dill", "cloudpickle",
                   "torch", "pip", "setuptools", "site", "inspect", "types", "gc", "weakref"}
RISKY_NAME = re.compile(r"(?i)(^|_)(eval|exec|system|popen|spawn|load|loads|read_pickle|import|open|compile|"
                        r"getattr|setattr|delattr|globals|locals|__import__|input|breakpoint|call|run)($|_)")


@dataclasses.dataclass
class Finding:
    file: str
    severity: str          # critical | warning
    kind: str              # global | onnx_domain | onnx_external_data | unreadable
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {Path(self.file).name}: {self.kind} {self.detail}"


@dataclasses.dataclass
class ScanReport:
    path: str
    files: list
    findings: list
    globals_seen: dict      # file -> sorted list of "module.name"

    @property
    def critical(self) -> list:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def passed(self) -> bool:
        return not self.critical

    def summary(self) -> str:
        head = (f"Model scan of {self.path}: {'PASSED' if self.passed else 'FAILED'} - {len(self.files)} file(s), "
                f"{len(self.critical)} critical, {len(self.findings) - len(self.critical)} warning(s)")
        return "\n".join([head] + [f"  {f}" for f in self.findings])

    def __repr__(self) -> str:
        return self.summary()


class _Inert:
    """Stand-in for a global the scanner refuses to resolve: accepts any construction, does nothing."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return self


def _classify(module: str, name: str) -> str | None:
    """None when the global is allowed; otherwise 'critical' or 'warning'."""
    root = module.split(".")[0]
    if root in ("builtins", "__builtin__"):
        return None if name in SAFE_BUILTINS else "critical"
    if root in DANGEROUS_ROOTS:
        return "critical"
    if RISKY_NAME.search(name):
        return "critical" if root not in ("numpy", "scipy", "sklearn", "polars", "pandas") else "warning"
    if root in ALLOWED_ROOTS:
        return None
    return "warning"


def _restricted(base):
    class Restricted(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen, self.flags = set(), []

        def find_class(self, module, name):
            self.seen.add(f"{module}.{name}")
            verdict = _classify(module, name)
            if verdict is None:
                return super().find_class(module, name)
            self.flags.append((verdict, f"{module}.{name}"))
            return type(f"Inert[{module}.{name}]", (_Inert,), {})

        def persistent_load(self, pid):          # never resolve external references
            self.flags.append(("warning", f"persistent id {pid!r}"))
            return _Inert()
    return Restricted


def _scan_pickle(path: Path) -> tuple:
    """Returns (globals_seen, [(severity, detail)])."""
    with path.open("rb") as f:
        head = f.read(16)
    joblib_like = path.suffix in (".joblib", ".jbl", ".z", ".gz", ".bz2", ".xz", ".lzma", ".lz4") or head[:2] == b"\x1f\x8b"
    if joblib_like:
        from joblib.numpy_pickle import NumpyUnpickler, _validate_fileobject_and_memmap
        with path.open("rb") as raw:
            with _validate_fileobject_and_memmap(raw, str(path), None) as (fobj, _):
                if isinstance(fobj, str):
                    return set(), [("warning", "pre-0.10 joblib format, not scanned")]
                up = _restricted(NumpyUnpickler)(str(path), fobj, True)
                up.load()
    else:
        with path.open("rb") as f:
            up = _restricted(pickle._Unpickler)(f)          # pure-Python unpickler: find_class is honoured
            up.load()
    return up.seen, up.flags


def _scan_onnx(path: Path) -> list:
    try:
        import onnx
    except ImportError:
        return [("warning", "onnx not installed; ONNX file not inspected")]
    model = onnx.load(str(path), load_external_data=False)
    out = []
    standard = {"", "ai.onnx", "ai.onnx.ml", "com.microsoft"}
    for d in sorted({op.domain for op in model.opset_import} - standard):
        out.append(("warning", f"custom operator domain {d!r} (needs a native library at load)"))
    for t in model.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            out.append(("warning", f"tensor {t.name!r} reads external data"))
    return out


PICKLE_SUFFIXES = {".pkl", ".pickle", ".joblib", ".jbl", ".pt", ".pth", ".bin", ".sav", ".dat"}


def scan_model(path, *, include_unknown: bool = False) -> ScanReport:
    """Scan a model file or directory without loading it the unsafe way. ``include_unknown``
    also scans files with unrecognised extensions as pickles."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    findings, seen_by_file, scanned = [], {}, []
    for f in files:
        suffix = f.suffix.lower()
        try:
            if suffix == ".onnx":
                scanned.append(str(f))
                for sev, detail in _scan_onnx(f):
                    findings.append(Finding(str(f), sev, "onnx", detail))
            elif suffix in PICKLE_SUFFIXES or include_unknown and suffix not in (".json", ".md", ".txt", ".html",
                                                                               ".yaml", ".yml", ".csv", ".parquet"):
                scanned.append(str(f))
                seen, flags = _scan_pickle(f)
                seen_by_file[str(f)] = sorted(seen)
                for sev, detail in flags:
                    findings.append(Finding(str(f), sev, "global", detail))
        except Exception as exc:                 # a file we can't parse can't be vouched for
            findings.append(Finding(str(f), "warning", "unreadable", f"{type(exc).__name__}: {exc}"[:200]))
    findings.sort(key=lambda x: (x.severity != "critical", x.file))
    return ScanReport(str(root), scanned, findings, seen_by_file)


# ── SBOM ────────────────────────────────────────────────────────────────────
def _license(meta) -> str | None:
    lic = meta.get("License-Expression") or meta.get("License")
    if lic and len(lic) < 80 and "\n" not in lic:
        return lic.strip()
    for c in meta.get_all("Classifier") or []:
        if c.startswith("License :: OSI Approved :: "):
            return c.split("::")[-1].strip()
    return None


def _requires(dist) -> set:
    from packaging.requirements import Requirement
    out = set()
    for r in dist.requires or []:
        try:
            req = Requirement(r)
        except Exception:
            continue
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            out.add(req.name.lower().replace("_", "-"))
    return out


def generate_sbom(path=None, *, root: str | None = "dataspoof-mlpilot", all_packages: bool = False) -> dict:
    """A CycloneDX 1.5 SBOM (JSON) of the current Python environment.

    By default it covers ``root`` and everything it depends on, as installed (``all_packages``:
    every installed distribution). Each component has a name, version, purl and licence, and the
    dependency graph links them. Written to ``path`` when given; the dict is returned either way."""
    from importlib import metadata
    dists = {}
    for d in metadata.distributions():
        name = (d.metadata["Name"] or "").lower().replace("_", "-")
        if name and name not in dists:
            dists[name] = d
    if all_packages or root is None or root not in dists:
        chosen = set(dists)
    else:
        chosen, todo = set(), [root]
        while todo:
            n = todo.pop()
            if n in chosen or n not in dists:
                continue
            chosen.add(n)
            todo.extend(_requires(dists[n]))
    components, deps = [], []
    for n in sorted(chosen):
        d = dists[n]
        ref = f"pkg:pypi/{n}@{d.version}"
        comp: dict = {"type": "library", "bom-ref": ref, "name": n, "version": d.version, "purl": ref}
        lic = _license(d.metadata)
        if lic:
            comp["licenses"] = [{"license": {"name": lic}}]
        components.append(comp)
        deps.append({"ref": ref, "dependsOn": sorted(f"pkg:pypi/{r}@{dists[r].version}" for r in _requires(d)
                                                      if r in chosen)})
    sbom = {"bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": f"urn:uuid:{uuid.uuid4()}", "version": 1,
            "metadata": {"timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                         "tools": [{"vendor": "DataSpoof", "name": "mlpilot.supply_chain"}],
                         **({"component": next(c for c in components if c["name"] == root)}
                            if root in chosen and not all_packages else {})},
            "components": components, "dependencies": deps}
    if path:
        Path(path).write_text(json.dumps(sbom, indent=2), encoding="utf-8")
    return sbom
