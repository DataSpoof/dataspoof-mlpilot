"""Backup, restore and disaster-recovery drills for everything MLPilot keeps on disk.

>>> m = backup({"registry": "registry", "runs": "runs", "monitors": "monitoring"}, "backups",
...            sqlite={"hub": "hub.db"})
>>> verify_backup(m.path)                      # every archive re-hashed against the manifest
>>> restore(m.path, {"registry": "restored/registry"})
>>> report = dr_drill({"registry": "registry"}, "dr-workdir")   # backup -> restore -> compare
>>> report.passed, report.rto_seconds, report.rpo_seconds
>>> prune_backups("backups", keep_last=7)

**Backups.** Each named source (a directory or a file) becomes one ``.tar.gz``. The manifest
records each archive's SHA-256 and every file's hash and size. SQLite databases, such as the Hub's
default store, are copied with SQLite's online backup API, so the copy is consistent while the
Hub is running. For PostgreSQL, run ``pg_dump`` or use your cloud's snapshots, then add the dump
file as a source.

**Restores.** Checksums are verified before anything is extracted. Extraction refuses absolute
paths, ``..`` and links. A target that isn't empty is left alone unless ``overwrite=True``.

**Drills.** A drill backs up, restores into a scratch directory and compares every file's hash.
It reports RTO (how long the restore took) and RPO (how old the restored data is).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree(root: Path) -> dict:
    root = Path(root)
    if root.is_file():
        return {root.name: _sha(root)}
    return {str(p.relative_to(root)).replace("\\", "/"): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


@dataclasses.dataclass
class BackupManifest:
    path: Path
    created: str
    sources: dict          # name -> {"origin", "archive", "sha256", "files", "bytes", "kind"}

    def to_dict(self) -> dict:
        return {"created": self.created, "sources": self.sources}


def backup(sources: dict, dest, *, sqlite: dict | None = None, label: str | None = None) -> BackupManifest:
    """Back up ``sources`` ({name: directory or file}) and ``sqlite`` ({name: database file})
    into a new timestamped folder under ``dest``."""
    from importlib import metadata
    created = _dt.datetime.now(_dt.timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(dest) / f"backup-{stamp}{'-' + label if label else ''}"
    out.mkdir(parents=True)
    entries: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        staged = dict(sources)
        for name, db in (sqlite or {}).items():
            copy = Path(tmp) / f"{name}.sqlite"
            src = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
            dst = sqlite3.connect(copy)
            with dst:
                src.backup(dst)                         # consistent snapshot of a live database
            src.close()
            dst.close()
            staged[name] = copy
        for name, origin in staged.items():
            origin = Path(origin)
            if not origin.exists():
                raise FileNotFoundError(f"backup source {name!r}: {origin} does not exist")
            files = _tree(origin)
            archive = out / f"{name}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                if origin.is_file():
                    tar.add(origin, arcname=origin.name)
                else:
                    for rel in files:
                        tar.add(origin / rel, arcname=rel)
            entries[name] = {"origin": str(sources.get(name, (sqlite or {}).get(name))), "archive": archive.name,
                             "sha256": _sha(archive), "kind": "sqlite" if name in (sqlite or {}) else
                             ("file" if origin.is_file() else "directory"),
                             "files": files, "bytes": sum((origin / r if origin.is_dir() else origin).stat().st_size
                                                          for r in files)}
    try:
        version = metadata.version("dataspoof-mlpilot")
    except metadata.PackageNotFoundError:
        version = None
    created_at = created.isoformat(timespec="seconds")
    manifest = {"created": created_at, "mlpilot": version, "sources": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return BackupManifest(out, created_at, entries)


def _manifest(backup_path) -> dict:
    p = Path(backup_path) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{backup_path} has no manifest.json (not an MLPilot backup)")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_backup(backup_path) -> dict:
    """``{source: "ok" | "missing" | "corrupt"}`` after re-hashing every archive."""
    m, out = _manifest(backup_path), {}
    for name, e in m["sources"].items():
        archive = Path(backup_path) / e["archive"]
        out[name] = "missing" if not archive.exists() else ("ok" if _sha(archive) == e["sha256"] else "corrupt")
    return out


def _safe_members(tar: tarfile.TarFile):
    for m in tar.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
            raise ValueError(f"refusing unsafe archive member {m.name!r}")
        yield m


def restore(backup_path, targets: dict, *, overwrite: bool = False) -> dict:
    """Restore the named sources into ``targets`` ({name: destination}). A ``file`` / ``sqlite``
    source is restored to exactly the destination path. Returns ``{name: files restored}``."""
    m = _manifest(backup_path)
    status = verify_backup(backup_path)
    restored = {}
    for name, dest in targets.items():
        if name not in m["sources"]:
            raise KeyError(f"{name!r} is not in this backup (has: {sorted(m['sources'])})")
        if status[name] != "ok":
            raise ValueError(f"backup of {name!r} is {status[name]}; not restoring it")
        e, dest = m["sources"][name], Path(dest)
        single = e["kind"] in ("file", "sqlite")
        if dest.exists() and (single or any(dest.iterdir())) and not overwrite:
            raise FileExistsError(f"{dest} exists; pass overwrite=True to replace it")
        with tarfile.open(Path(backup_path) / e["archive"], "r:gz") as tar:
            members = list(_safe_members(tar))
            if single:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(members[0]) as src, dest.open("wb") as fh:   # type: ignore[union-attr]
                    shutil.copyfileobj(src, fh)
            else:
                if dest.exists() and overwrite:
                    shutil.rmtree(dest)
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    tar.extractall(dest, members=members, filter="data")
                except TypeError:                       # Python < 3.11.4: members were validated above
                    tar.extractall(dest, members=members)
        restored[name] = len(e["files"])
    return restored


@dataclasses.dataclass
class DrillReport:
    passed: bool
    rto_seconds: float          # restore time
    rpo_seconds: float          # age of the data restored (backup start -> restore end)
    sources: dict               # name -> {"files", "mismatched", "missing"}
    backup_path: Path

    def summary(self) -> str:
        lines = [f"DR drill: {'PASSED' if self.passed else 'FAILED'} - RTO {self.rto_seconds:.2f}s, "
                 f"RPO {self.rpo_seconds:.2f}s"]
        lines += [f"  {k}: {v['files']} files, {len(v['mismatched'])} mismatched, {len(v['missing'])} missing"
                  for k, v in self.sources.items()]
        return "\n".join(lines)


def dr_drill(sources: dict, workdir, *, sqlite: dict | None = None, rto_limit: float | None = None) -> DrillReport:
    """Back up, restore into a scratch area under ``workdir`` and prove the copy is identical.
    ``rto_limit`` (seconds) fails the drill when the restore is slower."""
    workdir = Path(workdir)
    t0 = time.time()
    m = backup(sources, workdir / "backups", sqlite=sqlite, label="drill")
    scratch = workdir / f"restore-{m.path.name}"
    t1 = time.perf_counter()
    restore(m.path, {name: scratch / (name if e["kind"] == "directory" else f"{name}.restored")
                     for name, e in m.sources.items()})
    rto = time.perf_counter() - t1
    result, ok = {}, True
    for name, e in m.sources.items():
        got = _tree(scratch / (name if e["kind"] == "directory" else f"{name}.restored"))
        want = e["files"]
        if e["kind"] != "directory":                     # single files are compared by content only
            got, want = {"file": next(iter(got.values()))}, {"file": next(iter(want.values()))}
        mismatched = sorted(k for k in want if k in got and got[k] != want[k])
        missing = sorted(k for k in want if k not in got)
        ok = ok and not mismatched and not missing
        result[name] = {"files": len(want), "mismatched": mismatched, "missing": missing}
    if rto_limit is not None and rto > rto_limit:
        ok = False
    return DrillReport(ok, round(rto, 3), round(time.time() - t0, 3), result, m.path)


def prune_backups(dest, *, keep_last: int = 7) -> list:
    """Delete all but the newest ``keep_last`` backups under ``dest``; returns what was removed."""
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    backups = sorted((p for p in Path(dest).glob("backup-*") if (p / "manifest.json").exists()), key=lambda p: p.name)
    removed = backups[:-keep_last]
    for p in removed:
        shutil.rmtree(p)
    return [str(p) for p in removed]
