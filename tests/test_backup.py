"""Backup, verified restore, DR drills and retention."""

import io
import sqlite3
import tarfile
import time

import pytest

from mlpilot.backup import backup, dr_drill, prune_backups, restore, verify_backup


def _state(tmp_path):
    reg = tmp_path / "registry" / "churn" / "v1"
    reg.mkdir(parents=True)
    (reg / "manifest.json").write_text('{"version": 1}')
    (reg / "model.joblib").write_bytes(b"\x00\x01binary-model" * 100)
    (tmp_path / "monitor.json").write_text('{"baseline": [1, 2, 3]}')
    db = tmp_path / "hub.db"
    con = sqlite3.connect(db)
    con.execute("create table audit (id integer primary key, action text)")
    con.executemany("insert into audit (action) values (?)", [("promote",), ("approve",)])
    con.commit()
    return db, con                       # left open: backups must work while the Hub is running


def test_backup_verify_and_restore(tmp_path):
    db, con = _state(tmp_path)
    m = backup({"registry": tmp_path / "registry", "monitor": tmp_path / "monitor.json"}, tmp_path / "backups",
               sqlite={"hub": db})
    con.close()
    assert set(m.sources) == {"registry", "monitor", "hub"} and m.sources["registry"]["files"]
    assert verify_backup(m.path) == {"registry": "ok", "monitor": "ok", "hub": "ok"}
    out = tmp_path / "restored"
    counts = restore(m.path, {"registry": out / "registry", "hub": out / "hub.db", "monitor": out / "monitor.json"})
    assert counts["registry"] == 2
    assert (out / "registry" / "churn" / "v1" / "model.joblib").read_bytes() == \
        (tmp_path / "registry" / "churn" / "v1" / "model.joblib").read_bytes()
    rows = sqlite3.connect(out / "hub.db").execute("select action from audit order by id").fetchall()
    assert rows == [("promote",), ("approve",)]
    with pytest.raises(FileExistsError):
        restore(m.path, {"registry": out / "registry"})
    restore(m.path, {"registry": out / "registry"}, overwrite=True)


def test_corrupt_backups_are_never_restored(tmp_path):
    _state(tmp_path)[1].close()
    m = backup({"registry": tmp_path / "registry"}, tmp_path / "backups")
    archive = m.path / "registry.tar.gz"
    archive.write_bytes(archive.read_bytes()[:-20] + b"x" * 20)
    assert verify_backup(m.path) == {"registry": "corrupt"}
    with pytest.raises(ValueError, match="corrupt"):
        restore(m.path, {"registry": tmp_path / "out"})
    with pytest.raises(KeyError):
        restore(m.path, {"nope": tmp_path / "out"})


def test_path_traversal_archives_are_refused(tmp_path):
    _state(tmp_path)[1].close()
    m = backup({"registry": tmp_path / "registry"}, tmp_path / "backups")
    evil = m.path / "registry.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        data = b"pwned"
        info = tarfile.TarInfo("../../escaped.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    import json
    manifest = json.loads((m.path / "manifest.json").read_text())
    import hashlib
    manifest["sources"]["registry"]["sha256"] = hashlib.sha256(evil.read_bytes()).hexdigest()
    (m.path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        restore(m.path, {"registry": tmp_path / "out" / "registry"})
    assert not (tmp_path / "escaped.txt").exists()


def test_dr_drill_and_retention(tmp_path):
    db, con = _state(tmp_path)
    rep = dr_drill({"registry": tmp_path / "registry", "monitor": tmp_path / "monitor.json"}, tmp_path / "dr",
                   sqlite={"hub": db})
    con.close()
    assert rep.passed, rep.summary()
    assert rep.rto_seconds >= 0 and rep.rpo_seconds >= rep.rto_seconds
    assert rep.sources["registry"]["files"] == 2 and "PASSED" in rep.summary()
    assert not dr_drill({"registry": tmp_path / "registry"}, tmp_path / "dr2", rto_limit=0.0).passed
    for _ in range(3):
        backup({"monitor": tmp_path / "monitor.json"}, tmp_path / "many")
        time.sleep(0.01)
    removed = prune_backups(tmp_path / "many", keep_last=1)
    assert len(removed) == 2 and len(list((tmp_path / "many").glob("backup-*"))) == 1
