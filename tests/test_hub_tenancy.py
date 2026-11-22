"""Hub multi-tenancy: organisations, cross-tenant isolation, quotas, usage metering and migration."""

import sqlite3
import time

import polars as pl
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("fastapi")
pytest.importorskip("jwt")
from fastapi.testclient import TestClient  # noqa: E402

import mlpilot as ac  # noqa: E402
from mlpilot.hub.app import bootstrap_admin, create_app  # noqa: E402
from mlpilot.hub.storage import tar_directory  # noqa: E402


def H(tok):
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture(scope="module")
def model():
    X, y = make_classification(200, 4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y))
    return ac.AutoML("y", time_budget=5, verbose=False, max_trials=1, include=["logistic_regression"]).fit(df)


@pytest.fixture()
def hub(tmp_path):
    url = f"sqlite:///{tmp_path / 'hub.db'}"
    root = bootstrap_admin(url, name="root", email="root@platform.com")
    client = TestClient(create_app(database_url=url, artifact_root=str(tmp_path / "art")))
    for org, quotas in (("acme", {"max_workspaces": 1, "max_model_versions": 2, "max_runs_per_day": 2}),
                        ("globex", {})):
        assert client.post("/api/orgs", json={"name": org, "quotas": quotas}, headers=H(root)).status_code == 201
    users = {}
    for name in ("ana", "gus"):
        sa = client.post("/api/service-accounts", json={"name": name}, headers=H(root)).json()
        users[name] = sa
    assert client.post("/api/orgs/acme/users", json={"user": users["ana"]["id"]}, headers=H(root)).status_code == 200
    assert client.post("/api/orgs/globex/users", json={"user": users["gus"]["id"]}, headers=H(root)).status_code == 200
    assert client.post("/api/workspaces", json={"name": "acme-ds", "org": "acme"}, headers=H(root)).status_code == 201
    assert client.post("/api/workspaces", json={"name": "globex-ds", "org": "globex"}, headers=H(root)).status_code == 201
    return client, root, users, url


def test_cross_tenant_membership_is_refused(hub):
    client, root, users, _ = hub
    ok = client.put("/api/workspaces/acme-ds/members", json={"user": users["ana"]["id"], "role": "data_scientist"},
                    headers=H(root))
    assert ok.status_code == 200
    bad = client.put("/api/workspaces/acme-ds/members", json={"user": users["gus"]["id"], "role": "viewer"},
                     headers=H(root))
    assert bad.status_code == 403 and "Cross-tenant" in bad.json()["detail"]
    # gus can't see or touch acme's workspace
    assert client.get("/api/workspaces/acme-ds/runs", headers=H(users["gus"]["token"])).status_code == 404  # hidden
    assert [w["name"] for w in client.get("/api/workspaces", headers=H(users["gus"]["token"])).json()] == []
    # an org user can't be moved to another org while they belong to one
    r = client.post("/api/orgs/globex/users", json={"user": users["ana"]["id"]}, headers=H(root))
    assert r.status_code == 409
    # non-admins only see their own organisation
    assert [o["name"] for o in client.get("/api/orgs", headers=H(users["ana"]["token"])).json()] == ["acme"]
    assert client.get("/api/orgs/globex/usage", headers=H(users["ana"]["token"])).status_code == 403


def test_quotas_and_usage_metering(hub, model, tmp_path):
    client, root, users, _ = hub
    client.put("/api/workspaces/acme-ds/members", json={"user": users["ana"]["id"], "role": "data_scientist"},
               headers=H(root))
    tok = users["ana"]["token"]
    assert client.post("/api/workspaces", json={"name": "acme-2", "org": "acme"}, headers=H(root)).status_code == 429
    for _ in range(2):
        assert client.post("/api/workspaces/acme-ds/runs", json={"experiment": "e", "name": "r"},
                           headers=H(tok)).status_code == 201
    over = client.post("/api/workspaces/acme-ds/runs", json={"experiment": "e", "name": "r"}, headers=H(tok))
    assert over.status_code == 429 and "max_runs_per_day" in over.json()["detail"]

    def upload():
        blob = tar_directory(model.save(tmp_path / f"m{time.time_ns()}"))
        return client.post("/api/workspaces/acme-ds/models/churn/versions",
                           files={"artifact": ("m.tar.gz", blob, "application/gzip")}, headers=H(tok))
    assert upload().status_code == 201 and upload().status_code == 201
    assert upload().status_code == 429                                   # max_model_versions = 2
    usage = client.get("/api/orgs/acme/usage", headers=H(tok)).json()
    assert usage["totals"]["runs"] == 2 and usage["totals"]["model_versions"] == 2
    # only successful calls are metered: rejected (429) requests roll back with their transaction
    assert usage["totals"]["storage_bytes"] > 0 and usage["totals"]["api_calls"] == 5
    assert usage["in_use"]["max_model_versions"] == 2 and usage["in_use"]["max_workspaces"] == 1
    # raising the quota lets work continue; bad quota keys are rejected
    assert client.put("/api/orgs/acme/quotas", json={"quotas": {"max_model_versions": 10}},
                      headers=H(root)).status_code == 200
    assert upload().status_code == 201
    assert client.put("/api/orgs/acme/quotas", json={"quotas": {"max_gpus": 4}}, headers=H(root)).status_code == 422
    assert client.put("/api/orgs/acme/quotas", json={"quotas": {}}, headers=H(tok)).status_code == 403


def test_existing_databases_are_migrated(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)                               # a 2.2-era schema without the tenancy columns
    con.execute("create table workspaces (id varchar(32) primary key, name varchar(100) unique, "
                "require_approval boolean, policy json, created_at datetime)")
    con.execute("insert into workspaces values ('w1', 'legacy', 1, '{}', '2026-01-01')")
    con.commit()
    con.close()
    from mlpilot.hub.db import init_db
    init_db(f"sqlite:///{db}")
    cols = {r[1] for r in sqlite3.connect(db).execute("pragma table_info(workspaces)")}
    assert "org_id" in cols
    assert sqlite3.connect(db).execute("select name, org_id from workspaces").fetchall() == [("legacy", None)]
