"""REST API hardening: per-client rate limits and request validation against the model's schema."""

import io
import json

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.deploy.server import RateLimiter, validate_records


def call(app, path="/predict", body=None, key=None, addr="10.0.0.1", forwarded=None):
    data = json.dumps(body).encode() if body is not None else b""
    env = {"REQUEST_METHOD": "POST" if body is not None else "GET", "PATH_INFO": path,
           "CONTENT_LENGTH": str(len(data)), "CONTENT_TYPE": "application/json", "wsgi.input": io.BytesIO(data),
           "REMOTE_ADDR": addr}
    if key:
        env["HTTP_X_API_KEY"] = key
    if forwarded:
        env["HTTP_X_FORWARDED_FOR"] = forwarded
    out: dict = {}
    raw = b"".join(app(env, lambda s, h: out.update(status=int(s.split()[0]), headers=dict(h))))
    return out["status"], json.loads(raw), out["headers"]


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"age": rng.integers(18, 80, 300), "income": rng.normal(50, 10, 300),
                       "city": rng.choice(["a", "b"], 300).tolist()}).with_columns(
        y=((pl.col("income") + rng.normal(0, 8, 300)) > 50).cast(pl.Int8))
    return ac.AutoML("y", time_budget=5, include=["logistic_regression"], verbose=False).fit(df)


GOOD = {"records": [{"age": 30, "income": 55.0, "city": "a"}]}


def test_token_bucket():
    rl = RateLimiter("3/minute")
    assert [rl.check("c", now=0.0) for _ in range(3)] == [0.0, 0.0, 0.0]
    assert rl.check("c", now=0.0) == pytest.approx(20.0)        # 3 per minute -> one token every 20 s
    assert rl.check("other", now=0.0) == 0.0                    # buckets are per client
    assert rl.check("c", now=21.0) == 0.0                       # refilled
    with pytest.raises(ValueError):
        RateLimiter("lots/fortnight")


def test_rate_limit_per_client(model):
    app = ac.make_app(model, rate_limit="2/minute")
    assert [call(app, body=GOOD)[0] for _ in range(2)] == [200, 200]
    status, body, headers = call(app, body=GOOD)
    assert status == 429 and int(headers["Retry-After"]) >= 1 and "Rate limit" in body["error"]
    assert call(app, body=GOOD, addr="10.0.0.2")[0] == 200      # another client has its own bucket
    assert call(app, body=GOOD, key="k1")[0] == 200             # API keys are buckets too
    proxied = ac.make_app(model, rate_limit="1/minute", trust_proxy=True)
    assert call(proxied, body=GOOD, forwarded="1.1.1.1, 10.0.0.9")[0] == 200
    assert call(proxied, body=GOOD, forwarded="1.1.1.1")[0] == 429
    assert call(proxied, body=GOOD, forwarded="2.2.2.2")[0] == 200


def test_request_validation(model):
    bad = {"records": [{"age": "forty", "income": 55.0, "city": "a"}, {"income": 1.0, "city": "b"}]}
    assert validate_records(bad["records"], {"age": "Int64", "income": "Float64", "city": "String"},
                            ["age", "income", "city"]) == ["record 0: age='forty' is not a number",
                                                           "record 1: missing ['age']"]
    strict = ac.make_app(model, validate="strict")
    status, body, _ = call(strict, body=bad)
    assert status == 422 and "is not a number" in body["error"] and "missing ['age']" in body["error"]
    assert call(strict, body=GOOD)[0] == 200
    warn = ac.make_app(model)                                    # default: score, but say what was odd
    status, body, _ = call(warn, body={"records": [{"income": 50.0, "city": "a"}]})
    assert status == 200 and body["warnings"] == ["record 0: missing ['age']"]
    assert "warnings" not in call(warn, body=GOOD)[1]
    assert "warnings" not in call(ac.make_app(model, validate="off"), body={"records": [{"city": "a"}]})[1]
    with pytest.raises(ValueError):
        ac.make_app(model, validate="sometimes")
