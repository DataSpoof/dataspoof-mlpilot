"""Data observability and reconciliation: is today's data on time, complete and the same shape
as yesterday's, and did it arrive intact?

>>> mon = DataMonitor("orders", history="observability/orders.jsonl", timestamp="created_at",
...                   sla={"max_staleness_hours": 6, "min_rows": 10_000, "max_row_change": 0.3})
>>> rep = mon.observe(todays_orders)          # compares with the stored history, then appends to it
>>> rep.passed, rep.issues
>>> mon.history()                             # one row per observation: rows, freshness, issues

>>> rec = reconcile(source_df, warehouse_df, key="order_id")
>>> rec.passed, rec.summary()                 # missing / extra / duplicate keys, changed rows, sums

Five signals are tracked:
- **Freshness:** the age of the newest timestamp.
- **Volume:** the row count against the median of recent observations.
- **Schema:** columns added, removed or retyped.
- **Completeness:** null rates per column, as an absolute SLA and as a jump since last time.
- **Distribution:** the Population Stability Index of every numeric column against the previous
  observation, from stored decile bins.

The history holds statistics only, never rows.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_SLA = {"max_staleness_hours": None, "min_rows": None, "max_row_change": 0.5, "max_null_rate": None,
               "max_null_increase": 0.1, "schema": "additive", "max_psi": 0.25, "window": 7}


@dataclasses.dataclass
class ObservationIssue:
    check: str                # freshness | volume | schema | nulls | distribution
    severity: str             # error | warning
    message: str
    column: str | None = None

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}{f' · {self.column}' if self.column else ''}: {self.message}"


@dataclasses.dataclass
class ObservationReport:
    dataset: str
    observed_at: str
    rows: int
    issues: list
    snapshot: dict

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def summary(self) -> str:
        head = (f"{self.dataset} @ {self.observed_at}: {self.rows:,} rows - "
                f"{'OK' if self.passed else 'FAILED'} ({len(self.issues)} issue(s))")
        return "\n".join([head] + [f"  {i}" for i in self.issues])

    def __repr__(self) -> str:
        return self.summary()


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _snapshot(df: pl.DataFrame, timestamp: str | None, at: _dt.datetime) -> dict:
    n = df.height
    snap: dict = {"observed_at": at.isoformat(timespec="seconds"), "rows": n,
                  "schema": {c: str(t) for c, t in df.schema.items()},
                  "null_rate": {c: (df[c].null_count() / n if n else 0.0) for c in df.columns}, "bins": {}}
    for c, t in df.schema.items():
        if t.is_numeric() and c != timestamp:
            if t.is_integer() and n > 10 and df[c].n_unique() / n > 0.95:
                continue                              # identifiers: their "distribution" moves every batch
            x = df[c].drop_nulls().cast(pl.Float64).to_numpy()
            x = x[np.isfinite(x)]
            if x.size >= 20 and np.unique(x).size > 1:
                edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
                counts, _ = np.histogram(x, bins=edges)
                snap["bins"][c] = {"edges": edges.tolist(), "p": (counts / counts.sum()).tolist()}
    if timestamp and timestamp in df.columns and n:
        ts = df[timestamp]
        if ts.dtype == pl.String:
            ts = ts.str.to_datetime(strict=False, time_zone="UTC")
        latest = ts.max()
        if isinstance(latest, _dt.datetime):
            latest = latest if latest.tzinfo else latest.replace(tzinfo=_dt.timezone.utc)
            snap["latest"] = latest.isoformat(timespec="seconds")
            snap["staleness_hours"] = round((at - latest).total_seconds() / 3600, 3)
        elif isinstance(latest, _dt.date):
            latest = _dt.datetime(latest.year, latest.month, latest.day, tzinfo=_dt.timezone.utc)
            snap["latest"] = latest.isoformat(timespec="seconds")
            snap["staleness_hours"] = round((at - latest).total_seconds() / 3600, 3)
    return snap


def _psi_against(prev_bins: dict, x: np.ndarray) -> float | None:
    from ..monitoring.drift import psi
    edges = np.asarray(prev_bins["edges"])
    if x.size < 20 or edges.size < 2:
        return None
    counts, _ = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)
    if counts.sum() == 0:
        return None
    return psi(np.asarray(prev_bins["p"]), counts / counts.sum())


class DataMonitor:
    """Observe a dataset batch after batch and check it against SLAs and its own history.

    ``sla`` keys (all optional):
    - ``max_staleness_hours``: the newest ``timestamp`` may be at most this old.
    - ``min_rows``: the fewest rows a batch may have.
    - ``max_row_change``: relative change against the median of the last ``window`` batches
      (default 0.5).
    - ``max_null_rate``: one number, or ``{column: rate}``.
    - ``max_null_increase``: the largest jump since the last batch (default 0.1).
    - ``schema``:
      - ``"strict"``: any change is an error;
      - ``"additive"`` (the default): new columns are a warning, removed or retyped ones are
        errors;
      - ``None``: don't check the schema.
    - ``max_psi``: the distribution shift that counts as a warning (default 0.25)."""

    def __init__(self, name: str, *, history="data_monitor.jsonl", timestamp: str | None = None,
                 sla: dict | None = None):
        unknown = set(sla or {}) - set(DEFAULT_SLA)
        if unknown:
            raise ValueError(f"Unknown SLA keys {sorted(unknown)}; choose from {sorted(DEFAULT_SLA)}")
        self.name, self.path, self.timestamp = name, Path(history), timestamp
        self.sla = {**DEFAULT_SLA, **(sla or {})}

    def _history(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def observe(self, df, *, at: _dt.datetime | None = None, record: bool = True) -> ObservationReport:
        """Check one batch (a DataFrame, or anything ``mlpilot.read`` accepts) and append its
        statistics to the history (``record=False`` checks without recording)."""
        if not isinstance(df, pl.DataFrame):
            from ..ingestion import read
            df = read(df)
            df = df.collect() if isinstance(df, pl.LazyFrame) else df
        at = at or _now()
        at = at if at.tzinfo else at.replace(tzinfo=_dt.timezone.utc)
        sla, hist = self.sla, self._history()
        snap = _snapshot(df, self.timestamp, at)
        issues: list = []
        if sla["max_staleness_hours"] is not None:
            if "staleness_hours" not in snap:
                issues.append(ObservationIssue("freshness", "error", f"no usable timestamps in {self.timestamp!r}"))
            elif snap["staleness_hours"] > sla["max_staleness_hours"]:
                issues.append(ObservationIssue("freshness", "error", f"newest row is {snap['staleness_hours']:.1f} h old "
                                                                     f"(SLA {sla['max_staleness_hours']} h)"))
        if sla["min_rows"] is not None and df.height < sla["min_rows"]:
            issues.append(ObservationIssue("volume", "error", f"{df.height:,} rows (SLA at least {sla['min_rows']:,})"))
        recent = [h["rows"] for h in hist[-int(sla["window"]):]]
        if recent and sla["max_row_change"] is not None:
            median = float(np.median(recent))
            if median > 0:
                change = (df.height - median) / median
                if abs(change) > sla["max_row_change"]:
                    issues.append(ObservationIssue("volume", "error" if change < 0 else "warning",
                                                   f"{df.height:,} rows vs a recent median of {median:,.0f} "
                                                   f"({change:+.0%})"))
        prev = hist[-1] if hist else None
        if prev and sla["schema"]:
            old, new = prev["schema"], snap["schema"]
            for c in sorted(set(old) - set(new)):
                issues.append(ObservationIssue("schema", "error", "column removed", c))
            for c in sorted(set(new) - set(old)):
                issues.append(ObservationIssue("schema", "error" if sla["schema"] == "strict" else "warning",
                                               f"new column ({new[c]})", c))
            for c in sorted(set(old) & set(new)):
                if old[c] != new[c]:
                    issues.append(ObservationIssue("schema", "error", f"type changed {old[c]} -> {new[c]}", c))
        limit = sla["max_null_rate"]
        for c, rate in snap["null_rate"].items():
            cap = limit.get(c) if isinstance(limit, dict) else limit
            if cap is not None and rate > cap:
                issues.append(ObservationIssue("nulls", "error", f"{rate:.1%} missing (SLA {cap:.1%})", c))
            elif prev and sla["max_null_increase"] is not None and c in prev["null_rate"] \
                    and rate - prev["null_rate"][c] > sla["max_null_increase"]:
                issues.append(ObservationIssue("nulls", "warning",
                                               f"missing rose from {prev['null_rate'][c]:.1%} to {rate:.1%}", c))
        if prev and sla["max_psi"] is not None:
            for c, b in prev.get("bins", {}).items():
                if c in df.columns and df.schema[c].is_numeric():
                    x = df[c].drop_nulls().cast(pl.Float64).to_numpy()
                    v = _psi_against(b, x[np.isfinite(x)])
                    if v is not None and v > sla["max_psi"]:
                        issues.append(ObservationIssue("distribution", "warning", f"PSI {v:.2f} vs the last batch", c))
        snap["issues"] = [str(i) for i in issues]
        snap["passed"] = not any(i.severity == "error" for i in issues)
        if record:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(snap) + "\n")
        return ObservationReport(self.name, snap["observed_at"], df.height, issues, snap)

    def history(self) -> pl.DataFrame:
        rows = [{"observed_at": h["observed_at"], "rows": h["rows"], "columns": len(h["schema"]),
                 "staleness_hours": h.get("staleness_hours"), "passed": h.get("passed"), "issues": len(h.get("issues", []))}
                for h in self._history()]
        return pl.DataFrame(rows, schema={"observed_at": pl.Utf8, "rows": pl.Int64, "columns": pl.Int64,
                                          "staleness_hours": pl.Float64, "passed": pl.Boolean, "issues": pl.Int64})


# ── reconciliation ──────────────────────────────────────────────────────────
@dataclasses.dataclass
class ReconciliationReport:
    source_rows: int
    destination_rows: int
    key: str | None
    missing_keys: list            # in source, not in destination (sample)
    extra_keys: list              # in destination, not in source (sample)
    n_missing: int
    n_extra: int
    duplicate_keys: dict          # {"source": n, "destination": n}
    changed_rows: int             # same key, different values
    changed_sample: list
    columns: pl.DataFrame         # per column: nulls and (numeric) sums on both sides, match flag
    schema_differences: list

    @property
    def passed(self) -> bool:
        columns_match = bool(self.columns["match"].all()) if self.columns.height else True
        return (self.source_rows == self.destination_rows and not self.n_missing and not self.n_extra
                and not self.changed_rows and not any(self.duplicate_keys.values()) and not self.schema_differences
                and columns_match)

    def summary(self) -> str:
        lines = [f"Reconciliation: {'PASSED' if self.passed else 'FAILED'}  "
                 f"source {self.source_rows:,} rows, destination {self.destination_rows:,} rows"]
        if self.key:
            lines.append(f"  key {self.key!r}: {self.n_missing:,} missing in destination, {self.n_extra:,} extra, "
                         f"duplicates {self.duplicate_keys}, {self.changed_rows:,} changed rows")
        for d in self.schema_differences:
            lines.append(f"  schema: {d}")
        bad = self.columns.filter(~pl.col("match"))
        for r in bad.iter_rows(named=True):
            lines.append(f"  column {r['column']}: source nulls {r['source_nulls']}, destination nulls "
                         f"{r['destination_nulls']}, sums {r['source_sum']} vs {r['destination_sum']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def reconcile(source: pl.DataFrame, destination: pl.DataFrame, *, key: str | list | None = None, columns=None,
              rel_tolerance: float = 1e-9, sample: int = 20) -> ReconciliationReport:
    """Check that ``destination`` holds exactly what ``source`` had (after a copy, an ETL job or
    a migration): row counts, keys, per-column null counts and numeric sums, schema, and, with a
    ``key``, row-by-row differences."""
    cols = list(columns) if columns else [c for c in source.columns if c in destination.columns]
    schema_diff = [f"{c}: only in source" for c in source.columns if c not in destination.columns] + \
                  [f"{c}: only in destination" for c in destination.columns if c not in source.columns] + \
                  [f"{c}: {source.schema[c]} vs {destination.schema[c]}" for c in cols
                   if source.schema[c] != destination.schema[c]]
    rows = []
    for c in cols:
        s, d = source[c], destination[c]
        rec = {"column": c, "source_nulls": s.null_count(), "destination_nulls": d.null_count(),
               "source_sum": None, "destination_sum": None}
        match = rec["source_nulls"] == rec["destination_nulls"]
        if s.dtype.is_numeric() and d.dtype.is_numeric():
            a, b = float(s.cast(pl.Float64).sum() or 0.0), float(d.cast(pl.Float64).sum() or 0.0)
            rec["source_sum"], rec["destination_sum"] = a, b
            match = match and abs(a - b) <= rel_tolerance * max(1.0, abs(a), abs(b))
        rec["match"] = bool(match)
        rows.append(rec)
    col_frame = pl.DataFrame(rows, schema={"column": pl.Utf8, "source_nulls": pl.Int64, "destination_nulls": pl.Int64,
                                           "source_sum": pl.Float64, "destination_sum": pl.Float64, "match": pl.Boolean})
    missing = extra = changed = []
    n_missing = n_extra = n_changed = 0
    dups = {"source": 0, "destination": 0}
    key_cols = [key] if isinstance(key, str) else list(key or [])
    if key_cols:
        dups = {"source": source.height - source.select(key_cols).n_unique(),           # extra rows per side
                "destination": destination.height - destination.select(key_cols).n_unique()}
        src_keys, dst_keys = source.select(key_cols).unique(), destination.select(key_cols).unique()
        miss = src_keys.join(dst_keys, on=key_cols, how="anti")
        ext = dst_keys.join(src_keys, on=key_cols, how="anti")
        n_missing, n_extra = miss.height, ext.height
        missing, extra = miss.head(sample).to_dicts(), ext.head(sample).to_dicts()
        value_cols = [c for c in cols if c not in key_cols and source.schema[c] == destination.schema[c]]
        if value_cols:
            sh = source.select(key_cols).with_columns(_h=source.select(value_cols).hash_rows(seed=0))
            dh = destination.select(key_cols).with_columns(_h=destination.select(value_cols).hash_rows(seed=0))
            both = sh.unique(key_cols).join(dh.unique(key_cols), on=key_cols, suffix="_d")
            diff = both.filter(pl.col("_h") != pl.col("_h_d"))
            n_changed, changed = diff.height, diff.select(key_cols).head(sample).to_dicts()
    return ReconciliationReport(source.height, destination.height, ",".join(key_cols) or None, missing, extra,
                                n_missing, n_extra, dups, n_changed, changed, col_frame, schema_diff)
