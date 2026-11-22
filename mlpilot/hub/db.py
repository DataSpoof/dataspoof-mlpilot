"""Hub metadata database (SQLAlchemy 2.0): PostgreSQL in production, SQLite for development."""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
                        create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

ROLES = {"viewer": 1, "data_scientist": 2, "approver": 3, "admin": 4}
STAGES = ("dev", "qa", "staging", "production", "archived")


def utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


QUOTA_KEYS = ("max_workspaces", "max_model_versions", "max_storage_bytes", "max_runs_per_day")


class Organization(Base):
    """A tenant: owns workspaces and users; quotas bound what it may use."""
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    quotas: Mapped[dict] = mapped_column(JSON, default=dict)                   # keys from QUOTA_KEYS
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Usage(Base):
    """Metered usage per organisation and day (api_calls, runs, model_versions, storage_bytes)."""
    __tablename__ = "usage"
    __table_args__ = (UniqueConstraint("org_id", "day", "kind"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    day: Mapped[str] = mapped_column(String(10))                               # YYYY-MM-DD (UTC)
    kind: Mapped[str] = mapped_column(String(40))
    amount: Mapped[float] = mapped_column(Float, default=0.0)


class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    org_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)   # tenant (None: platform)
    require_approval: Mapped[bool] = mapped_column(Boolean, default=True)      # four-eyes for production
    # promotion gates checked server-side, e.g. {"min_metrics": {"test_roc_auc": 0.8}, "require_signed": true}
    policy: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    subject: Mapped[str] = mapped_column(String(255), unique=True)            # OIDC "sub" or "local:<name>"
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)            # platform admin
    is_service: Mapped[bool] = mapped_column(Boolean, default=False)          # service account
    org_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)   # tenant (None: any)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "workspace_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    role: Mapped[str] = mapped_column(String(20))
    user: Mapped[User] = relationship()


class ApiToken(Base):
    __tablename__ = "api_tokens"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(12))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)          # sha256; the token itself is never stored
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    user: Mapped[User] = relationship()


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    experiment: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="running")
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    datasets: Mapped[list] = mapped_column(JSON, default=list)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("workspace_id", "model_name", "version"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int] = mapped_column(Integer)
    stage: Mapped[str] = mapped_column(String(20), default="dev")
    task: Mapped[str | None] = mapped_column(String(30), nullable=True)
    target: Mapped[str | None] = mapped_column(String(200), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    artifact_uri: Mapped[str] = mapped_column(String(500))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    model_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signed: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PromotionRequest(Base):
    __tablename__ = "promotion_requests"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), index=True)
    to_stage: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="pending")      # pending | approved | rejected
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    requested_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    decided_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_version: Mapped[ModelVersion] = relationship()


class MonitorResult(Base):
    __tablename__ = "monitor_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checked_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    rows: Mapped[int] = mapped_column(Integer, default=0)
    drift_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    alerts: Mapped[list] = mapped_column(JSON, default=list)
    retrain_recommended: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditEntry(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(60), index=True)
    workspace: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)


def make_engine(url: str):
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk(conn, _):                      # enforce foreign keys in SQLite too
            conn.execute("PRAGMA foreign_keys=ON")
    return engine


_ADDED_COLUMNS = {   # columns added after 2.2: created in place on existing databases
    "workspaces": {"org_id": "VARCHAR(32)"},
    "users": {"org_id": "VARCHAR(32)"},
    "model_versions": {"artifact_bytes": "INTEGER"},
}


def _migrate(engine):
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for col, ddl in cols.items():
                if col not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))


def init_db(url: str):
    """Create tables (idempotent), add columns introduced since the database was created, and
    return a session factory."""
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    _migrate(engine)
    return sessionmaker(engine, expire_on_commit=False)
