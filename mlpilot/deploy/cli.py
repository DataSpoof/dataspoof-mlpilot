"""Command line: ``mlpilot serve | score | docker | export | monitor | hub`` and the
model CI/CD steps ``train | gate | register | promote | release | latest | deploy-sagemaker | cicd init``, and
the autonomous engine ``plan | run | approve | status``.

Also available as ``python -m mlpilot.deploy ...``.
"""

from __future__ import annotations

import argparse
import json
import sys


def _model(args):
    if getattr(args, "registry", None):
        from ..registry import ModelRegistry
        name, _, ver = args.model.partition(":")
        reg = ModelRegistry(args.registry)
        if ver.isdigit():
            return reg.get(name, version=int(ver))
        return reg.get(name, stage=ver or None)
    return None


def _model_path(args):
    mv = _model(args)
    return str(mv.model_path) if mv is not None else args.model


def container_command() -> list:
    """The gunicorn command line for the current environment (Vertex sets AIP_HTTP_PORT)."""
    import os
    port = os.environ.get("AIP_HTTP_PORT") or os.environ.get("AC_PORT") or os.environ.get("PORT") or "8080"
    return ["gunicorn", "--bind", f"0.0.0.0:{port}", "--workers", os.environ.get("AC_WORKERS", "2"),
            "--timeout", "120", "--access-logfile", "-", "mlpilot.deploy.wsgi:app"]


def _container():
    import os
    cmd = container_command()
    os.execvp(cmd[0], cmd)              # replaces this process; gunicorn handles signals


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlpilot", description="Serve, score and package models.")
    p.add_argument("--registry", help="registry directory; then --model is NAME, NAME:VERSION or NAME:STAGE")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serve a model over REST (development server)")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-rows", type=int, default=10_000)
    s.add_argument("--prediction-log")

    b = sub.add_parser("score", help="batch-score a CSV/Parquet/NDJSON file")
    b.add_argument("--model", required=True)
    b.add_argument("--input", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--chunk-size", type=int, default=100_000)
    b.add_argument("--keep", default="all", help="'all', 'none', or comma-separated id columns")

    d = sub.add_parser("docker", help="write a Docker build context for the model")
    d.add_argument("--model", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--name", default="model")
    d.add_argument("--version", default="1")
    d.add_argument("--port", type=int, default=None)
    d.add_argument("--target", default="generic", choices=["generic", "sagemaker", "vertex", "azureml"])

    c = sub.add_parser("container", help="run the production server inside a container (gunicorn)")
    c.add_argument("extra", nargs="*", help="ignored (SageMaker passes 'serve')")

    e = sub.add_parser("export", help="export a model (bundle, pickle or onnx)")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--format", default="bundle", choices=["bundle", "pickle", "onnx", "skops"])
    e.add_argument("--quantize", choices=["int8", "uint8", "fp16"], help="with --format onnx: 8/16-bit weights")

    q = sub.add_parser("quantize", help="quantize a model to int8 / fp16 ONNX and compare on held-out data")
    q.add_argument("--model", required=True)
    q.add_argument("--data", required=True, help="held-out rows with the target (CSV / Parquet / ...)")
    q.add_argument("--out", required=True)
    q.add_argument("--modes", default="int8,fp16")
    q.add_argument("--metric")
    q.add_argument("--max-drop", type=float, default=0.01)

    m = sub.add_parser("monitor", help="check a batch of data against a saved monitor")
    m.add_argument("--model", required=True)
    m.add_argument("--monitor", required=True, help="monitor JSON saved with Monitor.save()")
    m.add_argument("--input", required=True)

    h = sub.add_parser("hub", help="run / administer the MLPilot Hub server")
    hsub = h.add_subparsers(dest="hub_cmd", required=True)
    hs = hsub.add_parser("serve", help="start the Hub (uvicorn)")
    hs.add_argument("--host", default="127.0.0.1")
    hs.add_argument("--port", type=int, default=8080)
    hs.add_argument("--workers", type=int, default=1)
    ha = hsub.add_parser("create-admin", help="create a platform admin and print an API token (once)")
    ha.add_argument("--name", default="admin")
    ha.add_argument("--email")

    _pipeline_parsers(sub)
    _engine_parsers(sub)
    args = p.parse_args(argv)
    if args.cmd in _PIPELINE:
        return _run_pipeline(args)
    if args.cmd in _ENGINE:
        return _run_engine(args)
    if args.cmd == "container":
        return _container()
    if args.cmd == "hub":
        if args.hub_cmd == "create-admin":
            from ..hub.app import bootstrap_admin
            print(bootstrap_admin(name=args.name, email=args.email))
            return 0
        import uvicorn
        uvicorn.run("mlpilot.hub.app:app", host=args.host, port=args.port, workers=args.workers,
                    proxy_headers=True)
        return 0
    path = _model_path(args)
    if args.cmd == "serve":
        from .server import serve
        serve(path, host=args.host, port=args.port, max_rows=args.max_rows, prediction_log=args.prediction_log)
    elif args.cmd == "score":
        from .scoring import batch_score
        keep = "all" if args.keep == "all" else (None if args.keep == "none" else args.keep.split(","))
        print(json.dumps(batch_score(path, args.input, args.output, chunk_size=args.chunk_size, keep=keep),
                         indent=2, default=str))
    elif args.cmd == "docker":
        from .export import build_docker
        out = build_docker(path, args.out, name=args.name, version=args.version, port=args.port, target=args.target)
        print(f"Docker context written to {out}. Build it with: docker build -t {args.name} {out}")
    elif args.cmd == "export":
        from .export import export_model
        print(export_model(path, args.out, format=args.format, quantize=args.quantize))
    elif args.cmd == "quantize":
        from ..ingestion import read
        from .quantize import quantize_model
        rep = quantize_model(path, read(args.data), modes=[m.strip() for m in args.modes.split(",") if m.strip()],
                             out=args.out, metric=args.metric, max_drop=args.max_drop)
        print(rep.summary())
    elif args.cmd == "monitor":
        import polars as pl

        from ..monitoring import Monitor
        from .scoring import _batches, resolve_model
        from pathlib import Path
        monitor = Monitor.load(args.monitor, resolve_model(path))
        batch = pl.concat(list(_batches(Path(args.input), 1_000_000)), how="vertical_relaxed")
        report = monitor.check(batch)
        print(report.summary())
        return 1 if report.alerts else 0
    return 0


# ── model CI/CD commands (see mlpilot.cicd) ───────────────────────
_PIPELINE = ("train", "gate", "register", "promote", "release", "latest", "deploy-sagemaker", "cicd")
_ENGINE = ("plan", "run", "approve", "status", "scan-model", "sbom", "backup", "restore", "dr-drill")


def _engine_parsers(sub):
    def goal_args(p):
        p.add_argument("--goal", help="goal file (YAML / JSON)")
        p.add_argument("--text", help='the goal in words, e.g. "predict churned with AUC above 0.85"')
        p.add_argument("--data", help="data source (overrides the goal's)")
        p.add_argument("--name", help="model name (overrides the goal's)")
        p.add_argument("--no-llm", action="store_true", help="draft goals from --text with rules only")

    p = sub.add_parser("plan", help="show the goal and the workflow the engine would run")
    goal_args(p)
    p.add_argument("--save", help="write the (drafted) goal to this YAML file")
    r = sub.add_parser("run", help="run a goal end to end (resumable; stops at gates and approvals)")
    goal_args(r)
    r.add_argument("--out", required=True, help="run directory")
    r.add_argument("--resume", action="store_true", help="continue the run in --out")
    r.add_argument("--by", help="who is running it (default: the OS user)")
    a = sub.add_parser("approve", help="approve a step that is waiting (four-eyes: not the requester)")
    a.add_argument("run_dir")
    a.add_argument("--step", default="promote")
    a.add_argument("--by", required=True)
    a.add_argument("--note", default="")
    st = sub.add_parser("status", help="show a run's steps, costs and what it is waiting for")
    st.add_argument("run_dir")
    sc = sub.add_parser("scan-model", help="scan model files for code that would run on load (exit 1 if critical)")
    sc.add_argument("path")
    sb = sub.add_parser("sbom", help="write a CycloneDX SBOM of the installed dependencies")
    sb.add_argument("--out", default="sbom.cdx.json")
    sb.add_argument("--all", action="store_true", help="every installed package, not just MLPilot's dependencies")
    def pairs(p, flag, help_):
        p.add_argument(flag, action="append", default=[], metavar="NAME=PATH", help=help_)

    bk = sub.add_parser("backup", help="checksummed backup of registries, runs, monitors and SQLite databases")
    pairs(bk, "--source", "directory or file to back up (repeat)")
    pairs(bk, "--sqlite", "SQLite database to snapshot consistently, e.g. hub=hub.db (repeat)")
    bk.add_argument("--out", required=True)
    bk.add_argument("--keep-last", type=int, help="then delete all but the newest N backups")
    rs = sub.add_parser("restore", help="verify and restore sources from a backup")
    rs.add_argument("backup_path")
    pairs(rs, "--target", "where to restore a source, e.g. registry=/srv/registry (repeat)")
    rs.add_argument("--overwrite", action="store_true")
    dr = sub.add_parser("dr-drill", help="backup -> restore -> compare; reports RTO and RPO (exit 1 on failure)")
    pairs(dr, "--source", "directory or file (repeat)")
    pairs(dr, "--sqlite", "SQLite database (repeat)")
    dr.add_argument("--workdir", required=True)
    dr.add_argument("--rto-limit", type=float)


def _load_goal(args):
    from ..orchestration import Goal, goal_from_text
    overrides = {k: v for k, v in (("data", args.data), ("name", args.name)) if v}
    if args.goal:
        d = Goal.from_file(args.goal).to_dict()
        d.update(overrides)
        return Goal.from_dict(d)
    if args.text:
        return goal_from_text(args.text, use_llm=False if args.no_llm else "auto", **overrides)
    raise SystemExit("give --goal FILE or --text \"...\"")


def _run_engine(args) -> int:
    from .. import orchestration as orc
    if args.cmd == "plan":
        goal = _load_goal(args)
        print(goal.to_yaml())
        print(orc.plan(goal).describe())
        if args.save:
            goal.to_yaml(args.save)
        return 0
    if args.cmd == "run":
        res = orc.resume(args.out, requested_by=args.by) if args.resume else orc.run(_load_goal(args), args.out,
                                                                                     requested_by=args.by)
        print(res.summary())
        return {"succeeded": 0, "waiting_approval": 3}.get(res.status, 1)
    if args.cmd == "approve":
        entry = orc.approve(args.run_dir, args.step, by=args.by, note=args.note)
        print(f"approved {entry['step']} by {entry['by']}; continue with: mlpilot run --out {args.run_dir} --resume")
        return 0
    if args.cmd == "scan-model":
        from ..supply_chain import scan_model
        rep = scan_model(args.path)
        print(rep.summary())
        return 0 if rep.passed else 1
    if args.cmd in ("backup", "restore", "dr-drill"):
        from .. import backup as bk

        def kv(items):
            out = {}
            for it in items:
                if "=" not in it:
                    raise SystemExit(f"expected NAME=PATH, got {it!r}")
                k, v = it.split("=", 1)
                out[k] = v
            return out
        if args.cmd == "backup":
            m = bk.backup(kv(args.source), args.out, sqlite=kv(args.sqlite))
            print(f"backup written to {m.path}: {', '.join(m.sources)}")
            if args.keep_last:
                for p in bk.prune_backups(args.out, keep_last=args.keep_last):
                    print(f"removed old backup {p}")
            return 0
        if args.cmd == "restore":
            print(json.dumps(bk.restore(args.backup_path, kv(args.target), overwrite=args.overwrite)))
            return 0
        drill = bk.dr_drill(kv(args.source), args.workdir, sqlite=kv(args.sqlite), rto_limit=args.rto_limit)
        print(drill.summary())
        return 0 if drill.passed else 1
    if args.cmd == "sbom":
        from ..supply_chain import generate_sbom
        sbom = generate_sbom(args.out, all_packages=args.all)
        print(f"{args.out}: {len(sbom['components'])} components")
        return 0
    print(orc.status(args.run_dir).summary())
    return 0


def _pipeline_parsers(sub):
    S = argparse.SUPPRESS                    # don't clobber the top-level --registry
    t = sub.add_parser("train", help="split, check, fit AutoML, evaluate and save a model (CI step)")
    t.add_argument("--data", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--out", required=True, help="model directory to write")
    t.add_argument("--test", help="separate test file (default: split --data)")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--group", help="split by this column (no group on both sides)")
    t.add_argument("--time", help="split chronologically by this column")
    t.add_argument("--time-budget", type=float, default=300)
    t.add_argument("--include", help="comma-separated model families")
    t.add_argument("--max-suspicious-share", type=float, help="fail if more training rows look poisoned")
    t.add_argument("--report")

    g = sub.add_parser("gate", help="fail (exit 1) unless the model meets gates.yml on held-out data")
    g.add_argument("--model", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--gates", required=True)
    g.add_argument("--report")

    r = sub.add_parser("register", help="add a model directory to a registry")
    r.add_argument("--model", required=True)
    r.add_argument("--registry", default=S, required=False)
    r.add_argument("--name", required=True)
    r.add_argument("--metrics", help="metrics JSON (e.g. the metrics.json `train` writes)")
    r.add_argument("--stage", default="dev")
    r.add_argument("--description")
    r.add_argument("--sign-key", help="Ed25519 private key to sign the stored model")
    r.add_argument("--report")

    pr = sub.add_parser("promote", help="move a registered version to a stage")
    pr.add_argument("--registry", default=S)
    pr.add_argument("--name", required=True)
    pr.add_argument("--version", type=int, required=True)
    pr.add_argument("--stage", required=True)
    pr.add_argument("--reason")

    rel = sub.add_parser("release", help="release a version to the next environment (environments.yml)")
    rel.add_argument("--config", required=True)
    rel.add_argument("--name", required=True)
    rel.add_argument("--version", type=int, required=True)
    rel.add_argument("--from", dest="frm", required=True)
    rel.add_argument("--to", required=True)
    rel.add_argument("--released-by")
    rel.add_argument("--approved-by", default="", help="comma-separated approvers")
    rel.add_argument("--reason")
    rel.add_argument("--report")

    lt = sub.add_parser("latest", help="print the newest version of a model (in a stage)")
    lt.add_argument("--registry", default=S)
    lt.add_argument("--name", required=True)
    lt.add_argument("--stage")

    ds = sub.add_parser("deploy-sagemaker", help="push an image to ECR and blue/green-deploy a SageMaker endpoint")
    ds.add_argument("--image", required=True, help="local image (pushed to --repository) or an ECR URI")
    ds.add_argument("--repository", help="ECR repository to push a local image to")
    ds.add_argument("--endpoint", required=True)
    ds.add_argument("--role-arn", required=True)
    ds.add_argument("--instance-type", default="ml.m5.large")
    ds.add_argument("--instance-count", type=int, default=1)
    ds.add_argument("--serverless-memory-mb", type=int)
    ds.add_argument("--rollback-alarms", default="", help="comma-separated CloudWatch alarm names")

    ci = sub.add_parser("cicd", help="generate a model CI/CD pipeline")
    cisub = ci.add_subparsers(dest="cicd_cmd", required=True)
    ini = cisub.add_parser("init", help="write a GitHub Actions / GitLab CI pipeline, gates.yml, environments.yml")
    ini.add_argument("--provider", choices=["github", "gitlab"], default="github")
    ini.add_argument("--name", required=True)
    ini.add_argument("--data", required=True)
    ini.add_argument("--target", required=True)
    ini.add_argument("--out", default=".")
    ini.add_argument("--metric", default="roc_auc")
    ini.add_argument("--floor", type=float, default=0.7)
    ini.add_argument("--time-budget", type=int, default=300)
    ini.add_argument("--overwrite", action="store_true")


def _run_pipeline(args) -> int:
    from .. import cicd
    if args.cmd == "train":
        res = cicd.train(args.data, args.target, args.out, test=args.test, test_size=args.test_size, group=args.group,
                         time=args.time, time_budget=args.time_budget,
                         include=args.include.split(",") if args.include else None, report=args.report,
                         max_suspicious_share=args.max_suspicious_share)
        print(json.dumps({k: res[k] for k in ("model", "algorithm", "metrics", "split", "split_issues",
                                                 "suspicious_share")}, indent=2, default=str))
        return 0
    if args.cmd == "gate":
        res = cicd.gate(args.model, args.data, cicd.load_gates(args.gates), report=args.report)
        print("GATES PASSED" if res["passed"] else "GATES FAILED:\n  - " + "\n  - ".join(res["failures"]))
        return 0 if res["passed"] else 1
    if args.cmd == "register":
        if not getattr(args, "registry", None):
            raise SystemExit("register needs --registry")
        res = cicd.register(args.model, args.registry, args.name, metrics=args.metrics, stage=args.stage,
                            description=args.description, sign_key=args.sign_key)
        cicd._write_json(args.report, res)
        print(json.dumps(res))
        return 0
    if args.cmd in ("promote", "latest"):
        from ..registry import ModelRegistry
        if not getattr(args, "registry", None):
            raise SystemExit(f"{args.cmd} needs --registry")
        reg = ModelRegistry(args.registry)
        if args.cmd == "promote":
            mv = reg.promote(args.name, args.version, args.stage, reason=args.reason)
            print(json.dumps({"name": args.name, "version": mv.version, "stage": mv.stage}))
        else:
            print(reg.get(args.name, stage=args.stage).version if args.stage else reg.get(args.name).version)
        return 0
    if args.cmd == "release":
        from ..environments import Environments, ReleaseError
        envs = Environments.from_yaml(args.config)
        try:
            mv = envs.release(args.name, args.version, frm=args.frm, to=args.to, released_by=args.released_by,
                              approved_by=[a for a in args.approved_by.split(",") if a], reason=args.reason)
        except ReleaseError as exc:
            print(f"RELEASE REFUSED: {exc}", file=sys.stderr)
            return 1
        res = {"name": args.name, "environment": args.to, "version": mv.version, "stage": mv.stage}
        cicd._write_json(args.report, res)
        print(json.dumps(res))
        return 0
    if args.cmd == "deploy-sagemaker":
        from . import aws
        image = aws.push_image(args.image, args.repository) if args.repository else args.image
        res = aws.deploy_sagemaker(image, args.endpoint, role_arn=args.role_arn, instance_type=args.instance_type,
                                   instance_count=args.instance_count,
                                   serverless={"memory_mb": args.serverless_memory_mb} if args.serverless_memory_mb else None,
                                   rollback_alarms=[a for a in args.rollback_alarms.split(",") if a])
        print(json.dumps(res, default=str))
        return 0
    written = cicd.init_cicd(args.provider, name=args.name, data=args.data, target=args.target, out=args.out,
                             metric=args.metric, floor=args.floor, budget=args.time_budget, overwrite=args.overwrite)
    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
