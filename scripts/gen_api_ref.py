"""Generate the GitHub-readable API reference (docs/api/) from the source, without importing it.

usage: python scripts/gen_api_ref.py [repo root] [output dir]      (defaults: . and docs/api)

Reads every public module with ``ast``. For each public function and class it writes the
signature, the docstring (converted from reST to Markdown), dataclass fields, the constructor
and the public methods and properties. The site build (MkDocs) has its own reference in
docs/reference; this one renders on GitHub.
"""

import ast
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC / "docs" / "api"
PKG = SRC / "mlpilot"
CODE_REPO = "https://github.com/DataSpoof/dataspoof-mlpilot/blob/main"

AREAS = [  # (page, title, files or globs, guide page)
    ("core", "Pipeline core", ["core/*.py"], None),
    ("ingestion", "Ingestion & connectors", ["ingestion/*.py"], "ingestion.md"),
    ("cleaning", "Data cleaning", ["cleaners.py", "steps.py", "cleaning.py"], "cleaning.md"),
    ("quality", "Profiling, data quality & observability", ["quality/*.py"], "quality.md"),
    ("splitting", "Data splitting", ["splitting.py"], "splitting.md"),
    ("features", "Feature engineering", ["features/*.py"], "features.md"),
    ("feature_store", "Feature store", ["feature_store.py"], "feature-store.md"),
    ("versioning", "Data versioning", ["versioning.py"], "versioning.md"),
    ("stats", "Statistical testing", ["stats.py"], "stats.md"),
    ("automl", "AutoML", ["automl/*.py"], "automl.md"),
    ("evaluation", "Evaluation & explainability", ["evaluation/*.py"], "evaluation.md"),
    ("model_security", "Model security & privacy", ["robustness.py", "poisoning.py", "model_privacy.py"],
     "model-security.md"),
    ("registry", "Tracking, registry & model governance", ["registry/*.py"], "registry.md"),
    ("orchestration", "Autonomous execution engine", ["orchestration/*.py"], "enterprise-operations.md"),
    ("environments", "Environments & CI/CD", ["environments.py", "cicd.py"], "environments-cicd.md"),
    ("deploy", "Deployment", ["deploy/*.py"], "deployment.md"),
    ("compression", "Quantization & compression", ["compression.py"], "quantization.md"),
    ("monitoring", "Monitoring & incidents", ["monitoring/*.py"], "monitoring.md"),
    ("operations", "Operations: backups, labeling, supply chain", ["backup.py", "labeling.py", "supply_chain.py"],
     "enterprise-operations.md"),
    ("insights", "Insights & reporting", ["insights/*.py"], "insights.md"),
    ("hub", "Hub (platform server)", ["hub/*.py"], "hub.md"),
    ("secrets_security", "Secrets & model signing", ["secretstore.py", "security.py"], "secrets-policy.md"),
    ("lineage", "Lineage", ["lineage.py"], "lineage.md"),
    ("forecast", "Forecasting", ["forecast.py"], "forecasting.md"),
    ("causal", "Causal inference & uplift", ["causal.py"], "causal.md"),
    ("anomaly", "Anomaly detection", ["anomaly.py"], "anomaly.md"),
    ("media", "Image & audio data", ["media/__init__.py", "media/image.py", "media/audio.py"], "media.md"),
    ("vision", "Weather, faces & plates", ["media/conditions.py", "media/faces.py", "media/plates.py",
                                           "media/benchmark.py"], "vision.md"),
    ("scale", "Scale-out & SQL pushdown", ["scale.py"], "integrations.md"),
    ("integrations", "Integrations", ["integrations/*.py"], "integrations.md"),
]
SKIP_FILES = {"__main__.py"}


def files_for(patterns) -> list:
    out = []
    for pat in patterns:
        for p in sorted(PKG.glob(pat)):
            if p.name in SKIP_FILES or (p.name.startswith("_") and p.name != "__init__.py"):
                continue
            if p not in out:
                out.append(p)
    return sorted(out, key=lambda p: (p.parent != PKG and p.name != "__init__.py", p.name != "__init__.py", str(p)))


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def md_doc(doc: str | None) -> str:
    """reST-ish docstring -> Markdown: doctest blocks, ``literal::`` blocks and double backticks."""
    if not doc:
        return ""
    lines = doc.strip("\n").split("\n")
    out, i = [], 0
    item: tuple | None = None                          # (bullet indent, text column) of the open list item
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith(">>> "):
            block = []
            while i < len(lines) and lines[i].strip():
                t = lines[i].strip()
                block.append(t[4:] if t.startswith((">>> ", "... ")) else ("" if t in (">>>", "...") else t))
                i += 1
            out += ["```python", *block, "```"]
            continue
        if s.endswith("::") and not s.startswith(".."):
            out.append(line.rstrip()[:-2].rstrip() + ":" if s != "::" else "")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            indent = len(lines[i]) - len(lines[i].lstrip()) if i < len(lines) else 0
            block = []
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) >= indent):
                block.append(lines[i][indent:])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            out += ["", "```", *block, "```", ""]
            continue
        bullet = re.match(r"^(\s*)([*-]|\d+\.)\s", line)
        indent = len(line) - len(line.lstrip())
        if bullet:                                          # a list item: remember where its text starts
            item = (len(bullet.group(1)), len(bullet.group(1)) + len(bullet.group(2)) + 1)
        elif not s:
            item = None
        elif item is not None and indent > item[0]:
            line = " " * item[1] + s                        # continuation: align under the item's text
        else:
            item = None
        out.append(line)
        i += 1
    fenced, result = False, []
    for line in out:                                        # inline conversions, never inside code fences
        if line.strip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            line = re.sub(r"``([^`\n]+)``", r"`\1`", line)
            line = re.sub(r":(?:func|class|meth|attr|mod):`~([^`]+)`", lambda m: f"`{m.group(1).split('.')[-1]}`", line)
            line = re.sub(r":(?:func|class|meth|attr|mod):`([^`]+)`", r"`\1`", line)
        result.append(line)
    return "\n".join(result).strip()


def signature(fn, *, method: bool = False, name: str | None = None) -> str:
    import copy
    node = copy.copy(fn)                              # keep positions; drop the body and decorators
    node.body, node.decorator_list, node.name = [ast.Pass()], [], name or fn.name
    text = ast.unparse(node).split("\n")[0].rstrip(":")
    if method:
        text = re.sub(r"^def ", "", text)
        text = re.sub(r"\((self|cls)(, )?", "(", text, count=1)
    return text


def is_dataclass(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        src = ast.unparse(d)
        if src.split("(")[0] in ("dataclass", "dataclasses.dataclass"):
            return True
    return False


def decorators(fn) -> set:
    return {ast.unparse(d).split("(")[0] for d in fn.decorator_list}


def render_class(cls: ast.ClassDef) -> tuple:
    lines = [f"### class `{cls.name}`", ""]
    bases = ", ".join(ast.unparse(b) for b in cls.bases)
    lines += ["```python", f"class {cls.name}({bases}):" if bases else f"class {cls.name}:", "```", ""]
    doc = md_doc(ast.get_docstring(cls))
    if doc:
        lines += [doc, ""]
    if is_dataclass(cls):
        rows = []
        for st in cls.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and not st.target.id.startswith("_"):
                ann = ast.unparse(st.annotation).replace("|", "\\|")
                default = "required"
                if st.value is not None:
                    v = st.value
                    if isinstance(v, ast.Call) and ast.unparse(v.func).split(".")[-1] == "field":
                        kw = {k.arg: k.value for k in v.keywords}
                        if "default_factory" in kw:
                            default = f"`{ast.unparse(kw['default_factory'])}()`"
                        elif "default" in kw:
                            default = f"`{ast.unparse(kw['default'])}`"
                    else:
                        default = f"`{ast.unparse(v)}`"
                rows.append(f"| `{st.target.id}` | `{ann}` | {default.replace('|', '/')} |")
        if rows:
            lines += ["| Field | Type | Default |", "|---|---|---|", *rows, ""]
    else:
        init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is not None:
            lines += ["**Constructor**", "", "```python", signature(init, method=True, name=cls.name), "```", ""]
    for n in cls.body:
        if not isinstance(n, ast.FunctionDef) or n.name.startswith("_"):
            continue
        decs = decorators(n)
        if "property" in decs:
            decl = f"{n.name}: {ast.unparse(n.returns)}" if n.returns else n.name
            lines += [f"#### `{cls.name}.{n.name}` *(property)*", "", "```python", decl, "```", ""]
        else:
            kind = " *(classmethod)*" if "classmethod" in decs else (" *(staticmethod)*" if "staticmethod" in decs else "")
            lines += [f"#### `{cls.name}.{n.name}()`{kind}", "", "```python", signature(n, method=True), "```", ""]
        d = md_doc(ast.get_docstring(n))
        if d:
            lines += [d, ""]
    return lines


def public_names(tree: ast.Module) -> list | None:
    for st in tree.body:
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in st.targets):
            try:
                return list(ast.literal_eval(st.value))
            except ValueError:
                return None
    return None


def render_module(path: Path) -> tuple:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    name = module_name(path)
    rel = path.relative_to(SRC).as_posix()
    head = [f"## `{name}`", "", f"<sub>Source: [`{rel}`]({CODE_REPO}/{rel})</sub>", ""]
    mdoc = md_doc(ast.get_docstring(tree))
    if mdoc:
        head += [mdoc, ""]
    allowed = public_names(tree)
    body, toc = [], []
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.ClassDef)):
            if st.name.startswith("_") or (allowed is not None and st.name not in allowed):
                continue
            toc.append(st.name)
            if isinstance(st, ast.ClassDef):
                body += render_class(st)
            else:
                body += [f"### `{st.name}()`", "", "```python", signature(st), "```", ""]
                d = md_doc(ast.get_docstring(st))
                if d:
                    body += [d, ""]
    para = " ".join((ast.get_docstring(tree) or "").strip().split("\n\n")[0].split())
    para = re.sub(r"``([^`]+)``", r"`\1`", para)
    m = re.match(r"(.+?[.!?])(\s|$)", para)
    summary = m.group(1) if m else para
    if len(summary) > 160:
        summary = summary[:157].rsplit(" ", 1)[0] + "…"
    return name, head + body, toc, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index_rows, total, pages = [], 0, 0
    for page, title, patterns, guide in AREAS:
        mods = [render_module(p) for p in files_for(patterns)]
        mods = [m for m in mods if m[2] or m[3]]
        if not mods:
            continue
        lines = [f"# {title}: API reference", ""]
        if guide:
            lines += [f"Generated from the source docstrings. For a task-oriented introduction, read the "
                      f"[{title} guide](../guide/{guide}).", ""]
        else:
            lines += ["Generated from the source docstrings.", ""]
        lines += ["**On this page:**", ""]
        for name, _, toc, _ in mods:
            anchor = re.sub(r"[^a-z0-9 _-]", "", name.lower()).replace(" ", "-")
            items = ", ".join(f"`{t}`" for t in toc[:14]) + (" …" if len(toc) > 14 else "")
            lines.append(f"- [`{name}`](#{anchor})" + (f": {items}" if items else ""))
            total += len(toc)
        lines.append("")
        for _, body, _, _ in mods:
            lines += ["---", ""] + body
        (OUT / f"{page}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        pages += 1
        first = next((m[3] for m in mods if m[3]), "")
        index_rows.append((page, title, sum(len(m[2]) for m in mods), first))
    idx = ["# API reference", "",
           "Every public class and function in `mlpilot`, generated from the source code's docstrings "
           "(`python scripts/gen_api_ref.py`). Start with the [documentation](../../README.md#-documentation) "
           "for task-oriented examples.", "",
           "| Area | Public items | Summary |", "|---|---:|---|"]
    for page, title, n, first in index_rows:
        idx.append(f"| [{title}]({page}.md) | {n} | {first.replace('|', '/')} |")
    (OUT / "README.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"{pages} reference pages, {total} public items documented")


if __name__ == "__main__":
    main()
