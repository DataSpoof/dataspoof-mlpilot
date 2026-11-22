"""Autonomous execution engine: from a goal to a registered, monitored model, step by step.

>>> from mlpilot import orchestration as orc
>>> goal = orc.goal_from_text("Predict churned from data/customers.csv with AUC above 0.85, "
...                           "latency below 50 ms; deploy to production")
>>> print(orc.plan(goal).describe())
>>> res = orc.run(goal, "runs/churn")          # stops at the production approval
>>> orc.approve("runs/churn", "promote", by="priya")
>>> orc.resume("runs/churn")

Design rules (enforced in code, not by convention):

* **Plans use registered tools only** (``TOOLS``): a planner, rule-based or an LLM, chooses steps
  and typed parameters; it can't run arbitrary code.
* **Gates are deterministic** (data quality and contracts, poisoning, metric floors, latency and
  size limits, fairness, robustness) and are never retried away.
* **High-risk steps need people.** Promotion waits for the goal's number of approvers, none of
  whom may be the requester, and approvals are tied to the exact inputs they approved.
* **Runs are resumable and idempotent.** Each step's inputs are hashed, unchanged steps are
  reused, and every event, cost and decision is recorded in the run directory.

CLI: ``mlpilot plan``, ``mlpilot run``, ``mlpilot approve``, ``mlpilot status``.
"""

from .engine import RunResult, StepSpec, Workflow, approve, plan, resume, run, status
from .goal import Goal, GoalError, goal_from_text
from .tools import TOOLS, GateFailed, RunContext, ToolSpec, frame_fingerprint, register_tool

__all__ = ["Goal", "GoalError", "goal_from_text", "plan", "run", "resume", "approve", "status", "Workflow",
           "StepSpec", "RunResult", "TOOLS", "ToolSpec", "register_tool", "GateFailed", "RunContext", "frame_fingerprint"]
