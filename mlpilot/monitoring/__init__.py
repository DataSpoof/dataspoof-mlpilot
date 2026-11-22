"""Monitoring (module 9): data & prediction drift, accuracy decay, alerts, retraining triggers.

* ``drift_report(reference, current)``      PSI / JS distance / chi-square per column.
* ``Monitor(model, reference)``             ``check(batch)`` -> alerts + retrain recommendation;
                                            saves baselines only (no raw data).
* ``retrain(model, train, holdout, registry=..., name=...)``  champion vs challenger, with
                                            registration and gated promotion.
* ``webhook_alert(url)``                    send alerts to Slack/Teams/any webhook.
* ``PerformanceEstimator(model)``           accuracy / ROC AUC / MAE estimated *without labels*
                                            (Monitor.check uses it when a batch has no labels).
* ``domain_classifier_drift(ref, cur)``     multivariate drift: can a model tell the two apart?
* ``diagnose(...)`` / ``IncidentLog``       root cause (upstream data, infrastructure, bad release,
                                            concept drift, covariate shift), playbooks, approvals,
                                            verified recovery.
"""

from .drift import Baseline, DriftReport, drift_report, psi
from .estimate import PerformanceEstimator, domain_classifier_drift
from .incidents import ACTIONS, Action, Diagnosis, Incident, IncidentLog, diagnose
from .monitor import (Alert, Monitor, MonitorReport, RetrainPolicy, RetrainResult, read_prediction_log, retrain,
                      retrain_if_needed, webhook_alert)
from .ops import join_feedback, join_labels, opsgenie_alert, pagerduty_alert, sql_sink

__all__ = ["Baseline", "DriftReport", "drift_report", "psi", "Alert", "Monitor", "MonitorReport", "RetrainPolicy",
           "RetrainResult", "read_prediction_log", "retrain", "retrain_if_needed", "webhook_alert", "join_labels", "join_feedback",
           "opsgenie_alert", "pagerduty_alert", "sql_sink", "PerformanceEstimator", "domain_classifier_drift",
           "diagnose", "Diagnosis", "IncidentLog", "Incident", "Action", "ACTIONS"]
