"""
Fraud detection service: the only bridge between Flask and the Stage 1 models.

* Loads the saved artifacts once through services.fraud_inference.FraudInference
  (copied unchanged from Stage 1). Nothing is trained here, and no feature
  formula is re-implemented.
* History context for the 24h features is documented and deterministic:
  ml_artifacts/history_context.joblib (real cleaned transactions from Sep 8 to
  Sep 10 2022) supplies everything strictly before a batch's first minute; inside
  the batch, the batch's own earlier transactions are used. The coverage is
  reported with every analysis.
* Results are stored in Fraud_Analysis / Anomaly_Features / Fraud_Alerts with
  one analysis per (transaction, model version).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from flask import current_app

from database.models import AnomalyFeature, FinancialTransaction, FraudAlert, FraudAnalysis, now
from extensions import db
from services import feature_engineering as fe
from services import modeling as md
from services.dataset_service import dataset_frame
from services.fraud_inference import ArtifactError, FraudInference

_LOCK = threading.Lock()


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class AnalysisSummary:
    analyzed: int = 0
    skipped_existing: int = 0
    suspicious: int = 0
    alerts_created: int = 0
    risk_counts: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)


# ------------------------------------------------------------------ artifact loading
def _state(app):
    st = app.extensions.setdefault("codetect_ml", {})
    if "engine" not in st and "error" not in st:
        with _LOCK:
            if "engine" not in st and "error" not in st:
                try:
                    st["engine"] = FraudInference(app.config["ML_ARTIFACTS_DIR"])
                    meta = st["engine"].metadata
                    sel_file = "graph_model.joblib" if meta["selected_model"] == "graph_enhanced" \
                        else "transaction_model.joblib"
                    st["version"] = f"f{meta['feature_version']}-{meta['artifact_sha256'][sel_file][:10]}"
                    ctx = app.config["ML_ARTIFACTS_DIR"] / "history_context.joblib"
                    st["history"] = joblib.load(ctx) if ctx.exists() else st["engine"].history
                except ArtifactError as exc:
                    st["error"] = str(exc)
    return st


def model_status(app=None) -> dict:
    app = app or current_app._get_current_object()
    st = _state(app)
    if "error" in st:
        return {"ok": False, "message": st["error"]}
    meta = st["engine"].metadata
    return {"ok": True, "version": st["version"], "selected": meta["selected_model"],
            "model_name": md.MODEL_LABELS[meta["selected_model"]], "created": meta["created"]}


def get_engine():
    st = _state(current_app._get_current_object())
    if "error" in st:
        raise ModelUnavailable(st["error"])
    return st["engine"], st["version"], st["history"]


def reset_engine(app):
    app.extensions.pop("codetect_ml", None)


# ------------------------------------------------------------------ history context
def history_before(history: pd.DataFrame, start_minute: int):
    h = history[(history["minute"] < start_minute)
                & (history["minute"] >= start_minute - fe.HISTORY_WINDOW_MINUTES)]
    if len(history) == 0 or start_minute - fe.HISTORY_WINDOW_MINUTES < int(history["minute"].min()) \
            or int(history["minute"].max()) < start_minute - fe.HISTORY_WINDOW_MINUTES:
        coverage = "partial" if len(h) else "none"
    else:
        coverage = "full"
    info = {"coverage": coverage, "context_rows": int(len(h)),
            "message": {"full": "Full 24-hour history before the first transaction was available.",
                        "partial": "Only part of the previous 24 hours was available; early transactions "
                                   "may look less connected than they really are.",
                        "none": "No earlier transactions were available; history features for the first "
                                "24 hours of this dataset are built from the dataset itself only."}[coverage]}
    return h, info


# ------------------------------------------------------------------ analysis
def analyze_dataset(dataset, user) -> AnalysisSummary:
    engine, version, history = get_engine()
    df = dataset_frame(dataset.id)
    if df.empty:
        raise ValueError("This dataset has no transactions.")
    problems = engine.validate(df)
    if problems:
        raise ValueError(" ".join(problems))
    hist, info = history_before(history, int(df["minute"].min()))
    res = engine.score(df, history=hist, already_clean=True, explain="suspicious")
    summary = _persist(res, df, version, user, engine)
    summary.history = info
    dataset.status = "Analyzed"
    db.session.commit()
    return summary


def analyze_transaction(tx: FinancialTransaction, user):
    """Score one transaction with its true previous-24h context. Returns (analysis, created, history_info)."""
    engine, version, history = get_engine()
    existing = FraudAnalysis.query.filter_by(transaction_id=tx.id, model_version=version).first()
    if existing is not None:
        return existing, False, None
    window = dataset_frame(tx.dataset_id, start=tx.transaction_date - timedelta(days=1), end=tx.transaction_date)
    problems = engine.validate(window)
    if problems:
        raise ValueError(" ".join(problems))
    hist, info = history_before(history, int(window["minute"].min()))
    res = engine.score(window, history=hist, already_clean=True, explain="none")
    pos = np.flatnonzero(window["db_id"].to_numpy() == tx.id)
    idx = window.index[pos]
    expl = md.top_abnormal_features(res.features.loc[idx], engine.feature_reference,
                                    engine.metadata["input_features"][engine.selected_model], k=3)
    expl.insert(0, "transaction_reference", tx.transaction_reference)
    one = res.results.loc[idx]
    _persist_rows(one, window.loc[idx], expl, version, user)
    db.session.commit()
    return FraudAnalysis.query.filter_by(transaction_id=tx.id, model_version=version).one(), True, info


def _persist(res, df, version, user, engine) -> AnalysisSummary:
    summary = AnalysisSummary()
    done = {r[0] for r in db.session.query(FraudAnalysis.transaction_id)
            .filter(FraudAnalysis.model_version == version,
                    FraudAnalysis.transaction_id.in_(df["db_id"].tolist())).all()}
    keep = ~df["db_id"].isin(done).to_numpy()
    summary.skipped_existing = int((~keep).sum())
    results = res.results[keep]
    created = _persist_rows(results, df[keep], res.explanations, version, user)
    summary.analyzed = len(results)
    summary.suspicious = int((results["anomaly_status"] == "Suspicious").sum())
    summary.alerts_created = created
    summary.risk_counts = {k: int(v) for k, v in results["risk_level"].value_counts()
                           .reindex(["Low", "Medium", "High", "Critical"], fill_value=0).items()}
    return summary


def _persist_rows(results: pd.DataFrame, df: pd.DataFrame, explanations: pd.DataFrame, version, user) -> int:
    """Insert analyses (+ top features for explained rows, + alerts for High/Critical). Returns alerts created."""
    by_ref = {} if explanations is None or explanations.empty else \
        {ref: g.sort_values("rank") for ref, g in explanations.groupby("transaction_reference")}
    db_ids = dict(zip(df["transaction_reference"], df["db_id"]))
    alerts = 0
    stamp = now()
    rows = list(results.itertuples(index=False))
    for i in range(0, len(rows), 2000):
        objs = []
        for r in rows[i:i + 2000]:
            a = FraudAnalysis(transaction_id=int(db_ids[r.transaction_reference]), model_name=r.model_name,
                              model_version=version, transaction_anomaly_score=float(r.transaction_anomaly_score),
                              graph_anomaly_score=float(r.graph_anomaly_score),
                              combined_anomaly_score=float(r.combined_anomaly_score), risk_level=r.risk_level,
                              anomaly_status=r.anomaly_status, analyzed_date=stamp,
                              analyzed_by=getattr(user, "id", None))
            feats = by_ref.get(r.transaction_reference)
            if feats is not None:
                a.features = [AnomalyFeature(rank=int(f.rank), feature_name=f.feature_name,
                                             feature_value=_fmt_value(f.feature_value),
                                             contribution_score=float(f.contribution_score),
                                             description=f.description[:400])
                              for f in feats.itertuples(index=False)]
            if r.risk_level in md.SUSPICIOUS_LEVELS:
                top = feats.iloc[0]["description"] if feats is not None and len(feats) else ""
                a.alert = FraudAlert(
                    risk_level=r.risk_level, created_date=stamp, status="New",
                    alert_message=(f"{r.risk_level} risk: anomaly score {r.combined_anomaly_score:.1f} for "
                                   f"{r.transaction_reference} ({r.sender_id} -> {r.receiver_id}, "
                                   f"USD {r.amount_usd:,.2f}). {top}")[:400])
                alerts += 1
            objs.append(a)
        db.session.add_all(objs)
        db.session.flush()
    return alerts


def _fmt_value(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)[:60]


# ------------------------------------------------------------------ feature snapshot (detail page)
def feature_snapshot(tx: FinancialTransaction) -> pd.DataFrame | None:
    """All 21 features of a transaction with their training abnormality, computed live (same code path)."""
    try:
        engine, _, history = get_engine()
    except ModelUnavailable:
        return None
    window = dataset_frame(tx.dataset_id, start=tx.transaction_date - timedelta(days=1), end=tx.transaction_date)
    hist, _ = history_before(history, int(window["minute"].min()))
    feats = fe.compute_features(window, engine.fx, hist)
    idx = window.index[window["db_id"].to_numpy() == tx.id]
    row = feats.loc[idx]
    ab, direction = md.feature_abnormality(row, engine.feature_reference, fe.ALL_FEATURES)
    used = set(engine.metadata["input_features"][engine.selected_model])
    schema = {f["name"]: f for f in fe.FEATURE_SCHEMA}
    return pd.DataFrame({
        "feature": [md.feature_label(f) for f in fe.ALL_FEATURES],
        "name": fe.ALL_FEATURES,
        "group": [schema[f]["group"] for f in fe.ALL_FEATURES],
        "value": [row.iloc[0][f] for f in fe.ALL_FEATURES],
        "abnormality": ab.iloc[0].to_numpy(),
        "direction": direction.iloc[0].to_numpy(),
        "used_by_selected_model": [f in used for f in fe.ALL_FEATURES],
    })
