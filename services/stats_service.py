"""
Dashboard analytics. Every number comes from MySQL records (transactions and
stored model results) or from the saved model metadata. Nothing is hard-coded.
Charts are returned as Plotly.js figure dicts.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy import func

from database.models import AnomalyFeature as F
from database.models import FinancialDataset
from database.models import FinancialTransaction as T
from database.models import FraudAlert
from database.models import FraudAnalysis as A
from extensions import db
from services import modeling as md

RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
S1, S2, INK2 = "#2a78d6", "#eb6834", "#52514e"
RISK_ORDER = ["Low", "Medium", "High", "Critical"]


def _layout(**kw):
    base = {"margin": {"l": 50, "r": 15, "t": 28, "b": 45}, "hovermode": "closest",
            "xaxis": {"gridcolor": "#e1e0d9", "zeroline": False},
            "yaxis": {"gridcolor": "#e1e0d9", "zeroline": False},
            "legend": {"orientation": "h", "y": 1.12}, "bargap": 0.25}
    for k, v in kw.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return base


def _tx(dataset_id):
    q = db.session.query(T)
    return q.filter(T.dataset_id == dataset_id) if dataset_id else q


def _an(dataset_id, version):
    q = db.session.query(A).join(T, A.transaction_id == T.id)
    if version:
        q = q.filter(A.model_version == version)
    return q.filter(T.dataset_id == dataset_id) if dataset_id else q


def summary(dataset_id=None, version=None):
    analyzed = _an(dataset_id, version).with_entities(func.count(A.id)).scalar()
    suspicious = _an(dataset_id, version).filter(A.anomaly_status == "Suspicious").with_entities(func.count(A.id)).scalar()
    alerts_q = db.session.query(func.count(FraudAlert.id)).join(A).join(T, A.transaction_id == T.id)
    if dataset_id:
        alerts_q = alerts_q.filter(T.dataset_id == dataset_id)
    if version:
        alerts_q = alerts_q.filter(A.model_version == version)
    return {"transactions": _tx(dataset_id).with_entities(func.count(T.id)).scalar(),
            "datasets": db.session.query(func.count(FinancialDataset.id)).scalar(),
            "analyzed": analyzed, "suspicious": suspicious,
            "suspicious_share": (100 * suspicious / analyzed) if analyzed else 0.0,
            "open_alerts": alerts_q.filter(FraudAlert.status != "Closed").scalar(),
            "critical": _an(dataset_id, version).filter(A.risk_level == "Critical").with_entities(func.count(A.id)).scalar()}


def charts(dataset_id=None, version=None, max_points=100_000):
    out = {}
    by_type = (_tx(dataset_id).with_entities(T.transaction_type, func.count(T.id))
               .group_by(T.transaction_type).order_by(func.count(T.id)).all())
    if by_type:
        out["by_type"] = {"data": [{"type": "bar", "orientation": "h", "y": [r[0] for r in by_type],
                                    "x": [r[1] for r in by_type], "marker": {"color": S1},
                                    "hovertemplate": "%{y}: %{x:,} transactions<extra></extra>"}],
                          "layout": _layout(margin={"l": 100}, xaxis={"title": {"text": "transactions"}})}
    amounts = [r[0] for r in _tx(dataset_id).with_entities(T.transaction_amount).limit(max_points).all()]
    if amounts:
        out["amounts"] = {"data": [{"type": "histogram", "x": list(np.log10(np.clip(amounts, 1e-6, None))),
                                    "nbinsx": 50, "marker": {"color": S1},
                                    "hovertemplate": "10^%{x} : %{y:,} transactions<extra></extra>"}],
                          "layout": _layout(xaxis={"title": {"text": "amount paid (log10, payment currency)"}},
                                            yaxis={"title": {"text": "transactions"}})}
    risk = dict(_an(dataset_id, version).with_entities(A.risk_level, func.count(A.id)).group_by(A.risk_level).all())
    if risk:
        vals = [risk.get(k, 0) for k in RISK_ORDER]
        out["risk"] = {"data": [{"type": "bar", "x": RISK_ORDER, "y": vals,
                                 "marker": {"color": [RISK_COLORS[k] for k in RISK_ORDER]},
                                 "text": [f"{v:,}" for v in vals], "textposition": "outside", "cliponaxis": False,
                                 "hovertemplate": "%{x}: %{y:,}<extra></extra>"}],
                       "layout": _layout(yaxis={"type": "log", "title": {"text": "transactions (log)"}})}
        normal = sum(risk.get(k, 0) for k in ("Low", "Medium"))
        sus = sum(risk.get(k, 0) for k in md.SUSPICIOUS_LEVELS)
        out["status"] = {"data": [{"type": "bar", "x": ["Normal", "Suspicious"], "y": [normal, sus],
                                   "marker": {"color": [S1, RISK_COLORS["Critical"]]},
                                   "text": [f"{normal:,}", f"{sus:,}"], "textposition": "outside", "cliponaxis": False,
                                   "hovertemplate": "%{x}: %{y:,}<extra></extra>"}],
                         "layout": _layout(yaxis={"title": {"text": "analysed transactions"}})}
    rows = _an(dataset_id, version).with_entities(T.transaction_date, A.anomaly_status, A.combined_anomaly_score,
                                                  A.transaction_anomaly_score, A.graph_anomaly_score).limit(max_points).all()
    if rows:
        minutes = np.array([r[0] for r in rows], dtype="datetime64[m]").astype("int64")
        span = int(minutes.max() - minutes.min())
        # bucket size adapts to the data's time span: 5 min / 1 hour / 1 day
        step, unit_label = (5, "5 minutes") if span <= 180 else ((60, "hour") if span <= 72 * 60 else (1440, "day"))
        buckets = minutes // step * step
        sus = np.array([r[1] == "Suspicious" for r in rows])
        keys, inv = np.unique(buckets, return_inverse=True)
        tot, flagged = np.bincount(inv), np.bincount(inv, weights=sus)
        x = [str(np.datetime64(int(k), "m")) for k in keys]
        out["trend"] = {"data": [
            {"type": "scatter", "mode": "lines+markers", "x": x, "y": tot.tolist(), "name": "analysed",
             "line": {"color": S1, "width": 2}, "marker": {"size": 6}},
            {"type": "scatter", "mode": "lines+markers", "x": x, "y": flagged.tolist(), "name": "suspicious",
             "line": {"color": RISK_COLORS["Critical"], "width": 2}, "marker": {"size": 6}}],
            "layout": _layout(yaxis={"title": {"text": f"transactions per {unit_label}"}, "rangemode": "tozero"})}
        comb = np.array([r[2] for r in rows])
        out["scores"] = {"data": [
            {"type": "histogram", "x": comb[~sus].tolist(), "name": "Normal", "xbins": {"start": 0, "end": 100, "size": 2},
             "marker": {"color": S1}},
            {"type": "histogram", "x": comb[sus].tolist(), "name": "Suspicious", "xbins": {"start": 0, "end": 100, "size": 2},
             "marker": {"color": RISK_COLORS["Critical"]}}],
            "layout": _layout(barmode="stack", xaxis={"title": {"text": "anomaly score (training percentile)"}},
                              yaxis={"title": {"text": "transactions (log)"}, "type": "log"})}
        tr, gr = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
        cut = dict(md.RISK_LEVELS)["High"]
        both, t_only, g_only = int(((tr >= cut) & (gr >= cut)).sum()), int(((tr >= cut) & (gr < cut)).sum()), \
            int(((tr < cut) & (gr >= cut)).sum())
        out["comparison"] = {"data": [
            {"type": "bar", "name": "Transaction-only IF", "x": ["flagged (score ≥ 97)", "flagged by this model only"],
             "y": [int((tr >= cut).sum()), t_only], "marker": {"color": S1}},
            {"type": "bar", "name": "Graph-enhanced IF", "x": ["flagged (score ≥ 97)", "flagged by this model only"],
             "y": [int((gr >= cut).sum()), g_only], "marker": {"color": S2}}],
            "layout": _layout(barmode="group", yaxis={"title": {"text": "transactions"}})}
        out["comparison_overlap"] = both
    feats = (db.session.query(F.feature_name, func.count(F.id)).join(A, F.analysis_id == A.id)
             .join(T, A.transaction_id == T.id))
    if dataset_id:
        feats = feats.filter(T.dataset_id == dataset_id)
    if version:
        feats = feats.filter(A.model_version == version)
    feats = feats.group_by(F.feature_name).order_by(func.count(F.id).desc()).limit(10).all()
    if feats:
        feats = feats[::-1]
        out["features"] = {"data": [{"type": "bar", "orientation": "h", "y": [md.feature_label(f[0]) for f in feats],
                                     "x": [f[1] for f in feats], "marker": {"color": S2},
                                     "hovertemplate": "%{y}: in the top 3 of %{x:,} analyses<extra></extra>"}],
                           "layout": _layout(margin={"l": 280}, xaxis={"title": {"text": "times in a top-3 explanation"}})}
    return out


def labelled_evaluation(dataset_id, version):
    """If the dataset carries labels, compare both models on it (display only - labels never feed the models)."""
    rows = (_an(dataset_id, version).filter(T.actual_label.isnot(None))
            .with_entities(T.actual_label, A.transaction_anomaly_score, A.graph_anomaly_score).all())
    if not rows:
        return None
    y = np.array([r[0] for r in rows])
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": len(y), "positives": int(y.sum()), "models": []}
    cut = dict(md.RISK_LEVELS)["High"]
    res = []
    for name, idx in (("Transaction-only IF", 1), ("Graph-enhanced IF", 2)):
        s = np.array([r[idx] for r in rows])
        order = np.argsort(-s, kind="stable")
        hits = np.cumsum(y[order])
        prec = hits / np.arange(1, len(y) + 1)
        ap = float((prec * y[order]).sum() / y.sum())
        pred = s >= cut
        tp = int((pred & (y == 1)).sum())
        res.append({"model": name, "average_precision": ap, "flagged": int(pred.sum()), "tp": tp,
                    "precision": tp / pred.sum() if pred.sum() else 0.0, "recall": tp / y.sum()})
    return {"n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()), "models": res}
