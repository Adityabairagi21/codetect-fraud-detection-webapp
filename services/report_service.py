"""
Downloadable fraud analysis reports (CSV and PDF) built from stored analysis records.
"""
from __future__ import annotations

import csv
import uuid
from collections import Counter
from datetime import datetime

from flask import current_app
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy.orm import selectinload

from database.models import FinancialDataset, FinancialTransaction, FraudAnalysis, InvestigationReport
from extensions import db
from services import modeling as md

RISK_ORDER = ["Low", "Medium", "High", "Critical"]


class ReportError(ValueError):
    pass


def _analyses(dataset_id, version):
    q = (db.session.query(FraudAnalysis).join(FinancialTransaction)
         .options(selectinload(FraudAnalysis.features), selectinload(FraudAnalysis.transaction)))
    if version:
        q = q.filter(FraudAnalysis.model_version == version)
    if dataset_id:
        q = q.filter(FinancialTransaction.dataset_id == dataset_id)
    return q


def generate(dataset_id: int | None, fmt: str, user, version: str | None) -> InvestigationReport:
    fmt = fmt.upper()
    if fmt not in ("CSV", "PDF"):
        raise ReportError("Unknown report format.")
    ds = db.session.get(FinancialDataset, dataset_id) if dataset_id else None
    if dataset_id and ds is None:
        raise ReportError("Dataset not found.")
    total = _analyses(dataset_id, version).count()
    if total == 0:
        raise ReportError("No analysis results yet. Run fraud detection on the dataset first.")
    risk = Counter(r[0] for r in _analyses(dataset_id, version).with_entities(FraudAnalysis.risk_level).all())
    suspicious = (_analyses(dataset_id, version).filter(FraudAnalysis.anomaly_status == "Suspicious")
                  .order_by(FraudAnalysis.combined_anomaly_score.desc()).all())
    model_name = suspicious[0].model_name if suspicious else \
        _analyses(dataset_id, version).first().model_name
    analysis_date = _analyses(dataset_id, version).with_entities(db.func.max(FraudAnalysis.analyzed_date)).scalar()
    ctx = {"dataset": ds, "total": total, "risk": risk, "suspicious": suspicious, "model_name": model_name,
           "version": version, "analysis_date": analysis_date, "user": user, "generated": datetime.now()}

    folder = current_app.config["REPORTS_FOLDER"]
    name = f"codetect_report_{ds.id if ds else 'all'}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.{fmt.lower()}"
    path = folder / name
    (_write_csv if fmt == "CSV" else _write_pdf)(path, ctx)
    rep = InvestigationReport(generated_by=user.id, dataset_id=ds.id if ds else None,
                              report_type="Fraud Analysis Report", report_format=fmt, file_path=name)
    db.session.add(rep)
    db.session.commit()
    return rep


def _feature_cols(a):
    out = {}
    for f in a.features[:3]:
        out[f"top{f.rank}_feature"] = md.feature_label(f.feature_name)
        out[f"top{f.rank}_value"] = f.feature_value
        out[f"top{f.rank}_abnormality"] = round(f.contribution_score, 3)
        out[f"top{f.rank}_explanation"] = f.description
    return out


def _write_csv(path, c):
    ds = c["dataset"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# CoDetect fraud analysis report"])
        w.writerow(["# dataset", ds.dataset_name if ds else "All datasets"])
        w.writerow(["# source", ds.source_name if ds else "-"])
        w.writerow(["# model", c["model_name"], "version", c["version"]])
        w.writerow(["# analysis date", c["analysis_date"], "generated", f"{c['generated']:%Y-%m-%d %H:%M}",
                    "by", c["user"].email])
        w.writerow(["# total analysed", c["total"], "suspicious", len(c["suspicious"])])
        w.writerow(["# risk distribution"] + [f"{k}={c['risk'].get(k, 0)}" for k in RISK_ORDER])
        w.writerow(["# note", "Anomaly scores rank how unusual a transaction is versus training data; "
                              "they are not fraud probabilities. Suspicious does not mean confirmed fraud."])
        header = ["transaction_reference", "transaction_date", "sender", "receiver", "amount", "currency",
                  "transaction_type", "transaction_anomaly_score", "graph_anomaly_score", "combined_anomaly_score",
                  "risk_level", "anomaly_status", "model_name", "analyzed_date"]
        feat_header = [f"top{i}_{k}" for i in (1, 2, 3) for k in ("feature", "value", "abnormality", "explanation")]
        w.writerow(header + feat_header)
        for a in c["suspicious"]:
            t = a.transaction
            fc = _feature_cols(a)
            w.writerow([t.transaction_reference, t.transaction_date, t.sender_id, t.receiver_id,
                        t.transaction_amount, t.payment_currency, t.transaction_type,
                        round(a.transaction_anomaly_score, 3), round(a.graph_anomaly_score, 3),
                        round(a.combined_anomaly_score, 3), a.risk_level, a.anomaly_status, a.model_name,
                        a.analyzed_date] + [fc.get(h, "") for h in feat_header])


def _write_pdf(path, c, max_rows=150):
    ds = c["dataset"]
    st = getSampleStyleSheet()
    small = st["BodyText"].clone("small", fontSize=7.5, leading=9)
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="CoDetect fraud analysis report")
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14213d")),
                       ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                       ("VALIGN", (0, 0), (-1, -1), "TOP")])
    story = [Paragraph("CoDetect – Fraud Analysis Report", st["Title"]),
             Paragraph("Anomaly scores rank how unusual a transaction is compared with the training data. They are "
                       "<b>not</b> fraud probabilities, and a Suspicious result is an investigation lead, not "
                       "confirmed fraud.", st["BodyText"]), Spacer(1, 6)]
    info = [["Dataset", ds.dataset_name if ds else "All datasets", "Model", c["model_name"]],
            ["Source", ds.source_name if ds else "-", "Model version", c["version"] or "-"],
            ["Records in dataset", f"{ds.total_records:,}" if ds else "-", "Analysis date", str(c["analysis_date"])],
            ["Transactions analysed", f"{c['total']:,}", "Generated", f"{c['generated']:%Y-%m-%d %H:%M} by {c['user'].email}"],
            ["Suspicious (High + Critical)", f"{len(c['suspicious']):,}", "", ""]]
    t = Table(info, colWidths=[55 * mm, 75 * mm, 35 * mm, 100 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")),
                           ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0efec")),
                           ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0efec"))]))
    story += [t, Spacer(1, 8), Paragraph("Risk-level distribution", st["Heading2"])]
    total = max(c["total"], 1)
    rt = Table([["Risk level", "Transactions", "Share"]] +
               [[k, f"{c['risk'].get(k, 0):,}", f"{100 * c['risk'].get(k, 0) / total:.2f}%"] for k in RISK_ORDER],
               colWidths=[40 * mm, 40 * mm, 30 * mm])
    rt.setStyle(grid)
    story += [rt, Spacer(1, 8)]
    feat_counts = Counter(f.feature_name for a in c["suspicious"] for f in a.features)
    if feat_counts:
        story.append(Paragraph("Most frequent abnormal indicators among suspicious transactions", st["Heading2"]))
        ft = Table([["Feature", "Times in a top-3 explanation"]] +
                   [[md.feature_label(k), f"{v:,}"] for k, v in feat_counts.most_common(8)],
                   colWidths=[110 * mm, 50 * mm])
        ft.setStyle(grid)
        story += [ft, Spacer(1, 8)]
    story.append(Paragraph(f"Suspicious transactions (highest scores first"
                           f"{', first %d shown' % max_rows if len(c['suspicious']) > max_rows else ''})",
                           st["Heading2"]))
    rows = [["Reference", "Date", "Sender", "Receiver", "Amount", "Type", "Txn", "Graph", "Score", "Risk",
             "Top abnormal indicator"]]
    for a in c["suspicious"][:max_rows]:
        tr = a.transaction
        top = a.features[0].description if a.features else ""
        rows.append([tr.transaction_reference, f"{tr.transaction_date:%Y-%m-%d %H:%M}", tr.sender_id, tr.receiver_id,
                     f"{tr.transaction_amount:,.2f} {tr.payment_currency[:3]}", tr.transaction_type,
                     f"{a.transaction_anomaly_score:.1f}", f"{a.graph_anomaly_score:.1f}",
                     f"{a.combined_anomaly_score:.1f}", a.risk_level, Paragraph(top, small)])
    if len(rows) == 1:
        rows.append(["No suspicious transactions"] + [""] * 10)
    stt = Table(rows, colWidths=[27 * mm, 24 * mm, 30 * mm, 30 * mm, 27 * mm, 18 * mm, 10 * mm, 11 * mm, 11 * mm,
                                 14 * mm, 70 * mm], repeatRows=1)
    stt.setStyle(grid)
    story.append(stt)
    doc.build(story)


def report_path(rep: InvestigationReport):
    """Resolve a stored report file safely inside REPORTS_FOLDER (no traversal)."""
    folder = current_app.config["REPORTS_FOLDER"].resolve()
    p = (folder / rep.file_path).resolve()
    if folder not in p.parents or not p.exists():
        return None
    return p
