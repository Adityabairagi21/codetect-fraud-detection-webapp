"""Pages available to both roles (admin-only actions are enforced in routes/admin.py)."""
from datetime import datetime, timedelta

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required
from sqlalchemy import and_, or_

from database.models import (RISK_LEVELS, FinancialDataset, FinancialTransaction, FraudAlert, FraudAnalysis,
                             InvestigationReport)
from extensions import db
from forms import ActionForm, AlertReviewForm, ReportForm
from services import dataset_service, fraud_service, graph_service, report_service, stats_service

bp = Blueprint("analysis", __name__)
T, A = FinancialTransaction, FraudAnalysis


def _version():
    return fraud_service.model_status().get("version")


def _date_arg(name, end=False):
    raw = request.args.get(name, "").strip()
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(raw)
    except ValueError:
        flash(f"Ignored invalid date '{raw}'.", "warning")
        return None
    return d + timedelta(days=1) - timedelta(seconds=1) if end and len(raw) == 10 else d


def _page():
    return request.args.get("page", 1, type=int)


# ------------------------------------------------------------------ datasets
@bp.route("/datasets")
@login_required
def datasets():
    rows = FinancialDataset.query.order_by(FinancialDataset.uploaded_date.desc()).all()
    return render_template("shared/datasets.html", datasets=rows)


@bp.route("/datasets/<int:dataset_id>")
@login_required
def dataset_detail(dataset_id):
    ds = db.session.get(FinancialDataset, dataset_id) or abort(404)
    preview = ds.transactions.order_by(T.transaction_date).limit(15).all()
    v = _version()
    return render_template("shared/dataset_detail.html", ds=ds, preview=preview,
                           stats=dataset_service.dataset_stats(ds.id),
                           summary=stats_service.summary(ds.id, v), form=ActionForm())


@bp.route("/datasets/<int:dataset_id>/transactions")
@login_required
def transactions(dataset_id):
    ds = db.session.get(FinancialDataset, dataset_id) or abort(404)
    q = request.args.get("q", "").strip()
    ttype = request.args.get("type", "").strip()
    start, end = _date_arg("from"), _date_arg("to", end=True)
    v = _version()
    query = (db.session.query(T, A).outerjoin(A, and_(A.transaction_id == T.id, A.model_version == v))
             .filter(T.dataset_id == ds.id))
    if q:
        bank, acct = graph_service.parse_account(q)
        query = query.filter(or_(T.transaction_reference.ilike(f"%{q}%"), T.sender_account == acct,
                                 T.receiver_account == acct))
    if ttype:
        query = query.filter(T.transaction_type == ttype)
    if start:
        query = query.filter(T.transaction_date >= start)
    if end:
        query = query.filter(T.transaction_date <= end)
    page = query.order_by(T.transaction_date, T.id).paginate(page=_page(), per_page=25, error_out=False)
    types = [r[0] for r in db.session.query(T.transaction_type).filter(T.dataset_id == ds.id).distinct().all()]
    return render_template("shared/transactions.html", ds=ds, page=page, q=q, ttype=ttype, types=sorted(types),
                           start=request.args.get("from", ""), end=request.args.get("to", ""))


@bp.route("/transactions/<int:tx_id>")
@login_required
def transaction_detail(tx_id):
    tx = db.session.get(T, tx_id) or abort(404)
    v = _version()
    analysis = A.query.filter_by(transaction_id=tx.id, model_version=v).first() if v else None
    snapshot = fraud_service.feature_snapshot(tx) if analysis is not None else None
    return render_template("shared/transaction_detail.html", tx=tx, analysis=analysis, snapshot=snapshot,
                           form=ActionForm())


@bp.route("/transactions/<int:tx_id>/analyze", methods=["POST"])
@login_required
def analyze_transaction(tx_id):
    if not ActionForm().validate_on_submit():
        abort(400)
    tx = db.session.get(T, tx_id) or abort(404)
    try:
        analysis, created, info = fraud_service.analyze_transaction(tx, current_user)
    except fraud_service.ModelUnavailable as exc:
        flash(f"Fraud detection is unavailable: {exc}", "danger")
        return redirect(url_for("analysis.transaction_detail", tx_id=tx_id))
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
        return redirect(url_for("analysis.transaction_detail", tx_id=tx_id))
    if created:
        flash(f"Analysed with {analysis.model_name}: {analysis.anomaly_status} ({analysis.risk_level} risk).",
              "success")
        if info and info["coverage"] != "full":
            flash(info["message"], "warning")
    else:
        flash("This transaction was already analysed with the current model version; showing the stored result.",
              "info")
    return redirect(url_for("analysis.transaction_detail", tx_id=tx_id))


# ------------------------------------------------------------------ results
@bp.route("/datasets/<int:dataset_id>/results")
@login_required
def results(dataset_id):
    ds = db.session.get(FinancialDataset, dataset_id) or abort(404)
    v = _version()
    risk = request.args.get("risk", "")
    status = request.args.get("status", "Suspicious")
    query = db.session.query(A, T).join(T, A.transaction_id == T.id).filter(T.dataset_id == ds.id)
    if v:
        query = query.filter(A.model_version == v)
    if risk in RISK_LEVELS:
        query = query.filter(A.risk_level == risk)
    if status in ("Normal", "Suspicious"):
        query = query.filter(A.anomaly_status == status)
    page = query.order_by(A.combined_anomaly_score.desc()).paginate(page=_page(), per_page=25, error_out=False)
    return render_template("shared/results.html", ds=ds, page=page, risk_filter=risk, status=status,
                           summary=stats_service.summary(ds.id, v), charts=stats_service.charts(ds.id, v),
                           evaluation=stats_service.labelled_evaluation(ds.id, v))


# ------------------------------------------------------------------ graph
@bp.route("/graph")
@login_required
def graph():
    all_ds = FinancialDataset.query.order_by(FinancialDataset.uploaded_date.desc()).all()
    ds_id = request.args.get("dataset", type=int) or (all_ds[0].id if all_ds else None)
    account = request.args.get("account", "").strip()
    v = _version()
    connected = graph_service.connected_accounts(ds_id, v, account) if ds_id and account else []
    rels = graph_service.suspicious_relationships(ds_id, v, limit=25) if ds_id else []
    return render_template("shared/graph.html", datasets=all_ds, ds_id=ds_id, account=account,
                           connected=connected, relationships=rels)


@bp.route("/api/graph")
@login_required
def graph_data():
    ds_id = request.args.get("dataset", type=int)
    if not ds_id or db.session.get(FinancialDataset, ds_id) is None:
        return jsonify({"error": "Unknown dataset."}), 404
    account = request.args.get("account", "").strip() or None
    fig = graph_service.network_figure(ds_id, _version(), account)
    if fig is None:
        return jsonify({"error": "Enter an account as BANK|ACCOUNT or ACCOUNT."}), 400
    return jsonify(fig)


# ------------------------------------------------------------------ alerts
@bp.route("/alerts")
@login_required
def alerts():
    risk, status = request.args.get("risk", ""), request.args.get("status", "")
    ds_id = request.args.get("dataset", type=int)
    start, end = _date_arg("from"), _date_arg("to", end=True)
    query = db.session.query(FraudAlert, A, T).join(A, FraudAlert.analysis_id == A.id).join(T, A.transaction_id == T.id)
    if risk in RISK_LEVELS:
        query = query.filter(FraudAlert.risk_level == risk)
    if status in ("New", "Reviewed", "Closed"):
        query = query.filter(FraudAlert.status == status)
    if ds_id:
        query = query.filter(T.dataset_id == ds_id)
    if start:
        query = query.filter(FraudAlert.created_date >= start)
    if end:
        query = query.filter(FraudAlert.created_date <= end)
    page = query.order_by(FraudAlert.created_date.desc(), A.combined_anomaly_score.desc()).paginate(
        page=_page(), per_page=25, error_out=False)
    return render_template("shared/alerts.html", page=page, risk_filter=risk, status=status, ds_id=ds_id,
                           datasets=FinancialDataset.query.all(), start=request.args.get("from", ""),
                           end=request.args.get("to", ""))


@bp.route("/alerts/<int:alert_id>")
@login_required
def alert_detail(alert_id):
    alert = db.session.get(FraudAlert, alert_id) or abort(404)
    form = AlertReviewForm(obj=alert) if current_user.is_admin else None
    return render_template("shared/alert_detail.html", alert=alert, a=alert.analysis, tx=alert.analysis.transaction,
                           form=form)


# ------------------------------------------------------------------ reports
@bp.route("/reports", methods=["GET", "POST"])
@login_required
def reports():
    form = ReportForm()
    all_ds = FinancialDataset.query.order_by(FinancialDataset.uploaded_date.desc()).all()
    form.dataset_id.choices = [(0, "All datasets")] + [(d.id, d.dataset_name) for d in all_ds]
    if request.method == "GET" and request.args.get("dataset", type=int):
        form.dataset_id.data = request.args.get("dataset", type=int)
    if form.validate_on_submit():
        try:
            rep = report_service.generate(form.dataset_id.data or None, form.report_format.data, current_user,
                                          _version())
            flash(f"{rep.report_format} report generated.", "success")
            return redirect(url_for("analysis.download_report", report_id=rep.id))
        except report_service.ReportError as exc:
            flash(str(exc), "warning")
    q = InvestigationReport.query
    if not current_user.is_admin:
        q = q.filter_by(generated_by=current_user.id)
    return render_template("shared/reports.html", form=form,
                           reports=q.order_by(InvestigationReport.generated_date.desc()).limit(50).all())


@bp.route("/reports/<int:report_id>/download")
@login_required
def download_report(report_id):
    rep = db.session.get(InvestigationReport, report_id) or abort(404)
    if not current_user.is_admin and rep.generated_by != current_user.id:
        abort(403)
    path = report_service.report_path(rep)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


# ------------------------------------------------------------------ model comparison
@bp.route("/model")
@login_required
def model_info():
    st = fraud_service.model_status()
    meta = None
    if st["ok"]:
        meta = fraud_service.get_engine()[0].metadata
    ds_id = request.args.get("dataset", type=int)
    v = st.get("version")
    return render_template("shared/model.html", status=st, meta=meta,
                           datasets=FinancialDataset.query.all(), ds_id=ds_id,
                           charts=stats_service.charts(ds_id, v) if st["ok"] else {},
                           evaluation=stats_service.labelled_evaluation(ds_id, v) if st["ok"] else None)
