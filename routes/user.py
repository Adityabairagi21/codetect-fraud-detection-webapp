"""Financial-analyst (User role) pages."""
from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from database.models import FinancialDataset, FinancialTransaction, FraudAnalysis
from extensions import db
from services import fraud_service, stats_service

bp = Blueprint("user", __name__, url_prefix="/user")


@bp.route("/dashboard")
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for("admin.dashboard"))
    ds_id = request.args.get("dataset", type=int)
    v = fraud_service.model_status().get("version")
    return render_template("user/dashboard.html", summary=stats_service.summary(ds_id, v),
                           charts=stats_service.charts(ds_id, v), datasets=FinancialDataset.query.all(),
                           selected=ds_id)


@bp.route("/history")
@login_required
def history():
    page = (db.session.query(FraudAnalysis, FinancialTransaction)
            .join(FinancialTransaction, FraudAnalysis.transaction_id == FinancialTransaction.id)
            .filter(FraudAnalysis.analyzed_by == current_user.id)
            .order_by(FraudAnalysis.analyzed_date.desc(), FraudAnalysis.combined_anomaly_score.desc())
            .paginate(page=request.args.get("page", 1, type=int), per_page=25, error_out=False))
    return render_template("user/history.html", page=page)
