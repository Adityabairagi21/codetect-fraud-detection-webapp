"""Admin-only pages: users, dataset management, running detection, alert review."""
from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import or_

from database.models import FinancialDataset, FinancialTransaction, FraudAlert, User, now
from extensions import db
from forms import ActionForm, AlertReviewForm, UploadForm
from routes import admin_required
from services import dataset_service, fraud_service, stats_service

bp = Blueprint("admin", __name__, url_prefix="/admin")


def _version():
    st = fraud_service.model_status()
    return st.get("version")


@bp.route("/dashboard")
@admin_required
def dashboard():
    ds_id = request.args.get("dataset", type=int)
    v = _version()
    return render_template("admin/dashboard.html", summary=stats_service.summary(ds_id, v),
                           charts=stats_service.charts(ds_id, v), datasets=FinancialDataset.query.all(),
                           selected=ds_id, users=User.query.count())


# ------------------------------------------------------------------ users
@bp.route("/users")
@admin_required
def users():
    q = request.args.get("q", "").strip()
    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(User.name.ilike(like), User.email.ilike(like), User.mobile_number.ilike(like)))
    page = query.order_by(User.registered_date.desc()).paginate(page=request.args.get("page", 1, type=int),
                                                                per_page=25, error_out=False)
    return render_template("admin/users.html", page=page, q=q, form=ActionForm())


@bp.route("/users/<int:user_id>/toggle", methods=["POST"])
@admin_required
def toggle_user(user_id):
    if not ActionForm().validate_on_submit():
        abort(400)
    user = db.session.get(User, user_id) or abort(404)
    if user.id == current_user.id:
        flash("You cannot deactivate your own account.", "warning")
    else:
        user.status = "Inactive" if user.status == "Active" else "Active"
        db.session.commit()
        flash(f"{user.email} is now {user.status.lower()}.", "success")
    return redirect(request.referrer or url_for("admin.users"))


# ------------------------------------------------------------------ datasets
@bp.route("/datasets/upload", methods=["GET", "POST"])
@admin_required
def upload():
    form = UploadForm()
    result = None
    if form.validate_on_submit():
        try:
            engine = fraud_service.get_engine()[0]
        except fraud_service.ModelUnavailable:
            engine = None
        result = dataset_service.save_upload(form.file.data, form.dataset_name.data, form.source_name.data,
                                             current_user, engine)
        if result.ok:
            for w in result.warnings:
                flash(w, "warning")
            flash(f"Dataset '{result.dataset.dataset_name}' uploaded: {result.dataset.total_records:,} "
                  "valid transactions stored.", "success")
            return redirect(url_for("analysis.dataset_detail", dataset_id=result.dataset.id))
    return render_template("admin/upload.html", form=form, result=result, help=dataset_service.REQUIRED_HELP)


@bp.route("/datasets/<int:dataset_id>/delete", methods=["POST"])
@admin_required
def delete_dataset(dataset_id):
    if not ActionForm().validate_on_submit():
        abort(400)
    ds = db.session.get(FinancialDataset, dataset_id) or abort(404)
    ok, msg = dataset_service.delete_dataset(ds)
    flash(msg, "success" if ok else "danger")
    return redirect(url_for("analysis.datasets") if ok else url_for("analysis.dataset_detail", dataset_id=dataset_id))


@bp.route("/datasets/<int:dataset_id>/detect", methods=["POST"])
@admin_required
def run_detection(dataset_id):
    if not ActionForm().validate_on_submit():
        abort(400)
    ds = db.session.get(FinancialDataset, dataset_id) or abort(404)
    try:
        s = fraud_service.analyze_dataset(ds, current_user)
    except fraud_service.ModelUnavailable as exc:
        flash(f"Fraud detection is unavailable: {exc}", "danger")
        return redirect(url_for("analysis.dataset_detail", dataset_id=dataset_id))
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
        return redirect(url_for("analysis.dataset_detail", dataset_id=dataset_id))
    msg = (f"Analysed {s.analyzed:,} transactions: {s.suspicious:,} suspicious, {s.alerts_created:,} alerts created."
           if s.analyzed else "Every transaction was already analysed with the current model version.")
    if s.skipped_existing and s.analyzed:
        msg += f" {s.skipped_existing:,} were already analysed and were skipped."
    flash(msg, "success")
    flash(s.history["message"], "info" if s.history.get("coverage") == "full" else "warning")
    return redirect(url_for("analysis.results", dataset_id=dataset_id))


# ------------------------------------------------------------------ alerts
@bp.route("/alerts/<int:alert_id>/review", methods=["POST"])
@admin_required
def review_alert(alert_id):
    alert = db.session.get(FraudAlert, alert_id) or abort(404)
    form = AlertReviewForm()
    if form.validate_on_submit():
        alert.status = form.status.data
        alert.review_remarks = (form.review_remarks.data or "").strip() or None
        alert.reviewed_by = current_user.id
        alert.reviewed_date = now()
        db.session.commit()
        flash(f"Alert #{alert.id} saved as {alert.status}.", "success")
    else:
        flash("The review could not be saved: " + "; ".join(e for errs in form.errors.values() for e in errs), "danger")
    return redirect(url_for("analysis.alert_detail", alert_id=alert_id))
