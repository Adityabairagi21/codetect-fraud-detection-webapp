"""
CoDetect Flask tests (SQLite for speed; the SQLAlchemy models are the same ones MySQL uses).
Uses the real saved Stage 1 models and real IBM sample rows.
"""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from conftest import CSV_SAMPLE, PASSWORD, ROOT, STAGE1, XLSX_SAMPLE, login, upload
from database.models import (AnomalyFeature, FinancialDataset, FinancialTransaction, FraudAlert, FraudAnalysis,
                             InvestigationReport, User)
from extensions import db


def q(app, model, **kw):
    with app.app_context():
        return db.session.query(model).filter_by(**kw).all()


# ------------------------------------------------------------------ authentication
def test_admin_and_user_login_redirect_to_their_dashboards(client):
    assert login(client, "admin@example.com").headers["Location"].endswith("/")
    assert client.get("/", follow_redirects=False).headers["Location"].endswith("/admin/dashboard")
    client.post("/logout")
    login(client, "analyst@example.com")
    assert client.get("/", follow_redirects=False).headers["Location"].endswith("/user/dashboard")


def test_wrong_password_and_inactive_user_are_refused(client, app):
    r = client.post("/login", data={"email": "admin@example.com", "password": "wrong"}, follow_redirects=True)
    assert b"Incorrect email or password" in r.data
    with app.app_context():
        u = db.session.query(User).filter_by(email="analyst@example.com").one()
        u.status = "Inactive"
        db.session.commit()
    r = client.post("/login", data={"email": "analyst@example.com", "password": PASSWORD}, follow_redirects=True)
    assert b"inactive" in r.data


def test_registration_creates_user_role_with_hashed_password(client, app):
    r = client.post("/register", data={"name": "New Analyst", "email": "new@example.com", "mobile_number": "",
                                       "password": "Secure123", "confirm": "Secure123"}, follow_redirects=True)
    assert b"Account created" in r.data
    u = q(app, User, email="new@example.com")[0]
    assert u.user_type == "User" and u.password_hash != "Secure123" and u.check_password("Secure123")
    r = client.post("/register", data={"name": "Again", "email": "new@example.com", "password": "Secure123",
                                       "confirm": "Secure123"})
    assert b"already exists" in r.data
    r = client.post("/register", data={"name": "Weak", "email": "weak@example.com", "password": "short", "confirm": "short"})
    assert b"between 8 and 128" in r.data and not q(app, User, email="weak@example.com")


def test_profile_update_and_password_change(analyst, app):
    analyst.post("/profile", data={"name": "Ravi K", "mobile_number": "+91 98765 43210"})
    assert q(app, User, email="analyst@example.com")[0].name == "Ravi K"
    r = analyst.post("/change-password", data={"current": "nope", "password": "NewPass123", "confirm": "NewPass123"})
    assert b"Current password is incorrect" in r.data
    analyst.post("/change-password", data={"current": PASSWORD, "password": "NewPass123", "confirm": "NewPass123"})
    assert q(app, User, email="analyst@example.com")[0].check_password("NewPass123")


def test_csrf_is_enforced_when_enabled(app):
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        c = app.test_client()
        r = c.post("/login", data={"email": "admin@example.com", "password": PASSWORD})
        assert r.status_code == 400 and b"Session expired" in r.data   # no CSRF token -> rejected
        assert c.get("/admin/dashboard").status_code == 302              # and not logged in
    finally:
        app.config["WTF_CSRF_ENABLED"] = False


# ------------------------------------------------------------------ role-based access
def test_anonymous_users_are_redirected_to_login(client):
    for url in ("/admin/dashboard", "/user/dashboard", "/datasets", "/alerts", "/reports", "/graph"):
        r = client.get(url)
        assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_analyst_cannot_use_admin_functions(analyst, app, uploaded):
    assert analyst.get("/admin/dashboard").status_code == 403
    assert analyst.get("/admin/users").status_code == 403
    assert analyst.get("/admin/datasets/upload").status_code == 403
    assert analyst.post(f"/admin/datasets/{uploaded}/delete").status_code == 403
    assert analyst.post(f"/admin/datasets/{uploaded}/detect").status_code == 403
    assert analyst.post("/admin/alerts/1/review", data={"status": "Closed"}).status_code == 403
    assert q(app, FinancialDataset, id=uploaded)


def test_admin_can_search_and_deactivate_users(admin, app):
    assert b"analyst@example.com" in admin.get("/admin/users?q=Ravi").data
    assert b"analyst@example.com" not in admin.get("/admin/users?q=nobody").data
    uid = q(app, User, email="analyst@example.com")[0].id
    admin.post(f"/admin/users/{uid}/toggle")
    assert q(app, User, id=uid)[0].status == "Inactive"


# ------------------------------------------------------------------ dataset upload / validation
def test_xlsx_upload_stores_real_transactions(app, uploaded):
    ds = q(app, FinancialDataset, id=uploaded)[0]
    rows = pd.read_excel(XLSX_SAMPLE, dtype=str)
    assert ds.total_records == len(rows)
    t = q(app, FinancialTransaction, dataset_id=uploaded)
    assert len(t) == len(rows)
    assert any(x.sender_bank.startswith("0") for x in t)          # leading zeros kept


def test_csv_upload_works(admin, app):
    r = upload(admin, CSV_SAMPLE, name="csv window")
    assert b"valid transactions stored" in r.data


@pytest.mark.parametrize("content,filename,message", [
    (b"Timestamp,From Bank,Account\n2022/09/09 10:00,1,A\n", "bad.csv", b"Missing required columns"),
    (b"just,a,header\n", "empty.csv", b"no transaction rows"),
    (b"hello", "notes.txt", b"Only .csv and .xlsx"),
])
def test_invalid_uploads_are_rejected(admin, app, content, filename, message):
    r = upload(admin, content=content, filename=filename)
    assert message in r.data
    assert not q(app, FinancialDataset)


def test_unknown_currency_and_duplicate_reference_are_rejected(admin, app):
    df = pd.read_csv(CSV_SAMPLE, dtype=str).head(50)
    bad = df.copy()
    bad["Payment Currency"] = "Doubloon"
    r = upload(admin, content=bad.to_csv(index=False).encode(), filename="cur.csv")
    assert b"No exchange rate" in r.data
    dup = df.copy()
    dup.loc[1, "Transaction Reference"] = dup.loc[0, "Transaction Reference"]
    r = upload(admin, content=dup.to_csv(index=False).encode(), filename="dup.csv")
    assert b"duplicated" in r.data
    assert not q(app, FinancialDataset)


# ------------------------------------------------------------------ searching
def test_transaction_search_and_filters(analyst, app, uploaded):
    t = q(app, FinancialTransaction, dataset_id=uploaded)[5]
    base = f"/datasets/{uploaded}/transactions"
    assert t.transaction_reference.encode() in analyst.get(f"{base}?q={t.transaction_reference}").data
    assert t.transaction_reference.encode() in analyst.get(f"{base}?q={t.sender_bank}|{t.sender_account}").data
    r = analyst.get(f"{base}?type={t.transaction_type}&from=2022-09-10&to=2022-09-10")
    assert r.status_code == 200 and t.transaction_type.encode() in r.data
    assert b"No transactions match" in analyst.get(f"{base}?from=2030-01-01").data


# ------------------------------------------------------------------ models and inference
def test_model_artifacts_load(app):
    from services.fraud_service import model_status
    with app.app_context():
        st = model_status(app)
    assert st["ok"] and st["version"].startswith("f2.0.0-")


def test_ml_code_is_an_unmodified_copy_of_stage1():
    manifest = json.loads((ROOT / "ml_artifacts" / "SYNC_MANIFEST.json").read_text())
    for f, digest in manifest["code"].items():
        assert hashlib.sha256((ROOT / "services" / f).read_bytes()).hexdigest() == digest
        if (STAGE1 / "services" / f).exists():
            assert (STAGE1 / "services" / f).read_bytes() == (ROOT / "services" / f).read_bytes()


def test_detection_stores_scores_matching_the_notebook(app, analyzed):
    """Real IBM test-period rows: Flask scores == scores the Stage 1 notebook produced for the same rows."""
    nb_path = STAGE1 / "outputs" / "reports" / "scored_validation_test.csv.gz"
    with app.app_context():
        rows = (db.session.query(FinancialTransaction.transaction_reference, FraudAnalysis)
                .join(FraudAnalysis).filter(FinancialTransaction.dataset_id == analyzed).all())
        got = pd.DataFrame([{"ref": r, "t": a.transaction_anomaly_score, "g": a.graph_anomaly_score,
                             "risk": a.risk_level, "status": a.anomaly_status, "comb": a.combined_anomaly_score}
                            for r, a in rows]).set_index("ref")
    assert len(got) == q(app, FinancialDataset, id=analyzed)[0].total_records
    assert got[["t", "g", "comb"]].apply(lambda s: s.between(0, 100)).all().all()
    assert set(got["status"]) <= {"Normal", "Suspicious"}
    assert ((got["risk"].isin(["High", "Critical"])) == (got["status"] == "Suspicious")).all()
    if not nb_path.exists():
        pytest.skip("Stage 1 scored results not available")
    nb = pd.read_csv(nb_path).set_index("transaction_reference").loc[got.index]
    np.testing.assert_allclose(got["t"], nb["transaction_anomaly_score"], atol=1e-6)
    np.testing.assert_allclose(got["g"], nb["graph_anomaly_score"], atol=1e-6)
    assert (got["risk"] == nb["risk_level"]).all()


def test_duplicate_analysis_is_prevented(admin, app, analyzed):
    n = len(q(app, FraudAnalysis))
    r = admin.post(f"/admin/datasets/{analyzed}/detect", follow_redirects=True)
    assert b"already analysed" in r.data
    assert len(q(app, FraudAnalysis)) == n
    tx = q(app, FinancialTransaction, dataset_id=analyzed)[0]
    r = admin.post(f"/transactions/{tx.id}/analyze", follow_redirects=True)
    assert b"already analysed" in r.data and len(q(app, FraudAnalysis)) == n


def test_single_transaction_analysis_equals_dataset_run(analyst, admin, app, uploaded):
    tx = sorted(q(app, FinancialTransaction, dataset_id=uploaded), key=lambda t: t.transaction_date)[-1]
    r = analyst.post(f"/transactions/{tx.id}/analyze", follow_redirects=True)
    assert b"Analysed with" in r.data
    single = q(app, FraudAnalysis, transaction_id=tx.id)[0]
    assert len(q(app, AnomalyFeature, analysis_id=single.id)) == 3       # explanation even if Normal
    with app.app_context():                                               # run the full dataset in another version slot
        db.session.query(AnomalyFeature).delete()
        db.session.query(FraudAlert).delete()
        db.session.query(FraudAnalysis).delete()
        db.session.commit()
    admin.post(f"/admin/datasets/{uploaded}/detect")
    full = q(app, FraudAnalysis, transaction_id=tx.id)[0]
    assert full.graph_anomaly_score == pytest.approx(single.graph_anomaly_score, abs=1e-9)
    assert full.transaction_anomaly_score == pytest.approx(single.transaction_anomaly_score, abs=1e-9)
    hist = analyst.get("/user/history")
    assert hist.status_code == 200


def test_transaction_detail_shows_scores_features_and_snapshot(analyst, app, analyzed):
    a = sorted(q(app, FraudAnalysis), key=lambda a: -a.combined_anomaly_score)[0]
    r = analyst.get(f"/transactions/{a.transaction_id}")
    assert b"Top abnormal features" in r.data and b"All transaction and graph features" in r.data
    assert b"not proof of fraud" in r.data


# ------------------------------------------------------------------ alerts
def test_alerts_created_for_high_and_critical_only(app, analyzed):
    analyses = q(app, FraudAnalysis)
    expected = sum(a.risk_level in ("High", "Critical") for a in analyses)
    alerts = q(app, FraudAlert)
    assert len(alerts) == expected and all(al.risk_level in ("High", "Critical") for al in alerts)
    assert all(al.status == "New" for al in alerts)
    sus = [a for a in analyses if a.anomaly_status == "Suspicious"]
    with app.app_context():
        assert all(db.session.query(AnomalyFeature).filter_by(analysis_id=a.id).count() == 3 for a in sus)


def test_admin_reviews_alert_user_cannot(admin, analyst, app, analyzed):
    alerts = q(app, FraudAlert)
    if not alerts:
        pytest.skip("this sample produced no alerts")
    al = alerts[0]
    r = admin.post(f"/admin/alerts/{al.id}/review", data={"status": "Reviewed", "review_remarks": "Checked counterparties."},
                   follow_redirects=True)
    assert b"saved as Reviewed" in r.data
    al2 = q(app, FraudAlert, id=al.id)[0]
    assert al2.status == "Reviewed" and al2.review_remarks == "Checked counterparties." and al2.reviewed_by
    assert admin.get("/alerts?risk=High&status=Reviewed").status_code == 200
    assert admin.get("/alerts?from=2000-01-01&to=2100-01-01").status_code == 200


def test_dataset_delete_requires_closed_alerts(admin, app, analyzed):
    alerts = q(app, FraudAlert)
    if alerts:
        r = admin.post(f"/admin/datasets/{analyzed}/delete", follow_redirects=True)
        assert b"still open" in r.data and q(app, FinancialDataset, id=analyzed)
        with app.app_context():
            db.session.query(FraudAlert).update({"status": "Closed"})
            db.session.commit()
    r = admin.post(f"/admin/datasets/{analyzed}/delete", follow_redirects=True)
    assert b"were deleted" in r.data
    assert not q(app, FinancialDataset) and not q(app, FinancialTransaction) and not q(app, FraudAnalysis)


# ------------------------------------------------------------------ graph
def test_graph_api_returns_bounded_network(analyst, app, analyzed):
    r = analyst.get(f"/api/graph?dataset={analyzed}")
    fig = r.get_json()
    assert r.status_code == 200 and "data" in fig or fig.get("empty")
    tx = q(app, FinancialTransaction, dataset_id=analyzed)[0]
    fig = analyst.get(f"/api/graph?dataset={analyzed}&account={tx.sender_id}").get_json()
    assert not fig["empty"] and fig["stats"]["transactions"] >= 1
    nodes = next(t for t in fig["data"] if t.get("name") == "accounts")
    assert tx.sender_id in nodes["customdata"]
    assert analyst.get("/api/graph?dataset=999").status_code == 404
    page = analyst.get(f"/graph?dataset={analyzed}&account={tx.sender_id}")
    assert b"Connected accounts" in page.data and tx.receiver_id.encode() in page.data


# ------------------------------------------------------------------ reports
def test_reports_csv_and_pdf_download(analyst, admin, app, analyzed):
    r = analyst.post("/reports", data={"dataset_id": analyzed, "report_format": "CSV"})
    assert r.status_code == 302
    d = analyst.get(r.headers["Location"])
    assert d.status_code == 200 and b"CoDetect fraud analysis report" in d.data
    r = analyst.post("/reports", data={"dataset_id": analyzed, "report_format": "PDF"})
    d = analyst.get(r.headers["Location"])
    assert d.status_code == 200 and d.data[:4] == b"%PDF"
    reps = q(app, InvestigationReport)
    assert len(reps) == 2
    # another analyst cannot download them
    with app.app_context():
        u = User(name="Other", email="other@example.com", user_type="User")
        u.set_password(PASSWORD)
        db.session.add(u)
        db.session.commit()
    other = app.test_client()
    login(other, "other@example.com")
    assert other.get(f"/reports/{reps[0].id}/download").status_code == 403
    assert admin.get(f"/reports/{reps[0].id}/download").status_code == 200


def test_report_without_analysis_gives_message(analyst, uploaded):
    r = analyst.post("/reports", data={"dataset_id": uploaded, "report_format": "PDF"}, follow_redirects=True)
    assert b"Run fraud detection" in r.data


# ------------------------------------------------------------------ pages render
@pytest.mark.parametrize("url", ["/admin/dashboard", "/datasets", "/alerts", "/reports", "/model", "/graph",
                                 "/user/history", "/profile", "/change-password", "/admin/users", "/admin/datasets/upload"])
def test_admin_pages_render(admin, analyzed, url):
    assert admin.get(url).status_code == 200


def test_dataset_pages_render(analyst, analyzed):
    for url in (f"/datasets/{analyzed}", f"/datasets/{analyzed}/results", f"/datasets/{analyzed}/results?risk=Low&status=Normal",
                f"/model?dataset={analyzed}", "/user/dashboard", f"/user/dashboard?dataset={analyzed}"):
        assert analyst.get(url).status_code == 200, url


def test_unknown_record_gives_404(analyst):
    assert analyst.get("/transactions/999999").status_code == 404
    assert analyst.get("/alerts/999999").status_code == 404


# ------------------------------------------------------------------ missing models
def test_missing_models_are_handled_gracefully(no_models_app):
    c = no_models_app.test_client()
    login(c, "admin@example.com")
    page = c.get("/admin/dashboard")
    assert page.status_code == 200 and b"Fraud detection is unavailable" in page.data
    upload(c)
    with no_models_app.app_context():
        ds = db.session.query(FinancialDataset).one()
    r = c.post(f"/admin/datasets/{ds.id}/detect", follow_redirects=True)
    assert b"Fraud detection is unavailable" in r.data and b"Missing model artifacts" in r.data
