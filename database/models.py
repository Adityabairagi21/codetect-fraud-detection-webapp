"""
CoDetect - MySQL schema (SQLAlchemy). The seven logical tables from the FRD.

Additions beyond the FRD column lists, each needed by the ML contract:
* Financial_Transactions keeps sender/receiver bank, currencies and the
  received amount, because the saved models need them (an account id is only
  unique within a bank, and amounts must be converted to USD).
* Fraud_Analysis stores model_version (to prevent duplicate analyses of the same
  transaction by the same model version) and analyzed_by.
* Anomaly_Features stores the rank (1-3) of each abnormal feature.
* Fraud_Alerts stores reviewed_by / reviewed_date for the audit trail.
* Investigation_Reports stores dataset_id and report_format.
Raw training data and model artifacts stay in files, not in MySQL.
"""
from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db

USER_TYPES = ("Admin", "User")
USER_STATUSES = ("Active", "Inactive")
DATASET_STATUSES = ("Uploaded", "Analyzed")
RISK_LEVELS = ("Low", "Medium", "High", "Critical")
ANOMALY_STATUSES = ("Normal", "Suspicious")
ALERT_STATUSES = ("New", "Reviewed", "Closed")


def now():
    return datetime.now().replace(microsecond=0)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), nullable=False, unique=True, index=True)
    mobile_number = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    user_type = db.Column(db.Enum(*USER_TYPES, name="user_type"), nullable=False, default="User")
    registered_date = db.Column(db.DateTime, nullable=False, default=now)
    status = db.Column(db.Enum(*USER_STATUSES, name="user_status"), nullable=False, default="Active")

    datasets = db.relationship("FinancialDataset", back_populates="uploader")
    reports = db.relationship("InvestigationReport", back_populates="author")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.user_type == "Admin"

    @property
    def is_active(self):                    # Flask-Login: inactive users cannot sign in
        return self.status == "Active"


class FinancialDataset(db.Model):
    __tablename__ = "financial_datasets"
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(150), nullable=False)
    source_name = db.Column(db.String(150), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    total_records = db.Column(db.Integer, nullable=False, default=0)
    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    uploaded_date = db.Column(db.DateTime, nullable=False, default=now)
    status = db.Column(db.Enum(*DATASET_STATUSES, name="dataset_status"), nullable=False, default="Uploaded")

    uploader = db.relationship("User", back_populates="datasets")
    transactions = db.relationship("FinancialTransaction", back_populates="dataset", lazy="dynamic")


class FinancialTransaction(db.Model):
    __tablename__ = "financial_transactions"
    __table_args__ = (
        db.UniqueConstraint("dataset_id", "transaction_reference", name="uq_dataset_reference"),
        db.Index("ix_tx_dataset_date", "dataset_id", "transaction_date"),
    )
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("financial_datasets.id", ondelete="RESTRICT"), nullable=False)
    transaction_reference = db.Column(db.String(64), nullable=False, index=True)
    transaction_date = db.Column(db.DateTime, nullable=False)
    sender_bank = db.Column(db.String(20), nullable=False)
    sender_account = db.Column(db.String(40), nullable=False, index=True)
    receiver_bank = db.Column(db.String(20), nullable=False)
    receiver_account = db.Column(db.String(40), nullable=False, index=True)
    transaction_amount = db.Column(db.Float, nullable=False)          # amount paid, payment currency
    payment_currency = db.Column(db.String(40), nullable=False)
    amount_received = db.Column(db.Float, nullable=False)
    receiving_currency = db.Column(db.String(40), nullable=False)
    transaction_type = db.Column(db.String(40), nullable=False, index=True)   # payment format
    actual_label = db.Column(db.SmallInteger, nullable=True)          # evaluation/display only

    dataset = db.relationship("FinancialDataset", back_populates="transactions")
    analyses = db.relationship("FraudAnalysis", back_populates="transaction", lazy="dynamic")

    @property
    def sender_id(self):
        return f"{self.sender_bank}|{self.sender_account}"

    @property
    def receiver_id(self):
        return f"{self.receiver_bank}|{self.receiver_account}"


class FraudAnalysis(db.Model):
    __tablename__ = "fraud_analysis"
    __table_args__ = (db.UniqueConstraint("transaction_id", "model_version", name="uq_analysis_tx_version"),)
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey("financial_transactions.id", ondelete="CASCADE"),
                               nullable=False, index=True)
    model_name = db.Column(db.String(60), nullable=False)
    model_version = db.Column(db.String(40), nullable=False)
    transaction_anomaly_score = db.Column(db.Float, nullable=False)
    graph_anomaly_score = db.Column(db.Float, nullable=False)
    combined_anomaly_score = db.Column(db.Float, nullable=False, index=True)
    risk_level = db.Column(db.Enum(*RISK_LEVELS, name="risk_level"), nullable=False, index=True)
    anomaly_status = db.Column(db.Enum(*ANOMALY_STATUSES, name="anomaly_status"), nullable=False, index=True)
    analyzed_date = db.Column(db.DateTime, nullable=False, default=now, index=True)
    analyzed_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    transaction = db.relationship("FinancialTransaction", back_populates="analyses")
    features = db.relationship("AnomalyFeature", back_populates="analysis", cascade="all, delete-orphan",
                               order_by="AnomalyFeature.rank")
    alert = db.relationship("FraudAlert", back_populates="analysis", uselist=False, cascade="all, delete-orphan")


class AnomalyFeature(db.Model):
    __tablename__ = "anomaly_features"
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("fraud_analysis.id", ondelete="CASCADE"), nullable=False, index=True)
    rank = db.Column(db.SmallInteger, nullable=False)
    feature_name = db.Column(db.String(60), nullable=False, index=True)
    feature_value = db.Column(db.String(60), nullable=False)
    contribution_score = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(400), nullable=False)

    analysis = db.relationship("FraudAnalysis", back_populates="features")


class FraudAlert(db.Model):
    __tablename__ = "fraud_alerts"
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("fraud_analysis.id", ondelete="CASCADE"),
                            nullable=False, unique=True)
    alert_message = db.Column(db.String(400), nullable=False)
    risk_level = db.Column(db.Enum(*RISK_LEVELS, name="alert_risk_level"), nullable=False, index=True)
    created_date = db.Column(db.DateTime, nullable=False, default=now, index=True)
    status = db.Column(db.Enum(*ALERT_STATUSES, name="alert_status"), nullable=False, default="New", index=True)
    review_remarks = db.Column(db.Text, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_date = db.Column(db.DateTime, nullable=True)

    analysis = db.relationship("FraudAnalysis", back_populates="alert")
    reviewer = db.relationship("User")


class InvestigationReport(db.Model):
    __tablename__ = "investigation_reports"
    id = db.Column(db.Integer, primary_key=True)
    generated_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("financial_datasets.id", ondelete="SET NULL"), nullable=True)
    report_type = db.Column(db.String(60), nullable=False)
    report_format = db.Column(db.Enum("CSV", "PDF", name="report_format"), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    generated_date = db.Column(db.DateTime, nullable=False, default=now)

    author = db.relationship("User", back_populates="reports")
    dataset = db.relationship("FinancialDataset")
