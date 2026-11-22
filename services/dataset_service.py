"""
Dataset upload, validation, storage and deletion.

Validation reuses the Stage 1 cleaning code (services.feature_engineering),
so an uploaded file is judged by exactly the rules the models were trained with.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from flask import current_app
from sqlalchemy import func, insert
from werkzeug.utils import secure_filename

from database.models import FinancialDataset, FinancialTransaction, FraudAlert, FraudAnalysis, InvestigationReport
from extensions import db
from services import feature_engineering as fe

# Accepted header layouts: the raw IBM AML layout, or canonical snake_case names.
CANONICAL_MAP = {c: c for c in fe.IBM_AML_COLUMN_MAP}
REFERENCE_ALIASES = ("transaction_reference", "Transaction Reference", "Transaction ID", "transaction_id")
REQUIRED_HELP = ("Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency, "
                 "Amount Paid, Payment Currency, Payment Format (IBM AML layout). Optional: Is Laundering, "
                 "Transaction Reference.")


@dataclass
class UploadResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset: FinancialDataset | None = None
    report: dict = field(default_factory=dict)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/XLSX as text so bank codes keep their leading zeros."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return pd.read_excel(path, dtype=str, engine="openpyxl")


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Map either accepted layout to canonical columns. Raises fe.SchemaError."""
    raw = raw.rename(columns=lambda c: str(c).strip())
    ref = next((c for c in REFERENCE_ALIASES if c in raw.columns), None)
    column_map = CANONICAL_MAP if "sender_account" in raw.columns else fe.IBM_AML_COLUMN_MAP
    df = fe.standardize_columns(raw, column_map)
    if ref:
        df["transaction_reference"] = raw[ref].astype(str).str.strip()
    return df


def validate(raw: pd.DataFrame, engine=None):
    """Returns (clean_df, report, errors, warnings)."""
    errors, warnings = [], []
    max_rows = current_app.config["MAX_DATASET_ROWS"]
    if len(raw) == 0:
        return None, {}, ["The file contains no transaction rows."], warnings
    if len(raw) > max_rows:
        return None, {}, [f"The file has {len(raw):,} rows; the limit is {max_rows:,}."], warnings
    try:
        canon = to_canonical(raw)
    except fe.SchemaError as exc:
        return None, {}, [f"{exc}. Expected columns: {REQUIRED_HELP}"], warnings
    if "transaction_reference" in canon:
        refs = canon["transaction_reference"]
        if refs.eq("").any() or refs.isna().any():
            errors.append("Some rows have an empty transaction reference.")
        elif refs.duplicated().any():
            errors.append(f"{int(refs.duplicated().sum())} transaction references are duplicated.")
        elif refs.str.len().max() > 64:
            errors.append("Transaction references must be at most 64 characters.")
    if errors:
        return None, {}, errors, warnings
    clean, report = fe.clean_transactions(canon, reference_prefix="TX")
    if report["dropped_invalid"]:
        warnings.append(f"{report['dropped_invalid']:,} invalid rows were skipped "
                        f"(timestamp {report['invalid_timestamp']}, amount {report['invalid_amount']}, "
                        f"account id {report['invalid_account_id']}, missing category {report['missing_category']}).")
    if report["dropped_exact_duplicates"]:
        warnings.append(f"{report['dropped_exact_duplicates']:,} exact duplicate rows were skipped.")
    if clean.empty:
        errors.append("No valid transactions remain after validation.")
    elif engine is not None:
        errors.extend(engine.validate(clean))
    elif engine is None:
        warnings.append("ML models are not loaded, so currency compatibility could not be checked.")
    if "label" in clean and clean["label"].isna().any():
        warnings.append("Some 'Is Laundering' values are not 0/1; they are stored as empty.")
    return (None if errors else clean), report, errors, warnings


def save_upload(file_storage, dataset_name: str, source_name: str, user, engine=None) -> UploadResult:
    filename = secure_filename(file_storage.filename or "")
    if not filename or not allowed_file(filename):
        return UploadResult(False, ["Only .csv and .xlsx files are accepted."])
    stored = current_app.config["UPLOAD_FOLDER"] / f"{uuid.uuid4().hex}_{filename}"
    file_storage.save(stored)
    try:
        raw = read_table(stored)
    except Exception:  # noqa: BLE001 - any parser failure is a user-facing validation error
        stored.unlink(missing_ok=True)
        return UploadResult(False, ["The file could not be read as CSV/Excel."])
    clean, report, errors, warnings = validate(raw, engine)
    if errors:
        stored.unlink(missing_ok=True)
        return UploadResult(False, errors, warnings, report=report)

    ds = FinancialDataset(dataset_name=dataset_name.strip(), source_name=source_name.strip(),
                          file_path=stored.name, total_records=len(clean), uploaded_by=user.id)
    db.session.add(ds)
    db.session.flush()
    rows = [{
        "dataset_id": ds.id, "transaction_reference": r.transaction_reference,
        "transaction_date": r.timestamp.to_pydatetime(), "sender_bank": r.sender_bank,
        "sender_account": r.sender_account, "receiver_bank": r.receiver_bank,
        "receiver_account": r.receiver_account, "transaction_amount": float(r.amount_paid),
        "payment_currency": r.payment_currency, "amount_received": float(r.amount_received),
        "receiving_currency": r.receiving_currency, "transaction_type": r.payment_format,
        "actual_label": (None if "label" not in clean or pd.isna(getattr(r, "label", None)) else int(r.label)),
    } for r in clean.itertuples(index=False)]
    for i in range(0, len(rows), 5000):
        db.session.execute(insert(FinancialTransaction), rows[i:i + 5000])
    db.session.commit()
    return UploadResult(True, [], warnings, ds, report)


def dataset_frame(dataset_id: int, start=None, end=None) -> pd.DataFrame:
    """Transactions of a dataset as a cleaned canonical frame (+ db id), ready for scoring."""
    q = db.session.query(FinancialTransaction).filter(FinancialTransaction.dataset_id == dataset_id)
    if start is not None:
        q = q.filter(FinancialTransaction.transaction_date >= start)
    if end is not None:
        q = q.filter(FinancialTransaction.transaction_date <= end)
    rows = q.with_entities(
        FinancialTransaction.id, FinancialTransaction.transaction_reference, FinancialTransaction.transaction_date,
        FinancialTransaction.sender_bank, FinancialTransaction.sender_account, FinancialTransaction.receiver_bank,
        FinancialTransaction.receiver_account, FinancialTransaction.amount_received,
        FinancialTransaction.receiving_currency, FinancialTransaction.transaction_amount,
        FinancialTransaction.payment_currency, FinancialTransaction.transaction_type,
        FinancialTransaction.actual_label).all()
    df = pd.DataFrame(rows, columns=["db_id", "transaction_reference", "timestamp", "sender_bank", "sender_account",
                                     "receiver_bank", "receiver_account", "amount_received", "receiving_currency",
                                     "amount_paid", "payment_currency", "payment_format", "label"])
    if df.empty:
        return df
    ids = df.set_index("transaction_reference")["db_id"]
    clean, _ = fe.clean_transactions(df.drop(columns="db_id"), reference_prefix="TX")
    clean["db_id"] = clean["transaction_reference"].map(ids).to_numpy()
    return clean


def dataset_stats(dataset_id: int) -> dict:
    T = FinancialTransaction
    base = db.session.query(T).filter(T.dataset_id == dataset_id)
    agg = base.with_entities(func.count(T.id), func.min(T.transaction_date), func.max(T.transaction_date),
                             func.count(func.distinct(T.sender_account)), func.count(func.distinct(T.receiver_account)),
                             func.sum(T.actual_label), func.count(T.actual_label)).one()
    by_type = dict(base.with_entities(T.transaction_type, func.count(T.id)).group_by(T.transaction_type).all())
    by_cur = dict(base.with_entities(T.payment_currency, func.count(T.id)).group_by(T.payment_currency).all())
    return {"rows": agg[0], "first": agg[1], "last": agg[2], "senders": agg[3], "receivers": agg[4],
            "labelled_laundering": int(agg[5] or 0), "labelled_rows": agg[6], "by_type": by_type,
            "by_currency": by_cur}


def delete_dataset(ds: FinancialDataset) -> tuple[bool, str]:
    """Delete only when no alert of this dataset is still open (New / Reviewed)."""
    open_alerts = (db.session.query(func.count(FraudAlert.id))
                   .join(FraudAnalysis).join(FinancialTransaction)
                   .filter(FinancialTransaction.dataset_id == ds.id, FraudAlert.status != "Closed").scalar())
    if open_alerts:
        return False, (f"Cannot delete: {open_alerts} fraud alert(s) from this dataset are still open. "
                       "Close them first.")
    tx_ids = db.session.query(FinancialTransaction.id).filter(FinancialTransaction.dataset_id == ds.id)
    an_ids = db.session.query(FraudAnalysis.id).filter(FraudAnalysis.transaction_id.in_(tx_ids))
    from database.models import AnomalyFeature
    db.session.query(AnomalyFeature).filter(AnomalyFeature.analysis_id.in_(an_ids)).delete(synchronize_session=False)
    db.session.query(FraudAlert).filter(FraudAlert.analysis_id.in_(an_ids)).delete(synchronize_session=False)
    db.session.query(FraudAnalysis).filter(FraudAnalysis.transaction_id.in_(tx_ids)).delete(synchronize_session=False)
    db.session.query(FinancialTransaction).filter(FinancialTransaction.dataset_id == ds.id).delete(synchronize_session=False)
    db.session.query(InvestigationReport).filter(InvestigationReport.dataset_id == ds.id).update(
        {"dataset_id": None}, synchronize_session=False)
    path = current_app.config["UPLOAD_FOLDER"] / Path(ds.file_path).name
    name = ds.dataset_name
    db.session.delete(ds)
    db.session.commit()
    path.unlink(missing_ok=True)
    return True, f"Dataset '{name}' and its analyses were deleted."
