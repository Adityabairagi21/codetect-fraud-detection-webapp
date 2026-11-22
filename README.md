# CoDetect – Flask web application (Stage 2)

This is the web application for **CoDetect – Financial Fraud Detection With Anomaly Feature Detection**. It lets you:
* upload financial transaction datasets;
* score them with the **two saved Isolation Forest models** from Stage 1 (transaction-only and graph-enhanced);
* see Low / Medium / High / Critical risk levels and the top abnormal features;
* explore the transaction network;
* manage fraud alerts;
* download CSV and PDF reports.

The app never trains models and never re-implements feature logic. `services/feature_engineering.py`, `modeling.py`
and `fraud_inference.py` are **byte-identical copies** of the Stage 1 files. A test checks their checksums.

## Architecture

```
CoDetect_Flask_App/
├── app.py                  app factory, error pages, CLI (init-db, create-admin)
├── config.py               settings from .env (no secrets in code)
├── extensions.py           SQLAlchemy, Flask-Login, CSRF
├── forms.py                Flask-WTF forms (validation + CSRF)
├── database/
│   ├── models.py           7 tables: users, financial_datasets, financial_transactions, fraud_analysis,
│   │                       anomaly_features, fraud_alerts, investigation_reports
│   └── setup_mysql.sql     one-time database + user creation
├── routes/                 Blueprints: auth, admin (admin-only), user, analysis (shared pages)
├── services/
│   ├── fraud_service.py    loads artifacts, history context, runs analyses, stores results and alerts
│   ├── dataset_service.py  upload validation (Stage 1 cleaning rules), storage, safe deletion
│   ├── graph_service.py    bounded NetworkX subgraphs -> Plotly, connected accounts, suspicious relationships
│   ├── stats_service.py    dashboard aggregates from MySQL
│   ├── report_service.py   CSV / PDF reports (reportlab)
│   └── feature_engineering.py, modeling.py, fraud_inference.py   (copied from Stage 1, unchanged)
├── templates/  static/     Bootstrap 5 UI, Plotly.js charts
├── ml_artifacts/           saved models + metadata + history context (from Stage 1)
├── data/sample_uploads/    real IBM rows from the evaluation period, ready to upload
├── scripts/sync_ml_artifacts.py   copies Stage 1 code + artifacts, builds samples
├── uploads/  reports/      runtime files
└── tests/                  pytest (41 tests)
```

**How scoring works.** Each transaction's features use only transactions from the previous 24 hours
(strictly earlier minutes). The 24h context before a dataset's first transaction comes from
`ml_artifacts/history_context.joblib`, which holds real cleaned IBM transactions from Sep 8–10 2022 (no labels). After that,
the dataset's own earlier transactions are used. Every analysis reports whether the context was full, partial or
missing. The `Is Laundering` column, if uploaded, is stored for display and evaluation only.

**Data contract additions beyond the FRD columns.**
* Transactions also store bank codes, currencies and the received amount. The models need these: an account id is only
  unique within its bank, and amounts must be converted to USD.
* Analyses store `model_version`, with a unique constraint on (transaction, version) to prevent duplicates.
* Alerts store the reviewer and review date.

## Setup (Windows PowerShell)

Prerequisites: Python 3.12, MySQL 8.0, and the Stage 1 project at `..\CoDetect_Financial_Fraud_Detection` with its notebook already run.

```powershell
cd C:\Users\ASUS\CoDetect_Flask_App
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\sync_ml_artifacts.py           # copy Stage 1 code + models, build history context + samples
```

### MySQL
1. Copy `.env.example` to `.env`. Set `SECRET_KEY` (`python -c "import secrets; print(secrets.token_hex(32))"`)
   and pick a database password.
2. Put the same password in place of `CHANGE_ME` in `database\setup_mysql.sql`. Then run it as root, either in MySQL
   Workbench (*File → Open SQL Script → Execute*) or with:
   ```powershell
   & "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe" -u root -p -e "source database/setup_mysql.sql"
   ```
3. Create the tables and the first admin. The password is prompted and never stored in code:
   ```powershell
   flask --app app init-db
   flask --app app create-admin
   ```

### Run
```powershell
flask --app app run
```
Open http://127.0.0.1:5000. Analysts register themselves at `/register`.

## Demonstration workflow
1. **Sign in as admin.** Go to *Upload dataset* and upload `data/sample_uploads/ibm_hi_small_2022-09-09_1000-1030.csv`
   (12,962 real transactions).
2. **Run detection.** On the dataset page choose **Run fraud detection**. Both models score every transaction, and
   High/Critical results create alerts.
3. **Review results.** *Results* shows both models' scores, risk levels, top abnormal indicators and a check against
   the uploaded labels.
4. **Explore the network.** *Transaction network* shows suspicious edges; click an account to open its connections.
5. **Handle alerts.** *Fraud alerts*: filter by risk and date, open an alert, add investigation remarks, and mark it Reviewed or Closed.
6. **Download reports.** *Reports* generates PDF and CSV files.
7. **Try the analyst view.** Register an analyst. Analysts can search transactions, run **Analyse transaction**, view alerts, history and
   reports, but cannot upload, delete or change alert status. This is enforced on the server.

## Screenshots
`screenshots/` holds 20 screenshots from the real running app on MySQL. They cover the login, upload, detection
results, dashboards, network, alerts and reviews, transaction detail, reports, model comparison, users, the analyst
views, the access-denied page and the mobile layout. A sample PDF and CSV report are included.
They were produced by `scripts/capture_screenshots.py`, which drives the demo workflow in your installed Chrome.

## Tests
```powershell
venv\Scripts\python.exe -m pytest tests -q
```
The tests use SQLite (same SQLAlchemy models) for speed, with the real saved models and real IBM sample rows. They cover:
* authentication, roles and CSRF;
* upload validation (missing columns, bad type, empty file, unknown currency, duplicate references);
* search;
* artifact loading and the missing-model case;
* score parity with the notebook and the single-transaction vs dataset-run check;
* duplicate prevention;
* alerts and reviews, and dependency-checked deletion;
* the graph API;
* CSV/PDF reports and download permissions.

To run them against MySQL, set `TEST_DATABASE_URL` to a separate, empty test database.

## Security
* **Passwords and sessions.** Werkzeug password hashing, Flask-Login sessions, inactive accounts blocked.
* **Role checks.** Done on the server (`admin_required`).
* **Forms.** CSRF on every form and POST action.
* **Uploads.** Extension, size and row limits; `secure_filename` plus a random prefix; parsing errors are shown as messages.
* **Downloads.** Report downloads resolve only inside `reports/`, with an ownership check.
* **Model files.** Joblib files are loaded only from `ml_artifacts/` and checked against their SHA-256. Uploaded files are never unpickled.
* **Errors and secrets.** Error pages show no internals, and secrets live only in `.env`.

## Troubleshooting
* **"Fraud detection is unavailable".** Run `python scripts\sync_ml_artifacts.py` (it needs the Stage 1 notebook outputs).
* **`Access denied for user 'codetect_app'`.** The password in `.env` doesn't match the one in the setup SQL.
* **`cryptography` error on connect.** Run `pip install -r requirements.txt` (MySQL 8 authentication needs it).
* **"No exchange rate for currencies".** The upload uses a currency the models were never trained with.

## Repository notes
- **Stage 1 (research notebook, features and models):** https://github.com/Adityabairagi21/codetect-fraud-detection
- `ml_artifacts/` contains the trained models copied from Stage 1 by `scripts/sync_ml_artifacts.py`, so the app runs
  without re-training. Secrets (`.env`, demo credentials) are not committed. Create your own with the steps above.
