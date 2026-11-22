"""
Copy the Stage 1 ML code + artifacts into the Flask app (run once after the notebook).

    venv\\Scripts\\python.exe scripts\\sync_ml_artifacts.py [path\\to\\CoDetect_Financial_Fraud_Detection]

* services/feature_engineering.py, modeling.py, fraud_inference.py are copied
  UNCHANGED from Stage 1 (Flask never re-implements feature logic). Their SHA-256
  values go into ml_artifacts/SYNC_MANIFEST.json, and a test checks them.
* models/* -> ml_artifacts/ (loaded read-only; users can never upload joblib files).
* ml_artifacts/history_context.joblib: cleaned transactions from 2022-09-08 00:00
  to the end of the data (HISTORY_COLUMNS only, no labels). It gives uploads from
  the evaluation period their true previous-24h context.
* data/sample_uploads/: real IBM rows from the TEST period (complete time windows),
  in the raw IBM column layout, ready to upload.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parents[1]
STAGE1 = Path(sys.argv[1]) if len(sys.argv) > 1 else APP.parent / "CoDetect_Financial_Fraud_Detection"
ML_FILES = ["feature_engineering.py", "modeling.py", "fraud_inference.py"]
HISTORY_FROM = "2022-09-08 00:00"
SAMPLES = [("ibm_hi_small_2022-09-09_1000-1030.csv", "2022-09-09 10:00", "2022-09-09 10:30"),
           ("ibm_hi_small_2022-09-10_1400-1415.xlsx", "2022-09-10 14:00", "2022-09-10 14:15")]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if not (STAGE1 / "models" / "model_metadata.json").exists():
        sys.exit(f"Stage 1 artifacts not found under {STAGE1}. Run the Stage 1 notebook first.")
    manifest = {"stage1_path": str(STAGE1), "code": {}, "artifacts": {}}
    for f in ML_FILES:
        shutil.copy2(STAGE1 / "services" / f, APP / "services" / f)
        manifest["code"][f] = sha256(APP / "services" / f)
    out = APP / "ml_artifacts"
    out.mkdir(exist_ok=True)
    for f in (STAGE1 / "models").iterdir():
        shutil.copy2(f, out / f.name)
        manifest["artifacts"][f.name] = sha256(out / f.name)
    print("copied ML code + artifacts")

    sys.path.insert(0, str(APP))
    from services import feature_engineering as fe  # noqa: E402  (the copied module)
    import joblib  # noqa: E402

    clean = pd.read_csv(STAGE1 / "data" / "processed" / "cleaned_transactions.csv.gz",
                        dtype={c: str for c in fe.ID_COLUMNS})
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], format="ISO8601")  # midnight rows are written as a bare date
    clean = clean[clean["split"] != "excluded"]
    clean["minute"] = (clean["timestamp"].astype("datetime64[s]").astype("int64") // 60).astype("int64")
    hist = clean.loc[clean["timestamp"] >= HISTORY_FROM, fe.HISTORY_COLUMNS].reset_index(drop=True)
    joblib.dump(hist, out / "history_context.joblib", compress=3)
    manifest["history_context"] = {"rows": len(hist), "from": HISTORY_FROM, "to": str(clean["timestamp"].max())}
    print(f"history context: {len(hist):,} rows")

    samples = APP / "data" / "sample_uploads"
    samples.mkdir(parents=True, exist_ok=True)
    inv = fe.IBM_AML_COLUMN_MAP
    for name, start, end in SAMPLES:
        rows = clean[(clean["timestamp"] >= start) & (clean["timestamp"] < end)]
        raw = pd.DataFrame({"Transaction Reference": rows["transaction_reference"],
                            **{raw_col: rows[canon] for canon, raw_col in inv.items()}})
        raw["Timestamp"] = rows["timestamp"].dt.strftime(fe.TIMESTAMP_FORMAT)
        raw = raw.rename(columns={"Account.1": "Account"})          # restore the raw duplicate header
        if name.endswith(".csv"):
            raw.to_csv(samples / name, index=False)
        else:
            raw.to_excel(samples / name, index=False)
        print(f"sample {name}: {len(raw):,} rows, {int(rows['label'].sum())} labelled laundering")
    (out / "SYNC_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("done")


if __name__ == "__main__":
    main()
