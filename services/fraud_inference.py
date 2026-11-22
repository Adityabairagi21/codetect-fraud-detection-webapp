"""
CoDetect - reusable inference module (used by the notebook's consistency check
and, in Stage 2, by Flask).

    from services.fraud_inference import FraudInference
    engine = FraudInference("models")            # loads + validates artifacts
    result = engine.score(raw_df)                # raw CSV-shaped DataFrame
    result.results        # one row per transaction: scores, risk level, status
    result.explanations   # top-3 abnormal features for Suspicious rows
    result.cleaning_report

The module never trains anything and never reads the laundering label
(if present it is passed through to ``results.actual_label`` for display only).
All feature logic comes from services.feature_engineering and all score logic
from services.modeling, the same code the notebook trained with.

History context: features need the 24 hours before each transaction. By
default the saved history tail (the last 24h of the training data) is used,
which is only valid for batches that start after it. For any other batch, pass
``history=`` with cleaned earlier transactions (e.g. from the database).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from services import feature_engineering as fe
from services import modeling as md

ARTIFACT_FILES = {
    "transaction_model": "transaction_model.joblib",
    "graph_model": "graph_model.joblib",
    "preprocessing": "preprocessing.joblib",
    "scoring_reference": "scoring_reference.joblib",
    "history_tail": "history_tail.joblib",
    "metadata": "model_metadata.json",
}
MODEL_FILE_KEYS = {"transaction_only": "transaction_model", "graph_enhanced": "graph_model"}


class ArtifactError(RuntimeError):
    """Model artifacts are missing, corrupted or incompatible with this code."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ScoreResult:
    results: pd.DataFrame
    features: pd.DataFrame
    explanations: pd.DataFrame
    cleaning_report: dict
    history_info: dict = field(default_factory=dict)


class FraudInference:
    def __init__(self, models_dir, verify_checksums: bool = True):
        self.dir = Path(models_dir)
        missing = [f for f in ARTIFACT_FILES.values() if not (self.dir / f).exists()]
        if missing:
            raise ArtifactError(f"Missing model artifacts in {self.dir}: {missing}. "
                                "Run the Stage 1 notebook to create them.")
        try:
            self.metadata = json.loads((self.dir / ARTIFACT_FILES["metadata"]).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"model_metadata.json unreadable: {exc}") from exc

        if self.metadata.get("feature_version") != fe.FEATURE_VERSION:
            raise ArtifactError(
                f"Artifacts were built with feature version {self.metadata.get('feature_version')}, "
                f"code is {fe.FEATURE_VERSION}. Re-run the notebook.")
        if verify_checksums:
            for key, fname in ARTIFACT_FILES.items():
                expected = self.metadata.get("artifact_sha256", {}).get(fname)
                if key != "metadata" and expected and sha256(self.dir / fname) != expected:
                    raise ArtifactError(f"{fname} does not match the checksum recorded at training time")

        try:
            self.models = {m: joblib.load(self.dir / ARTIFACT_FILES[k]) for m, k in MODEL_FILE_KEYS.items()}
            self.preprocessors = joblib.load(self.dir / ARTIFACT_FILES["preprocessing"])
            ref = joblib.load(self.dir / ARTIFACT_FILES["scoring_reference"])
            self.history = joblib.load(self.dir / ARTIFACT_FILES["history_tail"])
        except Exception as exc:  # noqa: BLE001 - surface any unpickling problem as one error type
            raise ArtifactError(f"Could not load model artifacts: {type(exc).__name__}") from exc
        self.fx = ref["fx"]
        self.normalizers = ref["normalizers"]
        self.feature_reference = ref["feature_reference"]
        self.selected_model = self.metadata["selected_model"]
        for m in md.MODEL_FEATURES:
            if list(self.metadata["model_features"][m]) != md.MODEL_FEATURES[m]:
                raise ArtifactError(f"Feature list for {m} differs from the code's schema")

    # ------------------------------------------------------------ validation
    def prepare(self, raw: pd.DataFrame, column_map=None, reference_prefix="TX"):
        """Map columns and clean. Raises SchemaError for missing columns."""
        std = fe.standardize_columns(raw, column_map or fe.IBM_AML_COLUMN_MAP)
        return fe.clean_transactions(std, reference_prefix=reference_prefix)

    def validate(self, clean: pd.DataFrame) -> list[str]:
        """Problems that would stop scoring, as user-facing messages (empty = OK)."""
        issues = []
        if clean.empty:
            issues.append("No valid transactions remain after cleaning.")
            return issues
        cur = set(clean["payment_currency"]) | set(clean["receiving_currency"])
        unknown = sorted(cur - set(self.fx["rates"]))
        if unknown:
            issues.append(f"No exchange rate for currencies: {', '.join(unknown)}.")
        return issues

    # ------------------------------------------------------------ scoring
    def score(self, data: pd.DataFrame, history: pd.DataFrame | None = None,
              already_clean: bool = False, column_map=None, explain: str = "suspicious",
              top_k: int = 3) -> ScoreResult:
        """Score transactions with both saved models.

        explain: "suspicious" (default), "all" or "none" - which rows get top-k
        abnormal-feature explanations (for the selected model's feature set).
        """
        if already_clean:
            clean, report = data, {"input_rows": len(data), "output_rows": len(data)}
        else:
            clean, report = self.prepare(data, column_map)
        issues = self.validate(clean)
        if issues:
            raise ValueError(" ".join(issues))

        start = int(clean["minute"].min())
        if history is None:
            if int(self.history["minute"].max()) >= start:
                raise fe.HistoryOverlapError(
                    "This batch starts before the end of the saved training history "
                    f"({self.metadata['history_tail']['to']}). Pass the transactions from the "
                    "24 hours before the batch as history.")
            history = self.history
        hist_in_window = history[history["minute"] >= start - fe.HISTORY_WINDOW_MINUTES]
        history_info = {"history_rows_used": int(len(hist_in_window)),
                        "window_minutes": fe.HISTORY_WINDOW_MINUTES,
                        "covers_full_window": bool(len(hist_in_window)) and
                        int(history["minute"].min()) <= start - fe.HISTORY_WINDOW_MINUTES}

        feats = fe.compute_features(clean, self.fx, hist_in_window)
        out = pd.DataFrame({
            "transaction_reference": clean["transaction_reference"].to_numpy(),
            "timestamp": clean["timestamp"].to_numpy(),
            "sender_id": clean["sender_id"].to_numpy(),
            "receiver_id": clean["receiver_id"].to_numpy(),
            "amount_usd": feats["amount_usd"].to_numpy(),
            "payment_format": clean["payment_format"].astype(str).to_numpy(),
        }, index=clean.index)
        for m in md.MODEL_FEATURES:
            X = self.preprocessors[m].transform(feats[md.MODEL_FEATURES[m]])
            raw = md.raw_anomaly_score(self.models[m], X)
            short = "transaction" if m == "transaction_only" else "graph"
            out[f"{short}_raw_score"] = raw
            out[f"{short}_anomaly_score"] = md.normalize_scores(raw, self.normalizers[m])
            out[f"{short}_risk_level"] = md.risk_level(out[f"{short}_anomaly_score"])
        sel = "transaction" if self.selected_model == "transaction_only" else "graph"
        out["model_name"] = md.MODEL_LABELS[self.selected_model]
        out["combined_anomaly_score"] = out[f"{sel}_anomaly_score"]
        out["risk_level"] = out[f"{sel}_risk_level"]
        out["anomaly_status"] = md.anomaly_status(out["risk_level"])
        if "label" in clean:
            out["actual_label"] = clean["label"].to_numpy()

        if explain == "none":
            expl = pd.DataFrame()
        else:
            rows = out.index if explain == "all" else out.index[out["anomaly_status"] == "Suspicious"]
            names = self.metadata["input_features"][self.selected_model]  # only what the model used
            expl = md.top_abnormal_features(feats.loc[rows], self.feature_reference, names, k=top_k)
            if len(expl):
                expl.insert(0, "transaction_reference", out.loc[expl["row"], "transaction_reference"].to_numpy())
        return ScoreResult(out, feats, expl, report, history_info)
