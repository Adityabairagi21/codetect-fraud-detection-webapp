"""
CoDetect - shared model-side logic: preprocessing, score normalisation,
risk classification and abnormal-feature explanations.

Training (notebook) and inference (fraud_inference.py / Flask) both use these
functions, so a score means the same thing everywhere.

Score conventions
-----------------
* raw anomaly score   = -IsolationForest.score_samples(X). Higher = more abnormal.
* anomaly score (0-100) = percentile of the raw score within the TRAINING-period
  raw scores (empirical CDF x 100). 99 means "more abnormal than 99% of
  training transactions". It is a rarity rank, NOT a fraud probability.
* risk level comes from fixed percentile cut-points of that score (see
  RISK_LEVELS). Suspicious = High or Critical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from services.feature_engineering import FEATURE_SCHEMA, GRAPH_FEATURES, TRANSACTION_FEATURES

MODEL_FEATURES = {
    "transaction_only": list(TRANSACTION_FEATURES),
    "graph_enhanced": list(TRANSACTION_FEATURES) + list(GRAPH_FEATURES),
}
MODEL_LABELS = {"transaction_only": "Transaction-only IF", "graph_enhanced": "Graph-enhanced IF"}
_SCHEMA = {f["name"]: f for f in FEATURE_SCHEMA}

# Percentile cut-points of the 0-100 anomaly score (training distribution).
# Chosen as an alert budget, not tuned on labels: ~3% of ordinary traffic is
# flagged Suspicious (High + Critical) and 0.5% Critical.
RISK_LEVELS = [("Low", 0.0), ("Medium", 90.0), ("High", 97.0), ("Critical", 99.5)]
SUSPICIOUS_LEVELS = ("High", "Critical")
N_SCORE_QUANTILES = 2001
REFERENCE_SAMPLE_SIZE = 100_000


# ================================================================ preprocessing
# Candidate encodings of the categorical / binary inputs, chosen per model on
# validation PR-AUC. Added after the one-hot-only version showed (validation
# data) that Isolation Forest isolates rare one-hot / flag values first:
# rare currencies, cross-currency payments and self-transfers filled the top
# of the ranking, while labelled laundering is common-looking on exactly
# those columns.
#   onehot       : one-hot categoricals, binary flags as 0/1 (the original design)
#   frequency    : each categorical value -> its training frequency; flags as 0/1
#   numeric_only : amount + history/graph features only (no categoricals, no flags)
ENCODINGS = ("onehot", "frequency", "numeric_only")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its relative frequency in the training data (unseen -> 0)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.maps_ = {c: X[c].astype(str).value_counts(normalize=True).to_dict() for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return np.column_stack([X[c].astype(str).map(self.maps_[c]).fillna(0.0).to_numpy(dtype="float64")
                                for c in self.columns_])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_frequency" for c in self.columns_], dtype=object)


def input_features(features: list[str], encoding: str) -> list[str]:
    """Features the model actually consumes under ``encoding``."""
    if encoding == "numeric_only":
        return [f for f in features if _SCHEMA[f]["kind"] == "numeric"]
    return list(features)


def build_preprocessor(features: list[str], encoding: str = "onehot") -> ColumnTransformer:
    """log1p + standardise skewed numerics, standardise others, encode categoricals.

    Columns not listed are dropped, so the same feature frame can be passed
    under every encoding. Unknown categories at inference map to all-zeros
    (one-hot) or frequency 0.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}")
    used = input_features(features, encoding)
    log_cols = [f for f in used if _SCHEMA[f]["transform"] == "log1p"]
    num_cols = [f for f in used if _SCHEMA[f]["kind"] in ("numeric", "binary")
                and _SCHEMA[f]["transform"] != "log1p"]
    cat_cols = [f for f in used if _SCHEMA[f]["kind"] == "categorical"]
    cat_enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False) if encoding == "onehot"
               else Pipeline([("freq", FrequencyEncoder()), ("scale", StandardScaler())]))
    parts = [("log_numeric", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                       ("scale", StandardScaler())]), log_cols),
             ("numeric", StandardScaler(), num_cols)]
    if cat_cols:
        parts.append(("categorical", cat_enc, cat_cols))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def fit_isolation_forest(X, n_estimators=200, max_samples=256, max_features=1.0, seed=42):
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            max_features=max_features, contamination="auto",
                            random_state=seed, n_jobs=-1)
    return model.fit(X)


def raw_anomaly_score(model: IsolationForest, X) -> np.ndarray:
    """Higher = more abnormal (sign-flipped score_samples)."""
    return -model.score_samples(X)


# ================================================================ normalisation
def fit_score_normalizer(train_raw_scores: np.ndarray) -> dict:
    q = np.linspace(0, 1, N_SCORE_QUANTILES)
    return {"probabilities": q.tolist(),
            "raw_score_quantiles": np.quantile(train_raw_scores, q).tolist(),
            "n_training_scores": int(len(train_raw_scores)),
            "method": "empirical CDF of training-period raw scores, x100"}


def normalize_scores(raw: np.ndarray, normalizer: dict) -> np.ndarray:
    """Map raw scores to 0-100 via the stored training quantiles (monotone non-decreasing)."""
    xq = np.asarray(normalizer["raw_score_quantiles"])
    pq = np.asarray(normalizer["probabilities"])
    xq, idx = np.unique(xq, return_index=True)  # interp needs strictly increasing x
    return 100.0 * np.interp(raw, xq, pq[idx], left=0.0, right=1.0)


def risk_level(score_0_100: np.ndarray) -> np.ndarray:
    names = np.array([n for n, _ in RISK_LEVELS], dtype=object)
    cuts = np.array([c for _, c in RISK_LEVELS[1:]])
    return names[np.searchsorted(cuts, np.asarray(score_0_100), side="right")]


def anomaly_status(levels) -> np.ndarray:
    return np.where(np.isin(levels, SUSPICIOUS_LEVELS), "Suspicious", "Normal")


# ================================================================ abnormal features
def fit_feature_reference(features_df: pd.DataFrame, feature_names: list[str], seed=42) -> dict:
    """Training reference distributions for the abnormal-feature analysis.

    numeric/binary -> sorted sample of training values; categorical -> frequencies.
    """
    rng = np.random.default_rng(seed)
    n = len(features_df)
    take = rng.choice(n, size=min(n, REFERENCE_SAMPLE_SIZE), replace=False)
    sample = features_df.iloc[np.sort(take)]
    ref = {}
    for f in feature_names:
        if _SCHEMA[f]["kind"] == "categorical":
            freq = features_df[f].value_counts(normalize=True)
            ref[f] = {"kind": "categorical", "frequencies": {str(k): float(v) for k, v in freq.items()},
                      "n": int(n)}
        else:
            vals = np.sort(sample[f].to_numpy(dtype="float64"))
            ref[f] = {"kind": "numeric", "sorted_sample": vals,
                      "median": float(np.median(vals)), "n": int(len(vals))}
    return ref


def feature_abnormality(features_df: pd.DataFrame, reference: dict, feature_names: list[str]):
    """Per-feature abnormality = -log10(tail probability of the value in training).

    numeric: tail = min(P(X <= x), P(X >= x)) under the training sample;
    categorical: tail = training frequency of the category.
    A score of 3 means "a value this extreme occurs in about 1 in 1,000 training
    transactions". Returns (abnormality DataFrame, direction DataFrame of
    'high'/'low'/'rare').
    """
    scores, direction = {}, {}
    for f in feature_names:
        r = reference[f]
        x = features_df[f]
        if r["kind"] == "categorical":
            p = x.astype(str).map(r["frequencies"]).fillna(0.0).to_numpy()
            tail = np.maximum(p, 1.0 / r["n"])
            direction[f] = np.full(len(x), "rare", dtype=object)
        else:
            s = r["sorted_sample"]
            v = x.to_numpy(dtype="float64")
            p_le = np.searchsorted(s, v, side="right") / len(s)
            p_ge = 1.0 - np.searchsorted(s, v, side="left") / len(s)
            tail = np.maximum(np.minimum(p_le, p_ge), 1.0 / len(s))
            direction[f] = np.where(v >= r["median"], "high", "low").astype(object)
        scores[f] = -np.log10(np.minimum(tail, 1.0))
    return (pd.DataFrame(scores, index=features_df.index),
            pd.DataFrame(direction, index=features_df.index))


_LABELS = {
    "amount_usd": "Transaction amount",
    "hour": "Transaction hour",
    "payment_format": "Payment format",
    "payment_currency": "Payment currency",
    "is_cross_currency": "Cross-currency flag",
    "is_self_transfer": "Self-transfer flag",
    "sender_out_count_24h": "Sender's payments in the last 24h",
    "receiver_in_count_24h": "Receiver's incoming payments in the last 24h",
    "sender_amount_deviation_24h": "Amount relative to the sender's recent payments",
    "sender_amount_std_24h": "Variability of the sender's recent amounts",
    "pair_count_24h": "Repeated sender-receiver payments in the last 24h",
    "sender_in_degree_24h": "Distinct accounts paying the sender (24h)",
    "sender_out_degree_24h": "Distinct accounts the sender paid (24h)",
    "receiver_in_degree_24h": "Distinct accounts paying the receiver (24h)",
    "receiver_out_degree_24h": "Distinct accounts the receiver paid (24h)",
    "sender_counterparties_24h": "Sender's connected counterparties (24h)",
    "receiver_counterparties_24h": "Receiver's connected counterparties (24h)",
    "sender_txn_count_24h": "Sender's total activity (24h)",
    "receiver_txn_count_24h": "Receiver's total activity (24h)",
    "sender_volume_24h": "Value flowing through the sender (24h)",
    "receiver_volume_24h": "Value flowing through the receiver (24h)",
}


def feature_label(name: str) -> str:
    return _LABELS.get(name, name)


def _fmt_value(name, v):
    if name in ("amount_usd", "sender_volume_24h", "receiver_volume_24h"):
        return f"USD {float(v):,.2f}"
    if name in ("sender_amount_deviation_24h", "sender_amount_std_24h"):
        return f"{float(v):+.2f} (log scale)" if name.endswith("deviation_24h") else f"{float(v):.2f} (log scale)"
    if isinstance(v, str):
        return v
    fv = float(v)
    return f"{int(fv):,}" if fv.is_integer() else f"{fv:,.2f}"


def explain(name, value, direction, abnormality) -> str:
    """Human-readable abnormal-indicator sentence (an indicator, not a proven cause)."""
    rarity = 10 ** (-abnormality)
    share = f"{100 * rarity:.2g}%" if rarity >= 0.0001 else "under 0.01%"
    label = feature_label(name)
    if direction == "rare":
        return f"{label} '{value}' is uncommon: it occurs in {share} of training transactions."
    word = "unusually high" if direction == "high" else "unusually low"
    side = "this high or higher" if direction == "high" else "this low or lower"
    return (f"{label} is {word} ({_fmt_value(name, value)}); "
            f"only {share} of training transactions are {side}.")


def top_abnormal_features(features_df, reference, feature_names, k=3) -> pd.DataFrame:
    """Long table: k most abnormal features per transaction (index = features_df index)."""
    ab, dr = feature_abnormality(features_df, reference, feature_names)
    a = ab.to_numpy()
    order = np.argsort(-a, axis=1, kind="stable")[:, :k]
    rows = []
    names = np.array(feature_names, dtype=object)
    for pos, idx in enumerate(features_df.index):
        for rank, j in enumerate(order[pos], start=1):
            f = names[j]
            val = features_df.at[idx, f]
            rows.append({"row": idx, "rank": rank, "feature_name": f,
                         "feature_group": _SCHEMA[f]["group"],
                         "feature_value": val, "contribution_score": float(a[pos, j]),
                         "direction": dr.iat[pos, j],
                         "description": explain(f, val, dr.iat[pos, j], a[pos, j])})
    return pd.DataFrame(rows)
