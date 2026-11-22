"""
CoDetect - leakage-safe data preparation and feature engineering.

This file is the single source of truth for how a transaction becomes a
feature vector. The notebook imports it for training and Flask imports the
same file for inference. Nothing here touches the laundering label.

Time rule (prevents leakage)
----------------------------
Timestamps have minute resolution, and many transactions share a minute
with no reliable order inside it. Every history feature of a transaction at
minute ``m`` is computed from transactions in the trailing 24-hour window

        [m - 1440, m - 1]   (minutes, both ends inclusive)

* future transactions never contribute;
* transactions in the same minute (including the transaction itself) never
  contribute, so results do not depend on file row order;
* a fixed-length window keeps feature distributions comparable across the
  train / validation / test periods. Cumulative "since the start of the data"
  counts grow with time and would make later periods look abnormal; see
  outputs/metrics/feature_dry_run.json.

History context
---------------
``compute_features(batch, fx, history)`` takes ``history``: cleaned
transactions strictly before the batch (only the last 24h are used). The
notebook runs one pass over the full time-ordered data with no history.
Inference passes the saved history tail, and the result is identical
(verified by tests/test_feature_engineering.py).

Account identity
----------------
An account id is only unique within a bank, so nodes are ``bank|account``.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

FEATURE_VERSION = "2.0.0"
HISTORY_WINDOW_MINUTES = 24 * 60
VARIANCE_FLOOR = 1e-10     # relative floor for windowed log-amount variance (rounding noise)

# ---------------------------------------------------------------- schema
# canonical name -> column name as pandas reads the IBM AML CSV
# (the raw header repeats "Account"; pandas renames the 2nd to "Account.1")
IBM_AML_COLUMN_MAP = {
    "timestamp": "Timestamp",
    "sender_bank": "From Bank",
    "sender_account": "Account",
    "receiver_bank": "To Bank",
    "receiver_account": "Account.1",
    "amount_received": "Amount Received",
    "receiving_currency": "Receiving Currency",
    "amount_paid": "Amount Paid",
    "payment_currency": "Payment Currency",
    "payment_format": "Payment Format",
    "label": "Is Laundering",            # evaluation only
}
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
ID_COLUMNS = ["sender_bank", "sender_account", "receiver_bank", "receiver_account"]
REQUIRED_COLUMNS = ["timestamp", *ID_COLUMNS, "amount_received", "receiving_currency",
                    "amount_paid", "payment_currency", "payment_format"]
OPTIONAL_COLUMNS = ["label", "transaction_reference"]
# minimal columns a history context must carry
HISTORY_COLUMNS = ["transaction_reference", "minute", "sender_id", "receiver_id",
                   "amount_paid", "payment_currency"]
BASE_CURRENCY = "US Dollar"
_ID_PATTERN = r"[A-Za-z0-9_\-]+"

# Feature schema. "transform" is applied by the model preprocessing pipeline,
# so the feature frame keeps human-readable values for explanations.
# "24h" = trailing window [m-1440, m-1] minutes.
FEATURE_SCHEMA = [
    # ---- transaction-level (Model 1 and Model 2)
    dict(name="amount_usd", group="transaction", kind="numeric", transform="log1p",
         description="Amount paid, converted to US Dollar with dataset-implied FX rates"),
    dict(name="hour", group="transaction", kind="numeric", transform=None,
         description="Hour of day (0-23) of the transaction"),
    dict(name="payment_format", group="transaction", kind="categorical", transform="onehot",
         description="Transaction type / payment format"),
    dict(name="payment_currency", group="transaction", kind="categorical", transform="onehot",
         description="Currency the sender paid in"),
    dict(name="is_cross_currency", group="transaction", kind="binary", transform=None,
         description="1 if paid and received currencies differ"),
    dict(name="is_self_transfer", group="transaction", kind="binary", transform=None,
         description="1 if sender and receiver are the same account"),
    dict(name="sender_out_count_24h", group="transaction", kind="numeric", transform="log1p",
         description="Transactions sent by the sender in the previous 24 hours"),
    dict(name="receiver_in_count_24h", group="transaction", kind="numeric", transform="log1p",
         description="Transactions received by the receiver in the previous 24 hours"),
    dict(name="sender_amount_deviation_24h", group="transaction", kind="numeric", transform=None,
         description="log(1+amount) minus the mean log(1+amount) of the sender's payments in the "
                     "previous 24 hours (0 when there are none)"),
    dict(name="sender_amount_std_24h", group="transaction", kind="numeric", transform=None,
         description="Std of log(1+amount) of the sender's payments in the previous 24 hours "
                     "(0 if fewer than 2)"),
    dict(name="pair_count_24h", group="transaction", kind="numeric", transform="log1p",
         description="Transactions from this sender to this receiver in the previous 24 hours"),
    # ---- graph-derived (Model 2 only)
    dict(name="sender_in_degree_24h", group="graph", kind="numeric", transform="log1p",
         description="Distinct accounts that paid the sender in the previous 24 hours"),
    dict(name="sender_out_degree_24h", group="graph", kind="numeric", transform="log1p",
         description="Distinct accounts the sender paid in the previous 24 hours"),
    dict(name="receiver_in_degree_24h", group="graph", kind="numeric", transform="log1p",
         description="Distinct accounts that paid the receiver in the previous 24 hours"),
    dict(name="receiver_out_degree_24h", group="graph", kind="numeric", transform="log1p",
         description="Distinct accounts the receiver paid in the previous 24 hours"),
    dict(name="sender_counterparties_24h", group="graph", kind="numeric", transform="log1p",
         description="Distinct accounts connected to the sender (either direction) in the previous 24 hours"),
    dict(name="receiver_counterparties_24h", group="graph", kind="numeric", transform="log1p",
         description="Distinct accounts connected to the receiver (either direction) in the previous 24 hours"),
    dict(name="sender_txn_count_24h", group="graph", kind="numeric", transform="log1p",
         description="Transactions involving the sender (sent + received) in the previous 24 hours"),
    dict(name="receiver_txn_count_24h", group="graph", kind="numeric", transform="log1p",
         description="Transactions involving the receiver (sent + received) in the previous 24 hours"),
    dict(name="sender_volume_24h", group="graph", kind="numeric", transform="log1p",
         description="USD volume through the sender (sent + received) in the previous 24 hours"),
    dict(name="receiver_volume_24h", group="graph", kind="numeric", transform="log1p",
         description="USD volume through the receiver (sent + received) in the previous 24 hours"),
]
TRANSACTION_FEATURES = [f["name"] for f in FEATURE_SCHEMA if f["group"] == "transaction"]
GRAPH_FEATURES = [f["name"] for f in FEATURE_SCHEMA if f["group"] == "graph"]
ALL_FEATURES = TRANSACTION_FEATURES + GRAPH_FEATURES


class SchemaError(ValueError):
    """Input does not contain the columns required for scoring."""


class FXError(ValueError):
    """A currency has no fitted exchange rate."""


class HistoryOverlapError(ValueError):
    """History context is not strictly earlier than the batch."""


# ================================================================ loading / cleaning
def load_transactions(path, column_map=None, nrows=None) -> pd.DataFrame:
    """Read a transaction CSV and rename columns to canonical names.

    Bank / account columns are read as text so leading zeros survive.
    """
    column_map = column_map or IBM_AML_COLUMN_MAP
    str_cols = {column_map[c]: str for c in ID_COLUMNS if c in column_map}
    df = pd.read_csv(path, dtype=str_cols, nrows=nrows)
    return standardize_columns(df, column_map)


def standardize_columns(df: pd.DataFrame, column_map=None) -> pd.DataFrame:
    column_map = column_map or IBM_AML_COLUMN_MAP
    missing = [f"{canon} (expected column '{raw}')" for canon, raw in column_map.items()
               if canon in REQUIRED_COLUMNS and raw not in df.columns]
    if missing:
        raise SchemaError("Missing required columns: " + ", ".join(missing))
    keep = {raw: canon for canon, raw in column_map.items() if raw in df.columns}
    out = df[list(keep)].rename(columns=keep)
    if "transaction_reference" in df.columns and "transaction_reference" not in out:
        out["transaction_reference"] = df["transaction_reference"]
    return out


def clean_transactions(df: pd.DataFrame, reference_prefix: str = "TX"):
    """Validate and normalise canonical-column transactions.

    Returns (clean_df, report). Rows are never invented or repaired: invalid
    rows are dropped and counted in the report. Adds sender_id / receiver_id
    (bank|account), minute (int minutes since epoch) and a stable
    transaction_reference (1-based data-row number in the input when absent).
    """
    df = df.copy()
    report = {"input_rows": int(len(df))}
    if "transaction_reference" not in df:
        df["transaction_reference"] = [f"{reference_prefix}-{i + 1:08d}" for i in range(len(df))]
    df["transaction_reference"] = df["transaction_reference"].astype(str)

    for c in ID_COLUMNS + ["receiving_currency", "payment_currency", "payment_format"]:
        df[c] = df[c].astype("string").str.strip()
    for c in ("sender_account", "receiver_account"):
        df[c] = df[c].str.upper()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
    for c in ("amount_paid", "amount_received"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_ts = df["timestamp"].isna()
    bad_amt = ~(np.isfinite(df["amount_paid"]) & np.isfinite(df["amount_received"])
                & (df["amount_paid"] > 0) & (df["amount_received"] > 0))
    bad_id = pd.Series(False, index=df.index)
    for c in ID_COLUMNS:
        bad_id |= ~df[c].fillna("").str.fullmatch(_ID_PATTERN)
    bad_cat = pd.Series(False, index=df.index)
    for c in ("receiving_currency", "payment_currency", "payment_format"):
        bad_cat |= df[c].fillna("").eq("")
    invalid = bad_ts | bad_amt | bad_id | bad_cat
    report.update(invalid_timestamp=int(bad_ts.sum()), invalid_amount=int(bad_amt.sum()),
                  invalid_account_id=int(bad_id.sum()), missing_category=int(bad_cat.sum()),
                  dropped_invalid=int(invalid.sum()))
    df = df[~invalid]

    if "label" in df:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    content_cols = [c for c in df.columns if c != "transaction_reference"]
    dup = df.duplicated(subset=content_cols, keep="first")
    report["dropped_exact_duplicates"] = int(dup.sum())
    df = df[~dup]

    df["sender_id"] = df["sender_bank"] + "|" + df["sender_account"]
    df["receiver_id"] = df["receiver_bank"] + "|" + df["receiver_account"]
    df["minute"] = (df["timestamp"].astype("datetime64[s]").astype("int64") // 60).astype("int64")
    df = df.sort_values(["minute", "transaction_reference"], kind="stable").reset_index(drop=True)
    report["output_rows"] = int(len(df))
    return df, report


# ================================================================ currency
def fit_fx_rates(tx: pd.DataFrame, base: str = BASE_CURRENCY) -> dict:
    """Estimate USD-per-unit rates from the data's own cross-currency rows.

    For every row where one side is ``base``, the ratio of the base amount to
    the other amount is one observation of that currency's rate; the median
    over all observations is the rate. Fit on the training window only.
    """
    x = tx[(tx["payment_currency"] != tx["receiving_currency"])]
    a = x[x["payment_currency"] == base]
    b = x[x["receiving_currency"] == base]
    obs = pd.concat([
        pd.DataFrame({"cur": a["receiving_currency"], "rate": a["amount_paid"] / a["amount_received"]}),
        pd.DataFrame({"cur": b["payment_currency"], "rate": b["amount_received"] / b["amount_paid"]}),
    ])
    g = obs.groupby("cur")["rate"]
    rates = {k: float(v) for k, v in g.median().items()}
    rates[base] = 1.0
    seen = set(tx["payment_currency"]) | set(tx["receiving_currency"])
    return {"base": base, "rates": rates,
            "n_observations": {k: int(v) for k, v in g.size().items()},
            "currencies_without_rate": sorted(seen - set(rates)),
            "method": "median of base-currency/other-currency amount ratio over cross-currency rows"}


def to_usd(amount, currency, fx: dict) -> np.ndarray:
    rates = fx["rates"]
    cur = pd.Series(currency).astype(str)
    unknown = sorted(set(cur.unique()) - set(rates))
    if unknown:
        raise FXError(f"No exchange rate for currencies: {unknown}")
    return np.asarray(amount, dtype="float64") * cur.map(rates).to_numpy(dtype="float64")


# ================================================================ history context
def history_tail(tx: pd.DataFrame, end_minute: int | None = None) -> pd.DataFrame:
    """Rows needed as history for a batch starting right after ``end_minute``.

    That is the last 24h: minutes [end_minute - 1439, end_minute]. Only
    HISTORY_COLUMNS are kept (no labels).
    """
    end = int(tx["minute"].max()) if end_minute is None else int(end_minute)
    m = (tx["minute"] > end - HISTORY_WINDOW_MINUTES) & (tx["minute"] <= end)
    return tx.loc[m, HISTORY_COLUMNS].reset_index(drop=True)


# ================================================================ windowed aggregation helpers
_SPAN = np.int64(1) << np.int64(32)   # room for relative minute positions inside a group key


def _prefix_table(group: np.ndarray, pos: np.ndarray, values: np.ndarray):
    """Sort by (group, pos), sum duplicates, return keys + inclusive per-group prefix sums."""
    key = group.astype("int64") * _SPAN + pos.astype("int64")
    order = np.argsort(key, kind="stable")
    key, values = key[order], values[order]
    ukey, start = np.unique(key, return_index=True)
    sums = np.add.reduceat(values, start, axis=0) if len(key) else values[:0]
    ugroup = ukey // _SPAN
    cum = pd.DataFrame(sums).groupby(ugroup).cumsum().to_numpy()
    return ukey, ugroup, cum


def _asof(table, q_group: np.ndarray, q_pos: np.ndarray) -> np.ndarray:
    """Prefix value at the last entry with same group and pos <= q_pos (0 if none)."""
    ukey, ugroup, cum = table
    if len(ukey) == 0:
        return np.zeros((len(q_group), cum.shape[1] if cum.ndim == 2 else 1))
    q = q_group.astype("int64") * _SPAN + q_pos.astype("int64")
    idx = np.searchsorted(ukey, q, side="right") - 1
    safe = np.clip(idx, 0, None)
    ok = (idx >= 0) & (ugroup[safe] == q_group)
    out = cum[safe].astype("float64")
    out[~ok] = 0.0
    return out


def _window_sum(table, q_group, q_pos):
    """Sum over positions [q_pos - W, q_pos - 1]."""
    return _asof(table, q_group, q_pos - 1) - _asof(table, q_group, q_pos - HISTORY_WINDOW_MINUTES - 1)


def _active_partner_table(acct, partner, pos):
    """Delta table whose prefix at p = number of partners seen in [p - W, p - 1].

    A partner seen at minutes t1 < t2 < ... is "active" for queries in
    [t + 1, t + W]. Overlapping intervals are merged into runs; each run
    adds +1 at its start and -1 one minute after its end.
    """
    keep = acct != partner                                   # self-loops are not counterparties
    d = pd.DataFrame({"a": acct[keep], "p": partner[keep], "t": pos[keep]}).drop_duplicates()
    d = d.sort_values(["a", "p", "t"], kind="stable")
    a, p, t = d["a"].to_numpy(), d["p"].to_numpy(), d["t"].to_numpy()
    same = np.zeros(len(d), bool)
    same[1:] = (a[1:] == a[:-1]) & (p[1:] == p[:-1])
    gap = np.zeros(len(d), "int64")
    gap[1:] = t[1:] - t[:-1]
    run_start = ~same | (gap > HISTORY_WINDOW_MINUTES)
    run_end = np.ones(len(d), bool)
    run_end[:-1] = run_start[1:]
    g = np.concatenate([a[run_start], a[run_end]])
    q = np.concatenate([t[run_start] + 1, t[run_end] + HISTORY_WINDOW_MINUTES + 1])
    v = np.concatenate([np.ones(run_start.sum()), -np.ones(run_end.sum())]).reshape(-1, 1)
    return _prefix_table(g, q, v)


# ================================================================ features
def compute_features(tx: pd.DataFrame, fx: dict, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature matrix for cleaned transactions (one row per input row, same index).

    Uses only (a) the transaction's own attributes and (b) transactions in
    ``history`` or ``tx`` within the trailing 24h window [m-1440, m-1].
    The ``label`` column, if present, is never read.
    """
    missing = [c for c in REQUIRED_COLUMNS + ["sender_id", "receiver_id", "minute"] if c not in tx]
    if missing:
        raise SchemaError(f"compute_features needs cleaned input; missing {missing}")
    n = len(tx)
    if n == 0:
        return pd.DataFrame(columns=ALL_FEATURES, index=tx.index)

    batch_start = int(tx["minute"].min())
    ctx = tx[HISTORY_COLUMNS[1:]]
    if history is not None and len(history):
        hmiss = [c for c in HISTORY_COLUMNS[1:] if c not in history]
        if hmiss:
            raise SchemaError(f"history is missing columns {hmiss}")
        if int(history["minute"].max()) >= batch_start:
            raise HistoryOverlapError(
                "History must end strictly before the first minute of the batch")
        h = history[history["minute"] >= batch_start - HISTORY_WINDOW_MINUTES]
        ctx = pd.concat([h[HISTORY_COLUMNS[1:]], ctx], ignore_index=True)
    nb = len(ctx) - n                         # batch rows are the last n rows of ctx

    amount = to_usd(ctx["amount_paid"], ctx["payment_currency"], fx)
    log_amt = np.log1p(amount)
    codes, _ = pd.factorize(pd.concat([ctx["sender_id"], ctx["receiver_id"]], ignore_index=True))
    N = len(ctx)
    s, r = codes[:N].astype("int64"), codes[N:].astype("int64")
    origin = int(ctx["minute"].min()) - 2 * HISTORY_WINDOW_MINUTES - 2
    pos = ctx["minute"].to_numpy(dtype="int64") - origin

    # --- per-account event sums (each transaction = 1 out-event + 1 in-event)
    acct = np.concatenate([s, r])
    is_out = np.concatenate([np.ones(N), np.zeros(N)])
    amt2 = np.concatenate([amount, amount])
    lg = np.concatenate([log_amt, np.zeros(N)])
    vals = np.column_stack([is_out, 1 - is_out, amt2 * is_out, amt2 * (1 - is_out), lg, lg ** 2])
    acc_tab = _prefix_table(acct, np.concatenate([pos, pos]), vals)
    OUT_N, IN_N, OUT_AMT, IN_AMT, LSUM, LSQ = range(6)

    sb, rb, pb = s[nb:], r[nb:], pos[nb:]
    S = _window_sum(acc_tab, sb, pb)
    R = _window_sum(acc_tab, rb, pb)

    # --- distinct counterparties active in the window
    out_tab = _active_partner_table(s, r, pos)
    in_tab = _active_partner_table(r, s, pos)
    any_tab = _active_partner_table(acct, np.concatenate([r, s]), np.concatenate([pos, pos]))
    deg = {name: (_asof(tab, sb, pb)[:, 0], _asof(tab, rb, pb)[:, 0])
           for name, tab in (("out", out_tab), ("in", in_tab), ("any", any_tab))}

    # --- sender -> receiver pair count in the window
    pair_codes, _ = pd.factorize(s * np.int64(len(codes) + 1) + r)
    pair_tab = _prefix_table(pair_codes, pos, np.ones((N, 1)))
    pair_cnt = _window_sum(pair_tab, pair_codes[nb:], pb)[:, 0]

    out_n = S[:, OUT_N]
    has = out_n > 0
    mean_log = np.divide(S[:, LSUM], out_n, out=np.zeros(n), where=has)
    var_log = np.divide(S[:, LSQ], out_n, out=np.zeros(n), where=has) - mean_log ** 2
    # E[x^2]-E[x]^2 over windowed running sums leaves ~1e-14 rounding noise when all
    # amounts are equal; that noise depends on how much earlier history was summed.
    # Variances below this floor (std of log-amount < ~0.001%) are treated as zero.
    var_log = np.where(var_log < VARIANCE_FLOOR * (1.0 + mean_log ** 2), 0.0, var_log)
    std_log = np.where(out_n >= 2, np.sqrt(var_log), 0.0)

    f = pd.DataFrame(index=tx.index)
    f["amount_usd"] = amount[nb:]
    f["hour"] = tx["timestamp"].dt.hour.to_numpy()
    f["payment_format"] = tx["payment_format"].astype(str).to_numpy()
    f["payment_currency"] = tx["payment_currency"].astype(str).to_numpy()
    f["is_cross_currency"] = (tx["payment_currency"] != tx["receiving_currency"]).astype("int64").to_numpy()
    f["is_self_transfer"] = (sb == rb).astype("int64")
    f["sender_out_count_24h"] = out_n
    f["receiver_in_count_24h"] = R[:, IN_N]
    f["sender_amount_deviation_24h"] = np.where(has, log_amt[nb:] - mean_log, 0.0)
    f["sender_amount_std_24h"] = std_log
    f["pair_count_24h"] = pair_cnt
    f["sender_in_degree_24h"] = deg["in"][0]
    f["sender_out_degree_24h"] = deg["out"][0]
    f["receiver_in_degree_24h"] = deg["in"][1]
    f["receiver_out_degree_24h"] = deg["out"][1]
    f["sender_counterparties_24h"] = deg["any"][0]
    f["receiver_counterparties_24h"] = deg["any"][1]
    f["sender_txn_count_24h"] = S[:, OUT_N] + S[:, IN_N]
    f["receiver_txn_count_24h"] = R[:, OUT_N] + R[:, IN_N]
    f["sender_volume_24h"] = S[:, OUT_AMT] + S[:, IN_AMT]
    f["receiver_volume_24h"] = R[:, OUT_AMT] + R[:, IN_AMT]
    return f[ALL_FEATURES]


# ================================================================ NetworkX graph
def build_transaction_graph(tx: pd.DataFrame, fx: dict) -> nx.MultiDiGraph:
    """Directed multigraph: accounts are nodes, every transaction is its own edge.

    Use on a bounded window - one edge per transaction.
    """
    amt = to_usd(tx["amount_paid"], tx["payment_currency"], fx)
    G = nx.MultiDiGraph()
    G.add_edges_from(
        (s, r, {"reference": ref, "minute": int(m), "amount_usd": float(a), "payment_format": pf})
        for s, r, ref, m, a, pf in zip(tx["sender_id"], tx["receiver_id"], tx["transaction_reference"],
                                       tx["minute"], amt, tx["payment_format"]))
    return G


def window_graph(tx: pd.DataFrame, fx: dict, minute: int) -> nx.MultiDiGraph:
    """History graph a transaction at ``minute`` is scored against: [minute-1440, minute-1]."""
    m = (tx["minute"] >= minute - HISTORY_WINDOW_MINUTES) & (tx["minute"] < minute)
    return build_transaction_graph(tx[m], fx)


def graph_features_networkx(G: nx.MultiDiGraph, sender: str, receiver: str) -> dict:
    """Reference implementation of the windowed graph features on a history graph.

    Used to cross-check the vectorised computation. Self-loops are excluded
    from distinct-counterparty counts but included in transaction counts and
    volume (a self-transfer is both sent and received).
    """
    def node(a):
        if a not in G:
            return dict(in_degree=0, out_degree=0, unique=0, txn=0, volume=0.0)
        pred, succ = set(G.predecessors(a)) - {a}, set(G.successors(a)) - {a}
        vol = sum(d["amount_usd"] for *_, d in G.in_edges(a, data=True)) \
            + sum(d["amount_usd"] for *_, d in G.out_edges(a, data=True))
        return dict(in_degree=len(pred), out_degree=len(succ), unique=len(pred | succ),
                    txn=G.in_degree(a) + G.out_degree(a), volume=vol)
    a, b = node(sender), node(receiver)
    return {
        "sender_in_degree_24h": a["in_degree"], "sender_out_degree_24h": a["out_degree"],
        "receiver_in_degree_24h": b["in_degree"], "receiver_out_degree_24h": b["out_degree"],
        "sender_counterparties_24h": a["unique"], "receiver_counterparties_24h": b["unique"],
        "sender_txn_count_24h": a["txn"], "receiver_txn_count_24h": b["txn"],
        "sender_volume_24h": a["volume"], "receiver_volume_24h": b["volume"],
        "pair_count_24h": G.number_of_edges(sender, receiver) if sender in G and receiver in G else 0,
    }


# ================================================================ time split
def assign_time_split(timestamps: pd.Series, windows: dict) -> pd.Series:
    """Map each timestamp to the first window [start, end) containing it, else 'excluded'."""
    out = pd.Series("excluded", index=timestamps.index, dtype=object)
    for name, (start, end) in windows.items():
        m = (timestamps >= pd.Timestamp(start)) & (timestamps < pd.Timestamp(end))
        out[m & (out == "excluded")] = name
    return out
