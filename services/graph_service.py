"""
Graph analysis for the web app: bounded NetworkX subgraphs -> Plotly.js figures,
connected accounts and suspicious relationships. The full network is never rendered.
Suspicious edges come from stored model results, never from the actual label.
"""
from __future__ import annotations

import networkx as nx
from sqlalchemy import and_, func, or_

from database.models import FinancialTransaction as T
from database.models import FraudAnalysis as A
from extensions import db

COLORS = {"normal": "#c3c2b7", "suspicious": "#d03b3b", "node": "#2a78d6", "node_flag": "#eb6834",
          "center": "#0b0b0b", "ink": "#52514e"}


def parse_account(text: str):
    """'bank|account' or bare 'account' -> (bank or None, account)."""
    text = (text or "").strip().upper()
    if "|" in text:
        bank, acct = text.split("|", 1)
        return bank.strip(), acct.strip()
    return None, text


def _match(bank, acct):
    s = T.sender_account == acct
    r = T.receiver_account == acct
    if bank:
        s = and_(s, T.sender_bank == bank)
        r = and_(r, T.receiver_bank == bank)
    return or_(s, r)


def network_figure(dataset_id: int, version: str | None, account: str | None = None, max_edges: int = 250):
    q = (db.session.query(T.transaction_reference, T.sender_bank, T.sender_account, T.receiver_bank,
                          T.receiver_account, T.transaction_amount, T.payment_currency, T.transaction_type,
                          A.anomaly_status, A.combined_anomaly_score, A.risk_level)
         .outerjoin(A, and_(A.transaction_id == T.id, A.model_version == version))
         .filter(T.dataset_id == dataset_id))
    center = None
    if account:
        bank, acct = parse_account(account)
        if not acct:
            return None
        q = q.filter(_match(bank, acct))
        center = (bank, acct)
        q = q.order_by(A.combined_anomaly_score.desc(), T.transaction_date.desc())
    else:
        q = q.filter(A.anomaly_status == "Suspicious").order_by(A.combined_anomaly_score.desc())
    rows = q.limit(max_edges).all()
    if not rows:
        return {"empty": True}

    G = nx.MultiDiGraph()
    for r in rows:
        s, t = f"{r.sender_bank}|{r.sender_account}", f"{r.receiver_bank}|{r.receiver_account}"
        G.add_edge(s, t, ref=r.transaction_reference, suspicious=(r.anomaly_status == "Suspicious"),
                   amount=r.transaction_amount, currency=r.payment_currency, kind=r.transaction_type,
                   score=r.combined_anomaly_score)
    center_ids = set()
    if center:
        center_ids = {n for n in G if n.split("|", 1)[1] == center[1] and (center[0] is None or n.split("|")[0] == center[0])}
    S = nx.DiGraph()
    S.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        e = S.get_edge_data(u, v) or {"count": 0, "suspicious": 0, "refs": []}
        e["count"] += 1
        e["suspicious"] += int(d["suspicious"])
        if len(e["refs"]) < 3:
            e["refs"].append(d["ref"])
        S.add_edge(u, v, **e)
    pos = nx.spring_layout(S.to_undirected(), seed=42, k=1.2 / max(len(S) ** 0.5, 1))

    traces = []
    for kind, color, width in (("normal", COLORS["normal"], 1.2), ("suspicious", COLORS["suspicious"], 2.4)):
        xs, ys = [], []
        for u, v, d in S.edges(data=True):
            if (d["suspicious"] > 0) == (kind == "suspicious"):
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        traces.append({"type": "scatter", "mode": "lines", "x": xs, "y": ys, "hoverinfo": "skip",
                       "name": "suspicious transactions" if kind == "suspicious" else "other transactions",
                       "line": {"color": color, "width": width}})
    # edge midpoints carry the hover text
    mx, my, mt = [], [], []
    for u, v, d in S.edges(data=True):
        mx.append((pos[u][0] + pos[v][0]) / 2)
        my.append((pos[u][1] + pos[v][1]) / 2)
        mt.append(f"{u} → {v}<br>{d['count']} transaction(s), {d['suspicious']} suspicious<br>"
                  + ", ".join(d["refs"]))
    traces.append({"type": "scatter", "mode": "markers", "x": mx, "y": my, "text": mt, "hoverinfo": "text",
                   "marker": {"size": 6, "color": "rgba(0,0,0,0)"}, "showlegend": False})
    nodes = list(S.nodes())
    flagged = {n: sum(d["suspicious"] for *_, d in S.in_edges(n, data=True))
               + sum(d["suspicious"] for *_, d in S.out_edges(n, data=True)) for n in nodes}
    traces.append({
        "type": "scatter", "mode": "markers", "name": "accounts", "hoverinfo": "text",
        "x": [pos[n][0] for n in nodes], "y": [pos[n][1] for n in nodes],
        "text": [f"<b>{n}</b><br>paid {S.out_degree(n)} / paid by {S.in_degree(n)} account(s) here"
                 f"<br>{flagged[n]} suspicious relationship(s)" for n in nodes],
        "customdata": nodes,
        "marker": {"size": [22 if n in center_ids else min(9 + 2.5 * S.degree(n), 30) for n in nodes],
                   "color": [COLORS["center"] if n in center_ids else
                             (COLORS["node_flag"] if flagged[n] else COLORS["node"]) for n in nodes],
                   "line": {"color": "#ffffff", "width": 1.5}}})
    annotations = []
    if S.number_of_edges() <= 150:
        for u, v in S.edges():
            annotations.append({"ax": pos[u][0], "ay": pos[u][1], "x": pos[v][0], "y": pos[v][1],
                                "xref": "x", "yref": "y", "axref": "x", "ayref": "y", "showarrow": True,
                                "arrowhead": 2, "arrowsize": 1, "arrowwidth": 1,
                                "arrowcolor": COLORS["suspicious"] if S[u][v]["suspicious"] else COLORS["normal"],
                                "standoff": 6, "text": ""})
    return {"empty": False, "data": traces,
            "layout": {"showlegend": True, "hovermode": "closest", "annotations": annotations,
                       "xaxis": {"visible": False}, "yaxis": {"visible": False},
                       "legend": {"orientation": "h", "y": -0.05},
                       "margin": {"l": 10, "r": 10, "t": 10, "b": 30}},
            "stats": {"accounts": S.number_of_nodes(), "relationships": S.number_of_edges(),
                      "transactions": G.number_of_edges(),
                      "suspicious_transactions": sum(1 for *_, d in G.edges(data=True) if d["suspicious"]),
                      "truncated": len(rows) >= max_edges, "max_edges": max_edges}}


def connected_accounts(dataset_id: int, version: str | None, account: str):
    bank, acct = parse_account(account)
    if not acct:
        return []
    q = (db.session.query(T.sender_bank, T.sender_account, T.receiver_bank, T.receiver_account,
                          T.transaction_amount, T.payment_currency, A.anomaly_status)
         .outerjoin(A, and_(A.transaction_id == T.id, A.model_version == version))
         .filter(T.dataset_id == dataset_id, _match(bank, acct)))
    agg = {}
    for r in q.all():
        outgoing = r.sender_account == acct and (bank is None or r.sender_bank == bank)
        other = f"{r.receiver_bank}|{r.receiver_account}" if outgoing else f"{r.sender_bank}|{r.sender_account}"
        a = agg.setdefault(other, {"account": other, "sent_to": 0, "received_from": 0, "suspicious": 0,
                                   "currencies": set()})
        a["sent_to" if outgoing else "received_from"] += 1
        a["suspicious"] += int(r.anomaly_status == "Suspicious")
        a["currencies"].add(r.payment_currency)
    out = sorted(agg.values(), key=lambda a: (-a["suspicious"], -(a["sent_to"] + a["received_from"])))
    for a in out:
        a["currencies"] = ", ".join(sorted(a["currencies"]))
    return out


def suspicious_relationships(dataset_id: int | None, version: str | None, limit: int = 50):
    q = (db.session.query(T.sender_bank, T.sender_account, T.receiver_bank, T.receiver_account,
                          func.count(A.id).label("n"), func.max(A.combined_anomaly_score).label("max_score"),
                          func.min(T.transaction_date).label("first"), func.max(T.transaction_date).label("last"))
         .join(A, A.transaction_id == T.id)
         .filter(A.anomaly_status == "Suspicious"))
    if version:
        q = q.filter(A.model_version == version)
    if dataset_id:
        q = q.filter(T.dataset_id == dataset_id)
    q = (q.group_by(T.sender_bank, T.sender_account, T.receiver_bank, T.receiver_account)
         .order_by(func.count(A.id).desc(), func.max(A.combined_anomaly_score).desc()).limit(limit))
    return [{"sender": f"{r.sender_bank}|{r.sender_account}", "receiver": f"{r.receiver_bank}|{r.receiver_account}",
             "count": r.n, "max_score": r.max_score, "first": r.first, "last": r.last} for r in q.all()]
