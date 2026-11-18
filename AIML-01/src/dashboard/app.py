"""Analyst queue, explanations, linked accounts, labels and operations."""

import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from src.config import load_config
from src.dashboard.db import migrate, query, save_label, set_case
from src.dashboard.explanations import explain_case

st.set_page_config(page_title="Fraud Review", page_icon="🔎", layout="wide")
st.title("Fraud review desk")
st.caption("Review alerts, inspect the evidence, and send confirmed labels back to training.")


@st.cache_resource
def initialize():
    migrate()
    return True


initialize()
queue_tab, ops_tab, drift_tab = st.tabs(["Review queue", "Operations", "Drift reports"])


@st.fragment(run_every="5s")
def render_queue():
    status_filter = st.selectbox("Show cases", ["open", "reviewing", "closed", "all"])
    rows = query(
        """SELECT t.transaction_id, t.account_id, t.score, t.model_version,
        a.review_reason, c.status, a.created_at FROM app.alerts a
        JOIN app.transactions t USING(transaction_id) JOIN app.cases c USING(transaction_id)
        WHERE (%s='all' OR c.status=%s) ORDER BY a.created_at DESC LIMIT 200""",
        (status_filter, status_filter),
    )
    st.metric("Cases shown", len(rows))
    if not rows:
        st.info(
            "The queue is empty. New alerts and random audit samples appear here automatically."
        )
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    selected = st.selectbox("Open transaction", [r["transaction_id"] for r in rows])
    details = query(
        """SELECT t.result, c.status, c.analyst_note FROM app.transactions t
        JOIN app.cases c USING(transaction_id) WHERE t.transaction_id=%s""",
        (selected,),
    )[0]
    result = details["result"]
    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"Transaction {selected}")
        st.metric("Fraud risk", f"{result['score']:.1%}")
        st.write(f"Account: {result['account_id']} · Model version: {result['model_version']}")
        st.write(f"Alert threshold: {result['threshold']:.3f}")
        st.write("Main XGBoost reasons (signed contributions to log odds):")
        st.dataframe(result["reasons"] or explain_case(result), hide_index=True, width="stretch")
        with st.expander("Transaction and computed features"):
            st.json({"transaction": result["transaction"], "features": result["features"]})
        state = st.selectbox(
            "Case status",
            ["open", "reviewing", "closed"],
            index=["open", "reviewing", "closed"].index(details["status"]),
        )
        note = st.text_area("Analyst notes", value=details["analyst_note"])
        if st.button("Save case"):
            set_case(selected, state, note)
            st.success("Case saved")
        a, b = st.columns(2)
        if a.button("Confirm fraud", type="primary"):
            save_label(selected, 1)
            st.success("Fraud label saved; case closed")
        if b.button("Confirm legitimate"):
            save_label(selected, 0)
            st.success("Legitimate label saved; case closed")
        labels = query(
            "SELECT label, created_at FROM app.labels WHERE transaction_id=%s", (selected,)
        )
        if labels:
            st.write("Saved feedback:", labels[0])
    with right:
        st.subheader("Linked accounts")
        membership = query(
            "SELECT * FROM app.ring_members WHERE account_id=%s", (result["account_id"],)
        )
        if membership:
            ring = membership[0]
            st.write(f"Group {ring['ring_id']} · {ring['ring_size']} accounts")
            linked = query(
                "SELECT account_id,ring_score FROM app.ring_members "
                "WHERE ring_id=%s ORDER BY ring_score DESC LIMIT 30",
                (ring["ring_id"],),
            )
            st.dataframe(linked, hide_index=True)
            st.caption(
                "Shared identifiers indicate a link, not proof that an account is fraudulent."
            )
        else:
            st.info("No training-period graph link for this account.")


with queue_tab:
    render_queue()
with ops_tab:
    totals = query("""SELECT (SELECT COUNT(*) FROM app.transactions) AS scored,
        (SELECT COUNT(*) FROM app.alerts) AS queued,
        (SELECT COUNT(*) FROM app.labels) AS labeled""")[0]
    cols = st.columns(3)
    for col, key in zip(cols, ["scored", "queued", "labeled"]):
        col.metric(key.capitalize(), totals[key])
    try:
        response = requests.get(
            os.getenv("SCORER_URL", "http://127.0.0.1:8000") + "/health", timeout=3
        )
        st.json(response.json())
    except requests.RequestException as exc:
        st.error(f"Scorer unavailable: {exc}")
    recent = query(
        "SELECT score, received_at FROM app.transactions ORDER BY received_at DESC LIMIT 1000"
    )
    if recent:
        counts, edges = np.histogram(pd.DataFrame(recent).score, bins=20, range=(0, 1))
        labels = [f"{left:.2f}–{right:.2f}" for left, right in zip(edges[:-1], edges[1:])]
        st.bar_chart(pd.DataFrame({"Transactions": counts}, index=labels))
    st.caption("Metrics: scorer /metrics · API contract: scorer /docs")
with drift_tab:
    directory = load_config().data_processed_dir / "monitoring"
    summaries = sorted(directory.glob("*/summary.json"), reverse=True)
    if summaries:
        selected_report = st.selectbox(
            "Report", summaries, format_func=lambda path: path.parent.name
        )
        st.json(json.loads(selected_report.read_text()))
        html = selected_report.parent / "report.html"
        if html.exists():
            components.html(html.read_text(encoding="utf-8"), height=900, scrolling=True)
    else:
        st.info("A drift report appears after monitoring has observed recent traffic.")
