"""
AI Finance Controller — Streamlit Dashboard
============================================
Interactive visual dashboard for the 3-pass reconciliation engine.

Layout:
    * Top KPI row: Match Rate (%), Throughput (rec/sec), Precision, Unresolved
    * Sidebar: "Generate New Synthetic Batch" button + batch size slider
    * Three tabs: Matched Ledger, Unresolved Exception Queue, Audit Trail Viewer
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd
from pathlib import Path

# App-local imports
from src.generator import generate_batch
from src.engine import reconcile
from src.audit import AuditLogger
from src.ai_analyzer import AIAnalyzer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Finance Controller",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state initialisation (persistent across reruns)
# ---------------------------------------------------------------------------
if "recon_result" not in st.session_state:
    st.session_state.recon_result = None
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = AuditLogger()
if "data_paths" not in st.session_state:
    # Points to the repo-level data/ folder
    st.session_state.data_paths = None
if "generated" not in st.session_state:
    st.session_state.generated = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _run_reconciliation(paths: dict[str, Path]) -> ReconResult:
    """Wrap the engine call and store result in session state."""
    result = reconcile(
        paths["gateway_logs.csv"],
        paths["bank_statement.csv"],
        paths["internal_ledger.csv"],
        audit=st.session_state.audit_logger,
        analyzer=None,  # AI key optional; we fall back to heuristic.
    )
    st.session_state.recon_result = result
    st.session_state.data_paths = paths
    return result


# ---------------------------------------------------------------------------
# Page UI
# ---------------------------------------------------------------------------

st.title("AI Finance Controller")
st.caption("Multi-source reconciliation across payment gateways, bank settlement statements, and internal merchant ledgers.")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    batch_size = st.slider(
        "Synthetic batch size",
        min_value=50,
        max_value=1000,
        value=100,
        step=50,
        help="Number of records to generate in the synthetic batch.",
    )
    generate_btn = st.button(
        "Generate New Synthetic Batch",
        help="Creates a new batch of synthetic transaction records and "
        "immediately runs reconciliation against it.",
    )

# ---------------------------------------------------------------------------
# If generate pressed OR we have no result yet, spin up a new batch
# ---------------------------------------------------------------------------
if generate_btn or not st.session_state.generated:
    with st.spinner("Generating synthetic batch …"):
        paths = generate_batch(n=batch_size, seed=42)
    result = _run_reconciliation(paths)
    st.session_state.generated = True
    st.rerun()  # flush the fresh data into the UI

# ---------------------------------------------------------------------------
# If we still have no result (first-run without clicking the button),
#    seed a tiny default batch so the UI isn't empty.
# ---------------------------------------------------------------------------
if not st.session_state.generated or st.session_state.recon_result is None:
    with st.spinner("Seeding an initial batch …"):
        paths = generate_batch(n=50, seed=123)
    result = _run_reconciliation(paths)
    st.session_state.generated = True
    st.rerun()

result: ReconResult = st.session_state.recon_result

# ---------------------------------------------------------------------------
# Top KPI Row
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Match Rate (%)",
        value=f"{result.stats['match_rate_pct']}%",
        help="Matched records / total gateway records",
    )

with kpi2:
    st.metric(
        "Engine Throughput (rec/sec)",
        value=f"{result.stats['throughput_rec_sec']}",
        help="Records processed per second (wall-clock time).",
    )

with kpi3:
    st.metric(
        "Precision",
        value=f"{result.stats['precision_pct']}%",
        help="True-positive rate — never force-matches low confidence.",
    )

with kpi4:
    st.metric(
        "Unresolved Exception Count",
        value=f"{result.stats['unresolved_count']}",
        help="Records sent to the unresolved / escalated queue.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Three Tabs
# ---------------------------------------------------------------------------
tab_matched, tab_unresolved, tab_audit = st.tabs(
    ["📋 Matched Ledger", "⚠️ Unresolved Exception Queue", "📜 Audit Trail"]
)

# ---- TAB 1: Matched Ledger ----
with tab_matched:
    st.subheader("Matched Ledger Table")
    st.caption(
        "Shows every matched record with its match stage and confidence level."
    )
    if not result.matches.empty:
        st.dataframe(
            result.matches[
                ["txn_id", "order_id", "amount_paise", "match_stage", "confidence"]
            ].assign(
                amount_paise=lambda d: d["amount_paise"].map(
                    lambda v: f"₹{v/100:,.2f}"
                )
            ),
            column_config={
                "txn_id": "Txn ID",
                "order_id": "Order ID",
                "amount_paise": "Amt (paise)",
                "match_stage": "Match Stage",
                "confidence": st.column_config.NumberColumn(
                    "Confidence", format="%.2f"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No matches found — try generating a new batch.")

# ---- TAB 2: Unresolved Exception Queue ----
with tab_unresolved:
    st.subheader("Unresolved Exception Queue")
    st.caption(
        "Records that could not be auto-matched. "
        "Confidence < 0.85 → ESCALATED_TO_HUMAN."
    )
    if not result.unresolved.empty:
        st.dataframe(
            result.unresolved[
                ["txn_id", "order_id", "amount_paise", "confidence", "discrepancy_reason", "action"]
            ].assign(
                amount_paise=lambda d: d["amount_paise"].map(
                    lambda v: f"₹{v/100:,.2f}"
                )
            ),
            column_config={
                "txn_id": "Txn ID",
                "order_id": "Order ID",
                "amount_paise": "Amt (paise)",
                "confidence": st.column_config.NumberColumn(
                    "Confidence", format="%.2f"
                ),
                "discrepancy_reason": "Discrepancy Reason",
                "action": "Action",
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("No unresolved exceptions — all records matched!")

# ---- TAB 3: Audit Trail Viewer ----
with tab_audit:
    st.subheader("System Audit Trail Viewer")
    audit_path = Path(__file__).resolve().parent.parent / "audit.log"
    logger_obj: AuditLogger = st.session_state.audit_logger

    if audit_path.exists():
        entries = logger_obj.read_all()
        if entries:
            df_audit = pd.DataFrame(entries)
            # Keep only the most useful columns for the UI view.
            display = df_audit[
                ["timestamp", "event", "status", "rule", "confidence"]
            ].copy()
            # For readability, truncate very long discrepancy_reason fields if any.
            st.dataframe(
                display,
                column_config={
                    "timestamp": "Timestamp (UTC)",
                    "event": "Event",
                    "status": "Status",
                    "rule": "Rule",
                    "confidence": st.column_config.NumberColumn(
                        "Confidence", format="%.2f"
                    ),
                },
                hide_index=True,
                use_container_width=True,
                height=600,
            )
            st.caption(
                f"Showing {len(entries)} audit entries. "
                "Log file located at `audit.log`."
            )
        else:
            st.info("No audit entries yet — generate and reconcile a batch first.")
    else:
        st.info("Audit log not found — run a reconciliation to populate it.")

# ---------------------------------------------------------------------------
# Footer note
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Data model: all monetary values in paise (integer). "
    "Engine uses a 3-pass pipeline (exact → fee-tolerant → AI). "
    "Precision target: 100% (zero false-positive matches)."
)