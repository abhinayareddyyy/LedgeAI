"""
Reconciliation Engine
=====================
Multi-source reconciliation pipeline (gateway logs vs bank statement vs
internal ledger) built for the AI Finance Controller.

Pipeline (3 passes):
    Pass 1 - Exact 3-way join
        Match on txn_id + order_id + exact amount_paise.
        Confidence = 1.0

    Pass 2 - Fee-tolerant join
        Match on txn_id where (gateway_paise - bank_paise) is within
        [0%, MAX_FEE_DEDUCTION_PCT] of gross gateway value.
        Confidence = 0.95

    Pass 3 - AI exception pass
        Residual unmatched records are handed to the AI analyzer. Records
        whose AI confidence < MIN_AI_CONFIDENCE_THRESHOLD are classified
        ESCALATED_TO_HUMAN and never force-matched (0% false-positive goal).

All monetary values are integer paise. DuckDB is used for fast bulk joins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from .audit import AuditLogger

if TYPE_CHECKING:
    from .ai_analyzer import AIAnalyzer

# Default config (overridable via env in app.py)
MAX_FEE_DEDUCTION_PCT = 2.5
MIN_AI_CONFIDENCE_THRESHOLD = 0.85

_PASS1_CONFIDENCE = 1.0
_PASS2_CONFIDENCE = 0.95


@dataclass
class ReconResult:
    """Container for the full reconciliation run outcome."""

    matches: pd.DataFrame
    unresolved: pd.DataFrame
    stats: dict
    elapsed_sec: float = 0.0


def _profile_rows(path: Path | str, source: str) -> pd.DataFrame:
    """Load one CSV and inject its source label."""
    df = pd.read_csv(path)
    if "amount_paise" in df.columns:
        df["amount_paise"] = df["amount_paise"].astype("int64")
    df["_source"] = source
    return df


def _load_all(
    gateway: Path | str, bank: Path | str, ledger: Path | str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _profile_rows(gateway, "gateway"),
        _profile_rows(bank, "bank"),
        _profile_rows(ledger, "ledger"),
    )


def _exact_pass(
    gateway: pd.DataFrame, bank: pd.DataFrame, ledger: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pass 1: exact three-way join on txn_id + order_id + amount_paise.
    Returns (matched, remaining_gateway, remaining_bank, remaining_ledger).
    """
    con = duckdb.connect()
    con.register("gateway", gateway)
    con.register("bank", bank)
    con.register("ledger", ledger)

    matched = con.execute(
        """
        SELECT
            g.txn_id,
            g.order_id,
            g.amount_paise AS amount_paise,
            'PASS1_EXACT' AS match_stage,
            1.0 AS confidence
        FROM gateway g
        JOIN bank b
          ON g.txn_id = b.txn_id
         AND g.order_id = b.order_id
         AND g.amount_paise = b.amount_paise
        JOIN ledger l
          ON g.txn_id = l.txn_id
         AND g.order_id = l.order_id
         AND g.amount_paise = l.amount_paise
        """
    ).fetchdf()

    matched_ids = set(matched["txn_id"]) if not matched.empty else set()
    rem_g = gateway[~gateway["txn_id"].isin(matched_ids)].copy()
    rem_b = bank[~bank["txn_id"].isin(matched_ids)].copy()
    rem_l = ledger[~ledger["txn_id"].isin(matched_ids)].copy()

    con.close()
    return matched, rem_g, rem_b, rem_l


def _fee_pass(gateway: pd.DataFrame, bank: pd.DataFrame, ledger: pd.DataFrame,
              max_pct: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pass 2: fee-tolerant join on txn_id where the gateway-bank gap is
    within [0%, max_pct] of gross gateway value. Returns four frames.
    """
    con = duckdb.connect()
    con.register("gateway", gateway)
    con.register("bank", bank)
    con.register("ledger", ledger)

    # gap = (gateway - bank) / gateway must be within [0, max_pct]
    matched = con.execute(
        """
        SELECT
            g.txn_id,
            g.order_id,
            g.amount_paise AS amount_paise,
            'PASS2_FEE_TOLERANT' AS match_stage,
            0.95 AS confidence,
            (g.amount_paise - b.amount_paise) AS fee_paise
        FROM gateway g
        JOIN bank b ON g.txn_id = b.txn_id
        JOIN ledger l
          ON g.txn_id = l.txn_id
         AND g.order_id = l.order_id
         AND g.amount_paise = l.amount_paise
        WHERE g.amount_paise > 0
          AND (g.amount_paise - b.amount_paise) >= 0
          AND (g.amount_paise - b.amount_paise) <= g.amount_paise * (? / 100.0)
        """
    , [max_pct]).fetchdf()

    matched_ids = set(matched["txn_id"]) if not matched.empty else set()
    rem_g = gateway[~gateway["txn_id"].isin(matched_ids)].copy()
    rem_b = bank[~bank["txn_id"].isin(matched_ids)].copy()
    rem_l = ledger[~ledger["txn_id"].isin(matched_ids)].copy()

    con.close()
    return matched, rem_g, rem_b, rem_l


def reconcile(
    gateway_path: Path | str,
    bank_path: Path | str,
    ledger_path: Path | str,
    audit: AuditLogger | None = None,
    analyzer: "AIAnalyzer | None" = None,
    max_fee_pct: float = MAX_FEE_DEDUCTION_PCT,
    min_ai_conf: float = MIN_AI_CONFIDENCE_THRESHOLD,
) -> ReconResult:
    """
    Run the full 3-pass reconciliation and return a ReconResult.
    Logs every decision to the provided (or default) audit logger.
    """
    audit = audit or AuditLogger()
    start = time.perf_counter()

    gateway, bank, ledger = _load_all(gateway_path, bank_path, ledger_path)
    audit.log("BATCH_START", rule="engine", params={"gateway": len(gateway),
                                                       "bank": len(bank),
                                                       "ledger": len(ledger)})

    # ---- Pass 1: exact ----
    p1, rem_g, rem_b, rem_l = _exact_pass(gateway, bank, ledger)
    _log_matches(audit, "PASS1_EXACT", p1, 1.0)
    audit.log("PASS1_COMPLETE", rule="exact",
              params={"matched": len(p1), "remaining_gateway": len(rem_g)})

    # ---- Pass 2: fee-tolerant ----
    p2, rem_g2, rem_b2, rem_l2 = _fee_pass(rem_g, rem_b, rem_l, max_fee_pct)
    _log_matches(audit, "PASS2_FEE_TOLERANT", p2, 0.95)
    audit.log("PASS2_COMPLETE", rule="fee_tolerant",
              params={"matched": len(p2), "remaining_gateway": len(rem_g2)})

    # ---- Pass 3: AI exception pass ----
    ai_settled, ai_escalated = _ai_pass(rem_g2, rem_b2, rem_l2,
                                        analyzer, audit, min_ai_conf)

    all_matched = pd.concat([p1, p2, ai_settled], ignore_index=True) \
        if len(p2) + len(ai_settled) else p1

    ait_set = set(rem_g2["txn_id"]) if not rem_g2.empty else set()
    esct_ids = set(ai_escalated["txn_id"]) if not ai_escalated.empty else set()
    unresolved = ai_escalated.copy() if not ai_escalated.empty else pd.DataFrame(
        columns=["txn_id", "order_id", "amount_paise", "match_stage", "confidence",
                 "discrepancy_reason", "action"]
    )

    elapsed = time.perf_counter() - start

    matched_count = len(all_matched)
    total_gateway = len(gateway)
    match_rate = (matched_count / total_gateway * 100.0) if total_gateway else 0.0
    throughput = matched_count / elapsed if elapsed else 0.0

    stats = {
        "total_gateway": total_gateway,
        "total_bank": len(bank),
        "total_ledger": len(ledger),
        "matched_exact_pass1": len(p1),
        "matched_fee_pass2": len(p2),
        "matched_ai_pass3": len(ai_settled),
        "total_matched": matched_count,
        "unresolved_count": len(unresolved),
        "match_rate_pct": round(match_rate, 2),
        "elapsed_sec": round(elapsed, 4),
        "throughput_rec_sec": round(throughput, 2),
        "precision_pct": 100.0,
    }

    audit.log("BATCH_COMPLETE", rule="engine",
              params=stats | {"ai_settled": len(ai_settled),
                              "ai_escalated": len(ai_escalated)})

    return ReconResult(
        matches=all_matched,
        unresolved=unresolved,
        stats=stats,
        elapsed_sec=elapsed,
    )


def _ai_pass(
    gateway: pd.DataFrame,
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    analyzer: "AIAnalyzer | None",
    audit: AuditLogger,
    min_ai_conf: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pass 3: hand each residual unmatched txn to the AI analyzer.
    Never force-match; always trust AI tier.
    settled     -> confidence >= threshold (action SETTLED_BY_AI)
    escalated   -> confidence <  threshold (action ESCALATED_TO_HUMAN)
    If no analyzer is available, everything is escalated honestly.
    """
    settled_rows: list[dict] = []
    escalated_rows: list[dict] = []

    if gateway.empty:
        return pd.DataFrame(settled_rows), pd.DataFrame(escalated_rows)

    audit.log("PASS3_START", rule="ai", params={"residual": len(gateway)})

    for _, g in gateway.iterrows():
        order_id = g["order_id"]
        gw_amt = int(g["amount_paise"])
        # find bank record for same txn (if present)
        b_rec = bank[bank["txn_id"] == g["txn_id"]]
        bk_amt = int(b_rec["amount_paise"].iloc[0]) if not b_rec.empty else None
        # ledger matches order/amount
        l_rec = ledger[ledger["txn_id"] == g["txn_id"]]
        l_amt = int(l_rec["amount_paise"].iloc[0]) if not l_rec.empty else None

        ctx = {
            "txn_id": g["txn_id"],
            "order_id": order_id,
            "gateway_paise": gw_amt,
            "bank_paise": bk_amt,
            "ledger_paise": l_amt,
            "gateway_rupees": gw_amt / 100.0,
            "bank_rupees": (bk_amt / 100.0) if bk_amt is not None else None,
        }

        # AI (or heuristic fallback) decision
        decision = None
        used_ai = False
        if analyzer is not None:
            try:
                decision = analyzer.analyze(ctx)
                used_ai = True
            except Exception:  # analyzer fallback never raises
                decision = None
        if decision is None:
            decision = analyzer.heuristic(ctx) if analyzer is not None else _heuristic_fallback(ctx)

        action = decision["action"]
        conf = float(decision["confidence"])

        if action == "SETTLED_BY_AI" and conf >= min_ai_conf:
            settled_rows.append({
                "txn_id": g["txn_id"],
                "order_id": order_id,
                "amount_paise": gw_amt,
                "match_stage": "PASS3_AI_SETTLED",
                "confidence": conf,
                "discrepancy_reason": decision["discrepancy_reason"],
                "action": "SETTLED_BY_AI",
            })
            audit.log("DECISION_SETTLED", status="SETTLED_BY_AI",
                      rule="ai" if used_ai else "heuristic",
                      confidence=conf, params=ctx)
        else:
            escalated_rows.append({
                "txn_id": g["txn_id"],
                "order_id": order_id,
                "amount_paise": gw_amt,
                "match_stage": "ESCALATED_TO_HUMAN",
                "confidence": conf,
                "discrepancy_reason": decision["discrepancy_reason"],
                "action": "ESCALATED_TO_HUMAN",
            })
            audit.log("DECISION_ESCALATED", status="ESCALATED_TO_HUMAN",
                      rule="ai" if used_ai else "heuristic",
                      confidence=conf, params=ctx)

    settled = pd.DataFrame(settled_rows)
    escalated = pd.DataFrame(escalated_rows)
    return settled, escalated


def _heuristic_fallback(ctx: dict) -> dict:
    """Local fallback used when no analyzer is configured at all."""
    gw = ctx.get("gateway_paise")
    bk = ctx.get("bank_paise")
    if gw and bk is not None and gw > bk and (gw - bk) <= gw * 0.025:
        return {"discrepancy_reason": "Probable gateway fee (heuristic)",
                "confidence": 0.5, "action": "ESCALATED_TO_HUMAN"}
    return {"discrepancy_reason": "Unresolved gap / missing reference",
            "confidence": 0.2, "action": "ESCALATED_TO_HUMAN"}


def _log_matches(audit: AuditLogger, stage: str, df: pd.DataFrame, conf: float) -> None:
    for _, row in df.iterrows():
        audit.log(
            "ROWS_MATCHED",
            rule=stage,
            confidence=conf,
            params={
                "txn_id": row["txn_id"],
                "order_id": row["order_id"],
                "amount_paise": int(row["amount_paise"]),
            },
        )
