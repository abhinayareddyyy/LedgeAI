"""
AI Exception Analyzer
=====================
Uses OpenAI (gpt-4o-mini) with Pydantic Structured Outputs to explain and
classify residual reconciliation exceptions.

Output schema:
    discrepancy_reason : str   (human-readable explanation)
    confidence         : float (0.0 - 1.0)
    action             : "SETTLED_BY_AI" | "ESCALATED_TO_HUMAN"

Safety design:
    * Pydantic model enforced via response_format -> forced JSON schema
    * If OPENAI_API_KEY is missing OR the API call fails, we fall back to a
      deterministic heuristic so the engine never crashes and the audit
      trail still records a (lower-confidence) explanation.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Threshold below which we treat any AI decision as "escalate".
DEFAULT_MIN_AI_CONFIDENCE = 0.85


class DiscrepancyOutput(BaseModel):
    """Structured output enforced by the OpenAI API (JSON schema)."""

    discrepancy_reason: str = Field(
        description="Human-readable explanation of the value mismatch."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the explanation, 0.0 to 1.0."
    )
    action: str = Field(
        description="SETTLED_BY_AI if confident else ESCALATED_TO_HUMAN."
    )


_SYSTEM_PROMPT = (
    "You are a meticulous financial reconciliation analyst. Given a residual "
    "unmatched transaction across a payment gateway, a bank settlement "
    "statement, and an internal ledger, explain the discrepancy precisely. "
    "If the difference is small and plausibly explained by gateway processing "
    "fees or timestamp skew, mark SETTLED_BY_AI with high confidence. If the "
    "gap is unexplained (missing reference ids, odd rupee gaps), mark "
    "ESCALATED_TO_HUMAN with lower confidence. Never invent reference numbers."
)


def _build_user_prompt(ctx: dict) -> str:
    return (
        "Gateway txn_id: {txn_id}\n"
        "Order id: {order_id}\n"
        "Gateway amount (paise): {gateway_paise} (INR {gateway_rupees})\n"
        "Bank amount (paise): {bank_paise}\n"
        "Ledger amount (paise): {ledger_paise}\n"
        "Explain the discrepancy and classify it."
    ).format(**ctx)


class AIAnalyzer:
    """Analyzes residual exceptions using OpenAI structured outputs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        min_ai_conf: float = DEFAULT_MIN_AI_CONFIDENCE,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.min_ai_conf = min_ai_conf
        self._client = None
        self.available = bool(self.api_key)
        if self.available:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        else:
            self._client = None

    def analyze(self, ctx: dict) -> dict[str, Any]:
        """Return {discrepancy_reason, confidence, action}."""
        try:
            if not self.available or self._client is None:
                return self.heuristic(ctx)

            completion = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(ctx)},
                ],
                response_format=DiscrepancyOutput,
            )
            parsed: DiscrepancyOutput = completion.choices[0].message.parsed

            # Always re-classify based on the enforced threshold so the
            # "never force-match" guarantee holds regardless of model output.
            action = (
                "SETTLED_BY_AI"
                if parsed.action == "SETTLED_BY_AI"
                and parsed.confidence >= self.min_ai_conf
                else "ESCALATED_TO_HUMAN"
            )
            return {
                "discrepancy_reason": parsed.discrepancy_reason,
                "confidence": float(parsed.confidence),
                "action": action,
            }
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            # Fall back to heuristic without crashing the pipeline.
            return self.heuristic(ctx, error=str(exc))

    def heuristic(self, ctx: dict, error: str | None = None) -> dict[str, Any]:
        """Deterministic fallback explanation when API is unavailable/fails."""
        gw = ctx.get("gateway_paise")
        bk = ctx.get("bank_paise")
        gap = (gw - bk) if (gw is not None and bk is not None) else 0

        prefix = f"[heuristic{'; api_error=' + error if error else ''}] "
        if gw and bk is not None and 0 < gap <= gw * 0.025:
            return {
                "discrepancy_reason": prefix
                + "Probable gateway processing fee deduction (~2%).",
                "confidence": 0.6,
                "action": "ESCALATED_TO_HUMAN",
            }
        if gw and bk is not None and gap == 45000:
            return {
                "discrepancy_reason": prefix
                + "Unresolved INR 450 gap with no matching explanation.",
                "confidence": 0.2,
                "action": "ESCALATED_TO_HUMAN",
            }
        return {
            "discrepancy_reason": prefix
            + "Missing bank reference / unexplained variance; requires review.",
            "confidence": 0.2,
            "action": "ESCALATED_TO_HUMAN",
        }
