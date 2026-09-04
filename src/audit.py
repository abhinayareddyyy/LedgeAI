"""
Append-Only JSON Audit Logger
=============================
Writes a JSON-lines audit trail to an append-only `audit.log` file.

Every reconciliation decision (exact match, fee-tolerant match, AI-settled,
AI-escalated, system event) is recorded with:

    * timestamp (ISO 8601, UTC)
    * event / status
    * the model/rule that produced the decision
    * confidence
    * input parameters (txn_id, order_id, amounts, stage, etc.)

The file is opened in append mode ("a") so historical entries are never
overwritten or truncated -- giving a tamper-evident, reproducible audit log.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "audit.log"


class AuditLogger:
    """Thread-safe, append-only JSON-lines auditor."""

    def __init__(self, path: Path | str = _DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        # Touch the file so it exists on first use.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log(
        self,
        event: str,
        status: str = "OK",
        rule: str | None = None,
        confidence: float | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Append a single JSON line and return the record dict."""
        record = {
            "timestamp": self._now(),
            "event": event,
            "status": status,
            "rule": rule,
            "confidence": confidence,
            **params,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Read all audit entries back as a list of dicts (for the UI)."""
        entries: list[dict[str, Any]] = []
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipped malformed audit line: %s", line)
        return entries
