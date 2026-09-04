"""
Synthetic Data Generator
========================
Generates realistic, noisy multi-source financial records across three CSV
sources that simulate the real-world reconciliation pain points:

    gateway_logs.csv      -> payment gateway transactions
    bank_statement.csv    -> bank settlement statements
    internal_ledger.csv   -> internal merchant ledger entries

Noise distributions intentionally emulate production realities:
    * 65% : clean exact matches (identical txn_id, order_id, amount)
    * 17% : fee deductions (bank statement is ~2% lower = gateway fees)
    * 10% : timestamp skews (settlement offset +26h across a date boundary)
    *  8% : unresolved exceptions (missing refs / unexplainable gaps)

All monetary values are integers stored in *paise* to avoid float error.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default layout constants (packages the loaded env config for writer).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Distribution weights (sum to 100)
DISTRIBUTION = {
    "exact": 0.65,
    "fee": 0.17,
    "skew": 0.10,
    "exception": 0.08,
}

# Gateway fee rate used to simulate deduction (0.02 => 2%)
GATEWAY_FEE_RATE = 0.02

# Timestamp skew offset across a date boundary (hours)
SKEW_OFFSET_HOURS = 26


def _random_amount_paise(rng: random.Random) -> int:
    """Random transaction amount in paise between INR 150 and INR 15,000."""
    rupees = rng.randint(150, 15000)
    return rupees * 100


def _apply_fee(amount_paise: int, rng: random.Random) -> int:
    """
    Apply a gateway fee deduction (0% - ~2%) and return bank value.

    The deduction sits between 0% and the configured fee ceiling.
    """
    fee_ratio = rng.uniform(0.0, GATEWAY_FEE_RATE)
    fee_paise = int(round(amount_paise * fee_ratio))
    return amount_paise - fee_paise


def _base_timestamp(rng: random.Random) -> datetime:
    """Random settlement timestamp for the gateway log (UTC)."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 6, 30, tzinfo=timezone.utc)
    span = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, span))


def generate_records(n: int, seed: int | None = None) -> pd.DataFrame:
    """
    Generate a reconciled, noisy batch of transactions.

    Returns a single DataFrame with one row per source record. A `_kind`
    column tags the noise class and a `_source` column tags the CSV.
    """
    rng = random.Random(seed)
    rows: list[dict] = []

    # Determine how many records fall into each class.
    counts = _class_counts(n, rng)
    txn_counter = 0

    for kind, count in counts.items():
        for _ in range(count):
            txn_counter += 1
            txn_id = f"TXN{txn_counter:06d}"
            order_id = f"ORD{txn_counter:04d}"
            amount_paise = _random_amount_paise(rng)
            ts = _base_timestamp(rng)

            rows.append(
                {
                    "_source": "gateway",
                    "_kind": kind,
                    "txn_id": txn_id,
                    "order_id": order_id,
                    "amount_paise": amount_paise,
                    "timestamp": ts.isoformat(),
                }
            )

            if kind == "exact":
                # Identical on every join key.
                rows.append(
                    {
                        "_source": "bank",
                        "_kind": kind,
                        "txn_id": txn_id,
                        "order_id": order_id,
                        "amount_paise": amount_paise,
                        "timestamp": (ts + timedelta(days=1)).isoformat(),
                    }
                )
                rows.append(
                    {
                        "_source": "ledger",
                        "_kind": kind,
                        "txn_id": txn_id,
                        "order_id": order_id,
                        "amount_paise": amount_paise,
                        "timestamp": ts.isoformat(),
                    }
                )
                continue

            if kind == "fee":
                # Bank statement reflects gateway fee deduction (~2% lower).
                bank_amount = _apply_fee(amount_paise, rng)
                rows.append(
                    {
                        "_source": "bank",
                        "_kind": kind,
                        "txn_id": txn_id,
                        "order_id": order_id,
                        "amount_paise": bank_amount,
                        "timestamp": (ts + timedelta(days=1)).isoformat(),
                    }
                )
                rows.append(
                    {
                        "_source": "ledger",
                        "_kind": kind,
                        "txn_id": txn_id,
                        "order_id": order_id,
                        "amount_paise": amount_paise,
                        "timestamp": ts.isoformat(),
                    }
                )
                continue

            if kind == "skew":
                # Settlement timestamp offset +26h crossing a calendar boundary.
                skewed = ts + timedelta(hours=SKEW_OFFSET_HOURS)
                rows.append(
                    {
                        "_source": "bank",
                        "_kind": kind,
                        "txn_id": txn_id,
                        "order_id": order_id,
                        "amount_paise": amount_paise,
                        "timestamp": skewed.isoformat(),
                    }
                )
                rows.append(
                    {
                        "_source": "ledger",
                        "_kind": kind,
                        "txn_id": txn_id,
                        "order_id": order_id,
                        "amount_paise": amount_paise,
                        "timestamp": ts.isoformat(),
                    }
                )
                continue

            # kind == "exception": unresolved edge case.
            # Bank record is either missing its reference key or carries an
            # unexplainable rupee gap (~INR 450 / 45000 paise lower).
            if rng.random() < 0.5:
                bank_amount = amount_paise  # missing ref key only
            else:
                bank_amount = amount_paise - 45000  # rupee gap ~INR 450
            rows.append(
                {
                    "_source": "bank",
                    "_kind": kind,
                    "txn_id": txn_id,
                    "order_id": order_id,
                    "amount_paise": bank_amount,
                    "timestamp": (ts + timedelta(days=1)).isoformat(),
                }
            )
            rows.append(
                {
                    "_source": "ledger",
                    "_kind": kind,
                    "txn_id": txn_id,
                    "order_id": order_id,
                    "amount_paise": amount_paise,
                    "timestamp": ts.isoformat(),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _class_counts(n: int, rng: random.Random) -> dict[str, int]:
    """Split n transactions across noise classes following DISTRIBUTION."""
    counts: dict[str, int] = {}
    total_assigned = 0
    keys = list(DISTRIBUTION.keys())
    for i, kind in enumerate(keys):
        if i == len(keys) - 1:
            counts[kind] = n - total_assigned
        else:
            counts[kind] = int(round(n * DISTRIBUTION[kind]))
            total_assigned += counts[kind]
    return counts


def write_csvs(df: pd.DataFrame, data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Split the unified frame into the three source CSVs. Returns paths."""
    data_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "gateway_logs.csv": df[df["_source"] == "gateway"],
        "bank_statement.csv": df[df["_source"] == "bank"],
        "internal_ledger.csv": df[df["_source"] == "ledger"],
    }

    paths: dict[str, Path] = {}
    for fname, subset in sources.items():
        drop = subset.drop(columns=["_source", "_kind"])
        # Sort each file deterministically for stable output.
        drop = drop.sort_values(["txn_id", "timestamp"]).reset_index(drop=True)
        path = data_dir / fname
        drop.to_csv(path, index=False)
        paths[fname] = path
        logger.info("Wrote %s records to %s", len(drop), path)
    return paths


def generate_batch(n: int = 100, seed: int | None = None) -> dict[str, Path]:
    """Full pipeline: generate records and write the three CSVs."""
    df = generate_records(n, seed=seed)
    return write_csvs(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_batch(n=100, seed=42)
