"""
generate_data.py
================
Creates a synthetic, fully reproducible transaction dataset for AML rule
testing: 200 retail customers over 12 months, with a known set of suspicious
typologies injected so that rule performance can be measured (precision /
recall) instead of guessed.

Outputs (data/):
    customers.csv      - 200 customer profiles
    transactions.csv   - ~4k transactions
    ground_truth.csv   - which customers were injected with which typology

All names are masked; all IDs are synthetic. No real customer data is used.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

N_CUSTOMERS = 200
START = datetime(2025, 1, 1)
END = datetime(2025, 12, 31)

HIGH_RISK_JURISDICTIONS = ["JUR-A", "JUR-B", "JUR-C"]
CHANNELS = ["BRANCH", "ATM", "ONLINE", "MOBILE"]
TXN_TYPES = ["CASH_DEPOSIT", "TRANSFER_IN", "TRANSFER_OUT", "CARD_PAYMENT"]


def random_ts(start: datetime, end: datetime) -> datetime:
    delta = int((end - start).total_seconds())
    return start + timedelta(seconds=int(RNG.integers(0, delta)))


# ---------------------------------------------------------------------------
# 1. Customers
# ---------------------------------------------------------------------------
def build_customers() -> pd.DataFrame:
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        cid = f"C{i:04d}"
        rows.append(
            {
                "customer_id": cid,
                "name_masked": f"Customer {i:03d}",
                "segment": RNG.choice(["RETAIL", "SME"], p=[0.85, 0.15]),
                "occupation": RNG.choice(
                    ["Salaried", "Self-employed", "Student", "Retired", "Company director"],
                    p=[0.55, 0.2, 0.1, 0.1, 0.05],
                ),
                "home_country": "HK",
                "onboard_date": (START - timedelta(days=int(RNG.integers(30, 2500)))).date().isoformat(),
                "risk_rating": RNG.choice(["LOW", "MEDIUM", "HIGH"], p=[0.7, 0.25, 0.05]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Normal (benign) activity
# ---------------------------------------------------------------------------
def normal_txns(customer_ids: list[str]) -> list[dict]:
    rows = []
    tid = 0
    for cid in customer_ids:
        n = int(RNG.integers(8, 45))
        for _ in range(n):
            ttype = RNG.choice(TXN_TYPES, p=[0.25, 0.2, 0.25, 0.3])
            if ttype == "CARD_PAYMENT":
                amount = float(np.round(RNG.gamma(2.0, 600), 2))
            elif ttype == "CASH_DEPOSIT":
                amount = float(np.round(RNG.gamma(2.5, 4000), 2))
            else:
                amount = float(np.round(RNG.gamma(2.5, 8000), 2))

            rows.append(
                {
                    "txn_id": f"T{tid:06d}",
                    "customer_id": cid,
                    "timestamp": random_ts(START, END),
                    "type": ttype,
                    "amount_hkd": amount,
                    "channel": RNG.choice(CHANNELS),
                    "counterparty_country": "HK" if RNG.random() < 0.85 else RNG.choice(["SG", "CN", "GB", "US"]),
                    "counterparty_id": f"CP{int(RNG.integers(1, 500)):04d}",
                }
            )
            tid += 1
    return rows


# ---------------------------------------------------------------------------
# 3. Injected typologies
# ---------------------------------------------------------------------------
def inject_structuring(cid: str, start_tid: int) -> list[dict]:
    """4 cash deposits just below the HK$120,000 CDD threshold, different branches."""
    rows = []
    base = random_ts(START + timedelta(days=30), END - timedelta(days=40))
    for k in range(4):
        rows.append(
            {
                "txn_id": f"T{start_tid + k:06d}",
                "customer_id": cid,
                "timestamp": base + timedelta(days=k, hours=int(RNG.integers(1, 6))),
                "type": "CASH_DEPOSIT",
                "amount_hkd": float(np.round(RNG.uniform(95000, 118000), 2)),
                "channel": "BRANCH",
                "counterparty_country": "HK",
                "counterparty_id": "SELF",
            }
        )
    return rows


def inject_rapid_in_out(cid: str, start_tid: int) -> list[dict]:
    """Large inbound transfer, 85%+ moved out within 48 hours to another jurisdiction."""
    rows = []
    base = random_ts(START + timedelta(days=60), END - timedelta(days=60))
    inflow = float(np.round(RNG.uniform(250000, 600000), 2))
    rows.append(
        {
            "txn_id": f"T{start_tid:06d}",
            "customer_id": cid,
            "timestamp": base,
            "type": "TRANSFER_IN",
            "amount_hkd": inflow,
            "channel": "ONLINE",
            "counterparty_country": RNG.choice(["JUR-A", "JUR-B"]),
            "counterparty_id": f"CP{int(RNG.integers(600, 800)):04d}",
        }
    )
    out_total = inflow * float(RNG.uniform(0.85, 0.95))
    for k in range(2):
        rows.append(
            {
                "txn_id": f"T{start_tid + 1 + k:06d}",
                "customer_id": cid,
                "timestamp": base + timedelta(hours=int(RNG.integers(2, 40))),
                "type": "TRANSFER_OUT",
                "amount_hkd": float(np.round(out_total / 2, 2)),
                "channel": "ONLINE",
                "counterparty_country": RNG.choice(["JUR-B", "JUR-C"]),
                "counterparty_id": f"CP{int(RNG.integers(600, 800)):04d}",
            }
        )
    return rows


def inject_high_risk_jurisdiction(cid: str, start_tid: int) -> list[dict]:
    """Repeated transfers to a high-risk jurisdiction with no clear business rationale."""
    rows = []
    base = random_ts(START + timedelta(days=90), END - timedelta(days=60))
    for k in range(3):
        rows.append(
            {
                "txn_id": f"T{start_tid + k:06d}",
                "customer_id": cid,
                "timestamp": base + timedelta(days=k * 9),
                "type": "TRANSFER_OUT",
                "amount_hkd": float(np.round(RNG.uniform(80000, 180000), 2)),
                "channel": "ONLINE",
                "counterparty_country": "JUR-A",
                "counterparty_id": f"CP{int(RNG.integers(600, 800)):04d}",
            }
        )
    return rows


def inject_dormant_activation(cid: str, start_tid: int) -> list[dict]:
    """No activity for 6 months, then a burst of transfers in 5 days."""
    rows = []
    base = random_ts(START + timedelta(days=240), END - timedelta(days=20))
    for k in range(6):
        rows.append(
            {
                "txn_id": f"T{start_tid + k:06d}",
                "customer_id": cid,
                "timestamp": base + timedelta(hours=int(RNG.integers(0, 120))),
                "type": RNG.choice(["TRANSFER_IN", "TRANSFER_OUT", "CASH_DEPOSIT"]),
                "amount_hkd": float(np.round(RNG.uniform(60000, 140000), 2)),
                "channel": RNG.choice(["ONLINE", "ATM"]),
                "counterparty_country": RNG.choice(["HK", "JUR-B"]),
                "counterparty_id": f"CP{int(RNG.integers(1, 800)):04d}",
            }
        )
    return rows


def inject_velocity(cid: str, start_tid: int) -> list[dict]:
    """12 small transactions within 6 hours - classic account-mule behaviour."""
    rows = []
    base = random_ts(START + timedelta(days=45), END - timedelta(days=45))
    for k in range(12):
        rows.append(
            {
                "txn_id": f"T{start_tid + k:06d}",
                "customer_id": cid,
                "timestamp": base + timedelta(minutes=int(RNG.integers(0, 360))),
                "type": RNG.choice(["TRANSFER_IN", "TRANSFER_OUT"]),
                "amount_hkd": float(np.round(RNG.uniform(3000, 20000), 2)),
                "channel": "MOBILE",
                "counterparty_country": "HK",
                "counterparty_id": f"CP{int(RNG.integers(1, 800)):04d}",
            }
        )
    return rows


INJECTORS = {
    "STRUCTURING": (inject_structuring, 8),
    "RAPID_IN_OUT": (inject_rapid_in_out, 4),
    "HIGH_RISK_JURISDICTION": (inject_high_risk_jurisdiction, 3),
    "DORMANT_ACTIVATION": (inject_dormant_activation, 3),
    "VELOCITY": (inject_velocity, 3),
}


def add_prior_history(cid: str, tid: int, rows: list[dict]) -> int:
    """Give a customer a normal early-year history so a later dormancy is real."""
    for k in range(6):
        rows.append(
            {
                "txn_id": f"T{tid + k:06d}",
                "customer_id": cid,
                "timestamp": datetime(2025, 1, 5) + timedelta(days=k * 9, hours=int(RNG.integers(0, 12))),
                "type": RNG.choice(["CARD_PAYMENT", "CASH_DEPOSIT", "TRANSFER_OUT"]),
                "amount_hkd": float(np.round(RNG.uniform(2000, 45000), 2)),
                "channel": RNG.choice(CHANNELS),
                "counterparty_country": "HK",
                "counterparty_id": f"CP{int(RNG.integers(1, 500)):04d}",
            }
        )
    return tid + 10


def add_legitimate_outlier(cid: str, tid: int, rows: list[dict], kind: str) -> int:
    """Legitimate behaviour that should still trip a rule - realistic false positives."""
    if kind == "property":
        base = datetime(2025, 6, 3, 10, 0)
        for k in range(3):
            rows.append(
                {
                    "txn_id": f"T{tid + k:06d}",
                    "customer_id": cid,
                    "timestamp": base + timedelta(days=k),
                    "type": "CASH_DEPOSIT",
                    "amount_hkd": float(np.round(RNG.uniform(100000, 115000), 2)),
                    "channel": "BRANCH",
                    "counterparty_country": "HK",
                    "counterparty_id": "SELF",
                }
            )
        return tid + 10

    base = datetime(2025, 8, 28, 9, 0)
    for k in range(12):
        rows.append(
            {
                "txn_id": f"T{tid + k:06d}",
                "customer_id": cid,
                "timestamp": base + timedelta(minutes=k * 12),
                "type": "TRANSFER_OUT",
                "amount_hkd": float(np.round(RNG.uniform(12000, 28000), 2)),
                "channel": "ONLINE",
                "counterparty_country": "HK",
                "counterparty_id": f"CP{int(RNG.integers(1, 500)):04d}",
            }
        )
    return tid + 10


def main() -> None:
    if not DATA.exists():
        DATA.mkdir(parents=True)

    customers = build_customers()
    all_ids = customers["customer_id"].tolist()

    # Reserve customers for injection so benign and suspicious sets do not overlap
    n_cases = sum(n for _, n in INJECTORS.values())
    injected_ids = RNG.choice(all_ids, size=n_cases, replace=False)
    benign_ids = [c for c in all_ids if c not in set(injected_ids)]

    rows = normal_txns(benign_ids)
    truth = []
    tid = 100000
    idx = 0
    case_no = 0

    for typology, (fn, count) in INJECTORS.items():
        for _ in range(count):
            cid = injected_ids[idx]
            idx += 1
            case_no += 1
            case_id = f"CASE-{case_no:03d}"
            injected = fn(cid, tid)
            tid += len(injected) + 10
            rows.extend(injected)
            truth.append({"case_id": case_id, "customer_id": cid, "typology": typology})

    # Dormancy only means something if there is prior history to go dormant from
    for case in truth:
        if case["typology"] == "DORMANT_ACTIVATION":
            tid = add_prior_history(case["customer_id"], tid, rows)

    # Deliberate legitimate outliers: they should trip rules and be dispositioned
    # as false positives during review (property sale, payroll run).
    noise_ids = [c for c in benign_ids][:2]
    tid = add_legitimate_outlier(noise_ids[0], tid, rows, kind="property")
    tid = add_legitimate_outlier(noise_ids[1], tid, rows, kind="payroll")
    reviewed = pd.DataFrame(
        [
            {
                "customer_id": noise_ids[0],
                "reason_flagged": "STRUCTURING",
                "disposition": "CLOSED - LEGITIMATE",
                "rationale": "Proceeds from property sale; sale and purchase agreement obtained, source of funds verified.",
            },
            {
                "customer_id": noise_ids[1],
                "reason_flagged": "VELOCITY",
                "disposition": "CLOSED - LEGITIMATE",
                "rationale": "Monthly payroll run for a 12-staff SME; reconciled to payroll register.",
            },
        ]
    )

    txns = pd.DataFrame(rows)
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    # Re-number transaction IDs after sorting so they read chronologically
    txns["txn_id"] = [f"T{i:06d}" for i in range(len(txns))]

    customers.to_csv(DATA / "customers.csv", index=False, encoding="utf-8-sig")
    txns.to_csv(DATA / "transactions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(truth).to_csv(DATA / "ground_truth.csv", index=False, encoding="utf-8-sig")
    reviewed.to_csv(DATA / "reviewed_false_positives.csv", index=False, encoding="utf-8-sig")

    print(f"Customers      : {len(customers):,}")
    print(f"Transactions   : {len(txns):,}")
    print(f"Injected cases : {len(truth)} across {len(set(t['customer_id'] for t in truth))} customers")
    print(f"Period         : {txns['timestamp'].min():%Y-%m-%d} to {txns['timestamp'].max():%Y-%m-%d}")
    print(f"Total value    : HKD {txns['amount_hkd'].sum():,.0f}")


if __name__ == "__main__":
    main()
