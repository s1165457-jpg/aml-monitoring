"""
aml_rules.py
============
Rule-based transaction monitoring engine.

Five typologies are implemented, each with an explicit threshold and rationale:

  1. STRUCTURING              - 3+ cash deposits in 7 days, each below the
                                HK$120,000 CDD threshold, totalling >= HK$300,000
  2. RAPID_IN_OUT             - inbound >= HK$100,000 followed within 48h by
                                outbound transfers >= 80% of the inflow
  3. HIGH_RISK_JURISDICTION   - 2+ transfers to high-risk jurisdictions, total
                                >= HK$150,000
  4. DORMANT_ACTIVATION       - 90+ day dormancy, then 5+ transactions within
                                7 days totalling >= HK$300,000
  5. VELOCITY                 - 10+ transactions within a rolling 24-hour window

Because ground_truth.csv records which customers were deliberately injected,
the engine can report precision / recall instead of just "number of alerts".

Outputs:
    outputs/alerts.csv        - one row per alert, ranked by risk score
    outputs/case_summary.csv  - one row per customer with aggregated risk
    outputs/evaluation.txt    - precision / recall against ground truth
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"

HIGH_RISK_JURISDICTIONS = {"JUR-A", "JUR-B", "JUR-C"}
CDD_THRESHOLD = 120_000
STRUCTURING_MIN_DEPOSIT = 80_000
STRUCTURING_TOTAL = 300_000
RAPID_IN_MIN = 100_000
RAPID_OUT_RATIO = 0.80
HR_TOTAL = 150_000
DORMANT_DAYS = 90
DORMANT_BURST_TOTAL = 300_000
VELOCITY_COUNT = 10

RULE_WEIGHTS = {
    "STRUCTURING": 40,
    "RAPID_IN_OUT": 35,
    "HIGH_RISK_JURISDICTION": 30,
    "DORMANT_ACTIVATION": 25,
    "VELOCITY": 20,
}


def make_alert(customer_id, rule, start, end, count, total, evidence, severity):
    return {
        "customer_id": customer_id,
        "rule": rule,
        "window_start": start,
        "window_end": end,
        "txn_count": count,
        "total_amount_hkd": round(float(total), 2),
        "severity": severity,
        "score": RULE_WEIGHTS[rule],
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Rule 1: structuring (smurfing below the CDD threshold)
# ---------------------------------------------------------------------------
def rule_structuring(df: pd.DataFrame) -> list[dict]:
    alerts = []
    cash = df[df["type"] == "CASH_DEPOSIT"]
    for cid, group in cash.groupby("customer_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(g)):
            window_end = g.loc[i, "timestamp"] + timedelta(days=7)
            win = g[(g["timestamp"] >= g.loc[i, "timestamp"]) & (g["timestamp"] <= window_end)]
            band = win[
                (win["amount_hkd"] >= STRUCTURING_MIN_DEPOSIT)
                & (win["amount_hkd"] < CDD_THRESHOLD)
            ]
            if len(band) >= 3 and band["amount_hkd"].sum() >= STRUCTURING_TOTAL:
                alerts.append(
                    make_alert(
                        cid,
                        "STRUCTURING",
                        band["timestamp"].min(),
                        band["timestamp"].max(),
                        len(band),
                        band["amount_hkd"].sum(),
                        f"{len(band)} cash deposits between HK${band['amount_hkd'].min():,.0f} "
                        f"and HK${band['amount_hkd'].max():,.0f}, each below the HK$120,000 CDD threshold",
                        "HIGH",
                    )
                )
                break
    return alerts


# ---------------------------------------------------------------------------
# Rule 2: rapid in / out (layering)
# ---------------------------------------------------------------------------
def rule_rapid_in_out(df: pd.DataFrame) -> list[dict]:
    alerts = []
    for cid, group in df.groupby("customer_id"):
        ins = group[(group["type"] == "TRANSFER_IN") & (group["amount_hkd"] >= RAPID_IN_MIN)]
        if ins.empty:
            continue
        for _, inflow in ins.iterrows():
            window_end = inflow["timestamp"] + timedelta(hours=48)
            outs = group[
                (group["type"] == "TRANSFER_OUT")
                & (group["timestamp"] >= inflow["timestamp"])
                & (group["timestamp"] <= window_end)
            ]
            if outs.empty:
                continue
            out_total = outs["amount_hkd"].sum()
            if out_total >= RAPID_OUT_RATIO * inflow["amount_hkd"]:
                alerts.append(
                    make_alert(
                        cid,
                        "RAPID_IN_OUT",
                        inflow["timestamp"],
                        outs["timestamp"].max(),
                        len(outs) + 1,
                        out_total,
                        f"Inflow HK${inflow['amount_hkd']:,.0f} from {inflow['counterparty_country']}; "
                        f"HK${out_total:,.0f} ({out_total / inflow['amount_hkd'] * 100:.0f}%) moved out "
                        f"within 48 hours",
                        "HIGH",
                    )
                )
                break
    return alerts


# ---------------------------------------------------------------------------
# Rule 3: high-risk jurisdiction exposure
# ---------------------------------------------------------------------------
def rule_high_risk_jurisdiction(df: pd.DataFrame) -> list[dict]:
    alerts = []
    hr = df[df["counterparty_country"].isin(HIGH_RISK_JURISDICTIONS)]
    for cid, group in hr.groupby("customer_id"):
        out = group[group["type"].isin(["TRANSFER_OUT", "TRANSFER_IN"])]
        if len(out) >= 2 and out["amount_hkd"].sum() >= HR_TOTAL:
            alerts.append(
                make_alert(
                    cid,
                    "HIGH_RISK_JURISDICTION",
                    out["timestamp"].min(),
                    out["timestamp"].max(),
                    len(out),
                    out["amount_hkd"].sum(),
                    f"{len(out)} transfers to/from {', '.join(sorted(out['counterparty_country'].unique()))} "
                    f"totalling HK${out['amount_hkd'].sum():,.0f}",
                    "MEDIUM",
                )
            )
    return alerts


# ---------------------------------------------------------------------------
# Rule 4: dormant account suddenly activated
# ---------------------------------------------------------------------------
def rule_dormant_activation(df: pd.DataFrame) -> list[dict]:
    alerts = []
    for cid, group in df.groupby("customer_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 6:
            continue
        gaps = g["timestamp"].diff()
        for idx in range(1, len(g)):
            if pd.isna(gaps.iloc[idx]) or gaps.iloc[idx] < timedelta(days=DORMANT_DAYS):
                continue
            burst_end = g.loc[idx, "timestamp"] + timedelta(days=7)
            burst = g[(g["timestamp"] >= g.loc[idx, "timestamp"]) & (g["timestamp"] <= burst_end)]
            if len(burst) >= 5 and burst["amount_hkd"].sum() >= DORMANT_BURST_TOTAL:
                alerts.append(
                    make_alert(
                        cid,
                        "DORMANT_ACTIVATION",
                        g.loc[idx, "timestamp"],
                        burst["timestamp"].max(),
                        len(burst),
                        burst["amount_hkd"].sum(),
                        f"{gaps.iloc[idx].days} days of inactivity, then {len(burst)} transactions "
                        f"in 7 days totalling HK${burst['amount_hkd'].sum():,.0f}",
                        "MEDIUM",
                    )
                )
                break
    return alerts


# ---------------------------------------------------------------------------
# Rule 5: transaction velocity
# ---------------------------------------------------------------------------
def rule_velocity(df: pd.DataFrame) -> list[dict]:
    alerts = []
    for cid, group in df.groupby("customer_id"):
        g = group.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(g)):
            window_end = g.loc[i, "timestamp"] + timedelta(hours=24)
            win = g[(g["timestamp"] >= g.loc[i, "timestamp"]) & (g["timestamp"] <= window_end)]
            if len(win) >= VELOCITY_COUNT:
                alerts.append(
                    make_alert(
                        cid,
                        "VELOCITY",
                        win["timestamp"].min(),
                        win["timestamp"].max(),
                        len(win),
                        win["amount_hkd"].sum(),
                        f"{len(win)} transactions within 24 hours across "
                        f"{win['channel'].nunique()} channels",
                        "LOW",
                    )
                )
                break
    return alerts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(alerts_df: pd.DataFrame, truth_df: pd.DataFrame) -> str:
    flagged = set(alerts_df["customer_id"])
    actual = set(truth_df["customer_id"])
    tp = len(flagged & actual)
    fp = len(flagged - actual)
    fn = len(actual - flagged)
    precision = tp / (tp + fp) * 100 if tp + fp else 0.0
    recall = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_typology = []
    for typology, group in truth_df.groupby("typology"):
        caught = len(set(group["customer_id"]) & flagged)
        per_typology.append(f"  {typology:<24} {caught}/{len(group)} caught")

    return (
        "RULE ENGINE EVALUATION (customer level)\n"
        + "=" * 55
        + "\n"
        f"Injected suspicious customers : {len(actual)}\n"
        f"Customers flagged             : {len(flagged)}\n"
        f"True positives                : {tp}\n"
        f"False positives               : {fp}\n"
        f"False negatives               : {fn}\n"
        f"Precision                     : {precision:.1f}%\n"
        f"Recall                        : {recall:.1f}%\n"
        f"F1 score                      : {f1:.1f}\n\n"
        "Recall by typology:\n" + "\n".join(per_typology) + "\n"
    )


def main() -> None:
    if not OUT.exists():
        OUT.mkdir(parents=True)
    txns = pd.read_csv(DATA / "transactions.csv", parse_dates=["timestamp"])
    truth = pd.read_csv(DATA / "ground_truth.csv")

    alerts = []
    alerts += rule_structuring(txns)
    alerts += rule_rapid_in_out(txns)
    alerts += rule_high_risk_jurisdiction(txns)
    alerts += rule_dormant_activation(txns)
    alerts += rule_velocity(txns)

    alerts_df = pd.DataFrame(alerts)
    alerts_df = alerts_df.sort_values(
        ["score", "total_amount_hkd"], ascending=[False, False]
    ).reset_index(drop=True)
    alerts_df.insert(0, "alert_id", [f"ALT-{i:04d}" for i in range(1, len(alerts_df) + 1)])

    alerts_df.to_csv(OUT / "alerts.csv", index=False, encoding="utf-8-sig")

    # Customer-level roll-up: total risk score and which rules fired
    if not alerts_df.empty:
        summary = (
            alerts_df.groupby("customer_id")
            .agg(
                rules_fired=("rule", lambda s: " | ".join(sorted(set(s)))),
                alert_count=("alert_id", "count"),
                risk_score=("score", "sum"),
                total_amount_hkd=("total_amount_hkd", "sum"),
            )
            .reset_index()
            .sort_values("risk_score", ascending=False)
        )
        summary["risk_score"] = summary["risk_score"].clip(upper=100)
        summary.to_csv(OUT / "case_summary.csv", index=False, encoding="utf-8-sig")

    report = evaluate(alerts_df, truth)
    (OUT / "evaluation.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Alerts written : {len(alerts_df)} -> {OUT / 'alerts.csv'}")


if __name__ == "__main__":
    main()
