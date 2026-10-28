# aml-monitoring

Rule-based transaction monitoring engine with a measurable evaluation loop — built to practise the core workflow of an AML analyst: **monitor → alert → investigate → document → disposition**.

Built on 4,950 synthetic transactions across 200 customers over 12 months (2025), with 21 known suspicious cases injected so rule performance can be **measured**, not assumed.

## Results

| Metric | Value |
|---|---|
| Transactions monitored | 4,950 (HKD 68.4M) |
| Alerts generated | 28 |
| Customers flagged | 23 |
| Precision | **91.3%** |
| Recall | **100%** |
| F1 | **95.5** |

Recall by typology:

| Typology | Caught |
|---|---|
| Structuring | 8 / 8 |
| Rapid in-out (layering) | 4 / 4 |
| High-risk jurisdiction | 3 / 3 |
| Velocity | 3 / 3 |
| Dormant activation | 3 / 3 |

Two false positives were left in the dataset deliberately (a property-sale cash deposit sequence and a 12-person payroll run) and dispositioned in `data/reviewed_false_positives.csv` — the alert-to-disposition loop is part of the exercise, not an afterthought.

## Rules

| Rule | Threshold | Why this threshold |
|---|---|---|
| `STRUCTURING` | ≥3 cash deposits in 7 days, each HKD 80,000–119,999, total ≥ HKD 300,000 | Each deposit stays under the HK$120,000 CDD trigger while the aggregate is material |
| `RAPID_IN_OUT` | Inflow ≥ HKD 100,000, ≥80% transferred out within 48h | Layering signal: funds pass through without economic purpose |
| `HIGH_RISK_JURISDICTION` | ≥2 transfers to/from a listed jurisdiction, total ≥ HKD 150,000 | Jurisdiction risk weighting |
| `DORMANT_ACTIVATION` | ≥90 days inactive, then ≥5 transactions in 7 days totalling ≥ HKD 300,000 | Dormant accounts reactivated for a purpose inconsistent with history |
| `VELOCITY` | ≥10 transactions in any rolling 24h window | Account-mule / testing behaviour |

Risk score = weighted sum of fired rules (structuring 40, rapid in-out 35, high-risk jurisdiction 30, dormant 25, velocity 20), capped at 100.

> Thresholds are documented assumptions for a synthetic dataset, not regulatory limits. In production they are calibrated against the institution's risk appetite and alert-volume capacity.

## Repository structure

```
aml-monitoring/
├── generate_data.py                    # synthetic customers + transactions + injected typologies
├── aml_rules.py                        # rule engine, scoring, evaluation
├── queries.sql                         # SQL versions of the rules (window functions / CTEs)
├── data/
│   ├── customers.csv
│   ├── transactions.csv
│   ├── ground_truth.csv                # injected cases (for evaluation only)
│   └── reviewed_false_positives.csv    # legitimate outliers + disposition rationale
├── outputs/
│   ├── alerts.csv                      # one row per alert, ranked by score
│   ├── case_summary.csv                # customer-level risk roll-up
│   └── evaluation.txt                  # precision / recall report
└── reports/
    └── STR_case_C0029.md               # STR-style narrative for the highest-risk case
```

## How to run

```bash
pip install pandas numpy
python generate_data.py     # rebuilds data/ from a fixed seed (42) — fully reproducible
python aml_rules.py         # prints evaluation, writes outputs/
```

## Design notes

- **Reproducibility.** A fixed seed means every run produces identical data and identical metrics — so a threshold change can be attributed to the rule, not to data drift.
- **Ground truth by construction.** Injecting known typologies is the only way to state precision and recall honestly on synthetic data. The ground-truth file is never read by the rule engine.
- **Two false positives are intentional.** A perfect score would say more about the dataset than about the rules.
- **Rules before ML.** Rule-based monitoring is the industry baseline; it is explainable, auditable, and easy to tune. An anomaly-detection layer (e.g. isolation forest) is the natural next step, but only once the rule baseline and disposition process are solid.

## Limitations

- Synthetic data: generation assumes independence between customers, which real transaction networks do not have.
- No name screening, sanctions screening, or fuzzy matching against watchlists.
- No alert deduplication across rules beyond customer-level roll-up — in production, one case would consolidate multiple alerts.
- Jurisdictions are anonymised (`JUR-A/B/C`) rather than mapped to real lists.
- No ML layer, no network analysis, no case management workflow.

## Disclaimer

Educational portfolio project. All data is synthetic; no real customer, institution, or regulatory filing is represented.
