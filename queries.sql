-- ============================================================================
-- aml-monitoring / queries.sql
-- SQL implementations of the five monitoring rules.
-- Dialect: ANSI SQL with window functions (PostgreSQL / MySQL 8+ / SQL Server).
-- Assumes tables `transactions` and `customers` loaded from data/*.csv
--   transactions(txn_id, customer_id, timestamp, type, amount_hkd, channel,
--                counterparty_country, counterparty_id)
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 1. STRUCTURING
--    3+ cash deposits in a rolling 7-day window, each between HKD 80,000 and
--    119,999, totalling at least HKD 300,000.
-- ----------------------------------------------------------------------------
WITH cash AS (
    SELECT customer_id, timestamp, amount_hkd
    FROM transactions
    WHERE type = 'CASH_DEPOSIT'
      AND amount_hkd >= 80000
      AND amount_hkd < 120000
),
windows AS (
    SELECT
        c1.customer_id,
        c1.timestamp                                   AS window_start,
        COUNT(*)                                       AS deposit_count,
        SUM(c1.amount_hkd)                             AS total_amount
    FROM cash c1
    JOIN cash c2
      ON c2.customer_id = c1.customer_id
     AND c2.timestamp BETWEEN c1.timestamp AND c1.timestamp + INTERVAL '7 days'
    GROUP BY c1.customer_id, c1.timestamp
)
SELECT *
FROM windows
WHERE deposit_count >= 3
  AND total_amount >= 300000
ORDER BY total_amount DESC;


-- ----------------------------------------------------------------------------
-- 2. RAPID IN / OUT (layering)
--    Inflow >= HKD 100,000 followed within 48 hours by outflows >= 80% of it.
-- ----------------------------------------------------------------------------
WITH inflows AS (
    SELECT txn_id, customer_id, timestamp AS in_time, amount_hkd AS inflow
    FROM transactions
    WHERE type = 'TRANSFER_IN'
      AND amount_hkd >= 100000
)
SELECT
    i.customer_id,
    i.in_time,
    i.inflow,
    COUNT(o.txn_id)                     AS out_count,
    SUM(o.amount_hkd)                   AS outflow,
    ROUND(100.0 * SUM(o.amount_hkd) / i.inflow, 1) AS pass_through_pct
FROM inflows i
JOIN transactions o
  ON o.customer_id = i.customer_id
 AND o.type = 'TRANSFER_OUT'
 AND o.timestamp BETWEEN i.in_time AND i.in_time + INTERVAL '48 hours'
GROUP BY i.customer_id, i.in_time, i.inflow
HAVING SUM(o.amount_hkd) >= 0.8 * i.inflow
ORDER BY pass_through_pct DESC;


-- ----------------------------------------------------------------------------
-- 3. HIGH-RISK JURISDICTION
--    2+ transfers to/from listed jurisdictions totalling >= HKD 150,000.
-- ----------------------------------------------------------------------------
SELECT
    customer_id,
    COUNT(*)                        AS txn_count,
    ROUND(SUM(amount_hkd), 2)       AS total_amount,
    MIN(timestamp)                  AS first_seen,
    MAX(timestamp)                  AS last_seen
FROM transactions
WHERE counterparty_country IN ('JUR-A', 'JUR-B', 'JUR-C')
  AND type IN ('TRANSFER_IN', 'TRANSFER_OUT')
GROUP BY customer_id
HAVING COUNT(*) >= 2
   AND SUM(amount_hkd) >= 150000
ORDER BY total_amount DESC;


-- ----------------------------------------------------------------------------
-- 4. DORMANT ACTIVATION
--    Previous transaction >= 90 days earlier, then a burst of >= 5 transactions
--    within 7 days totalling >= HKD 300,000.
--    LAG() gives the dormancy gap without a self-join.
-- ----------------------------------------------------------------------------
WITH ordered AS (
    SELECT
        customer_id,
        timestamp,
        amount_hkd,
        LAG(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp) AS prev_ts
    FROM transactions
),
reactivation AS (
    SELECT customer_id, timestamp AS burst_start
    FROM ordered
    WHERE prev_ts IS NOT NULL
      AND timestamp - prev_ts >= INTERVAL '90 days'
)
SELECT
    r.customer_id,
    r.burst_start,
    COUNT(t.txn_id)                 AS burst_count,
    ROUND(SUM(t.amount_hkd), 2)     AS burst_total
FROM reactivation r
JOIN transactions t
  ON t.customer_id = r.customer_id
 AND t.timestamp BETWEEN r.burst_start AND r.burst_start + INTERVAL '7 days'
GROUP BY r.customer_id, r.burst_start
HAVING COUNT(t.txn_id) >= 5
   AND SUM(t.amount_hkd) >= 300000
ORDER BY burst_total DESC;


-- ----------------------------------------------------------------------------
-- 5. VELOCITY
--    10+ transactions inside any rolling 24-hour window.
-- ----------------------------------------------------------------------------
SELECT
    c1.customer_id,
    c1.timestamp                                    AS window_start,
    COUNT(*)                                        AS txn_count,
    ROUND(SUM(c1.amount_hkd), 2)                    AS total_amount
FROM transactions c1
JOIN transactions c2
  ON c2.customer_id = c1.customer_id
 AND c2.timestamp BETWEEN c1.timestamp AND c1.timestamp + INTERVAL '24 hours'
GROUP BY c1.customer_id, c1.timestamp
HAVING COUNT(*) >= 10
ORDER BY txn_count DESC;


-- ----------------------------------------------------------------------------
-- 6. CUSTOMER-LEVEL RISK ROLL-UP
--    Consolidates alerts per customer, the way a case would be opened.
-- ----------------------------------------------------------------------------
WITH alert_flags AS (
    SELECT customer_id, 'STRUCTURING' AS rule FROM transactions
    WHERE type = 'CASH_DEPOSIT' AND amount_hkd BETWEEN 80000 AND 119999
    GROUP BY customer_id HAVING COUNT(*) >= 3
    UNION ALL
    SELECT customer_id, 'HIGH_RISK_JURISDICTION' FROM transactions
    WHERE counterparty_country IN ('JUR-A','JUR-B','JUR-C')
    GROUP BY customer_id HAVING COUNT(*) >= 2
)
SELECT
    customer_id,
    STRING_AGG(rule, ' | ' ORDER BY rule)   AS rules_fired,
    COUNT(*)                                AS rule_count
FROM alert_flags
GROUP BY customer_id
ORDER BY rule_count DESC;


-- ----------------------------------------------------------------------------
-- 7. PEER COMPARISON
--    Flags customers whose monthly volume sits far above their segment median -
--    a simple behavioural baseline before moving to anomaly detection.
-- ----------------------------------------------------------------------------
WITH monthly AS (
    SELECT
        t.customer_id,
        c.segment,
        DATE_TRUNC('month', t.timestamp) AS month,
        SUM(t.amount_hkd)                AS volume
    FROM transactions t
    JOIN customers c ON c.customer_id = t.customer_id
    GROUP BY t.customer_id, c.segment, DATE_TRUNC('month', t.timestamp)
),
benchmarked AS (
    SELECT
        *,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY volume)
            OVER (PARTITION BY segment, month) AS segment_median
    FROM monthly
)
SELECT *
FROM benchmarked
WHERE volume > 5 * segment_median
ORDER BY volume DESC;


-- ----------------------------------------------------------------------------
-- 8. ALERT DISPOSITION TRACKING
--    Every alert needs a documented outcome - the audit trail regulators ask for.
--    Assumes alerts(id, customer_id, rule, score, disposition) where disposition
--    is one of OPEN / ESCALATED / CLOSED_LEGITIMATE / STR_FILED.
-- ----------------------------------------------------------------------------
SELECT
    disposition,
    COUNT(*)                                             AS alerts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)    AS pct_of_total
FROM alerts
GROUP BY disposition
ORDER BY alerts DESC;
