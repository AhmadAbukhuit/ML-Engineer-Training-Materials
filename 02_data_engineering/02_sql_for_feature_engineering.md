# Advanced SQL for Machine Learning Feature Engineering

Relational databases and cloud data warehouses (PostgreSQL, Snowflake, BigQuery) are the primary source of truth for enterprise features. Rather than pulling raw event tables into Python and running slow loops, production ML engineers compute feature aggregations directly in the SQL database.

---

## 1. Window Functions (The Foundation of Feature Engineering)

A standard `GROUP BY` collapses multiple rows into a single summary row. In contrast, a **Window Function** computes aggregations over a subset of rows (the "window") while **preserving individual row identities**.

### Syntax Template
```sql
FUNCTION(...) OVER (
    PARTITION BY entity_id
    ORDER BY event_time
    ROWS BETWEEN <start> AND <end>
)
```

---

## 2. Essential Window Function Patterns for ML

### Pattern A: Rolling Aggregations (e.g., 7-Day Rolling Spend)
Calculating rolling feature behavior (velocity, mean spend, variance) for a specific user:

```sql
SELECT
    user_id,
    transaction_time,
    amount,
    -- 7-transaction moving average
    AVG(amount) OVER (
        PARTITION BY user_id
        ORDER BY transaction_time
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_amount_7tx,
    
    -- Cumulative transaction count up to this point
    COUNT(transaction_id) OVER (
        PARTITION BY user_id
        ORDER BY transaction_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_transaction_count
FROM transactions;
```

### Pattern B: Time Deltas and Sequence Shifts (`LAG` and `LEAD`)
To detect abnormal velocity or bot behavior (e.g., seconds elapsed since last login):

```sql
SELECT
    user_id,
    login_time,
    -- Timestamp of the previous login
    LAG(login_time, 1) OVER (
        PARTITION BY user_id
        ORDER BY login_time
    ) AS prev_login_time,
    
    -- Time elapsed in seconds between consecutive logins
    EXTRACT(EPOCH FROM (login_time - LAG(login_time, 1) OVER (
        PARTITION BY user_id
        ORDER BY login_time
    ))) AS seconds_since_last_login
FROM user_logins;
```

### Pattern C: Deduplication & Finding the Latest Record (`ROW_NUMBER`)
In distributed logs, events arrive out of order or with duplicate deliveries. Use `ROW_NUMBER()` to select the latest state for each entity:

```sql
WITH RankedEvents AS (
    SELECT
        device_id,
        battery_level,
        status,
        timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY device_id
            ORDER BY timestamp DESC
        ) AS rn
    FROM device_heartbeats
)
SELECT
    device_id,
    battery_level,
    status,
    timestamp AS latest_heartbeat
FROM RankedEvents
WHERE rn = 1;
```

---

## 3. Common Table Expressions (CTEs) for Multi-Stage Feature Pipelines

CTEs (`WITH ... AS (...)`) improve readability, maintainability, and allow query planners to optimize intermediate feature steps:

```sql
WITH RawTransactions AS (
    SELECT 
        user_id,
        amount,
        created_at,
        is_fraud
    FROM transactions
    WHERE created_at >= NOW() - INTERVAL '90 days'
),
UserAggregates AS (
    SELECT
        user_id,
        COUNT(*) AS total_tx_90d,
        SUM(amount) AS total_spend_90d,
        AVG(amount) AS avg_spend_90d,
        STDDEV(amount) AS std_spend_90d
    FROM RawTransactions
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.total_tx_90d,
    u.total_spend_90d,
    COALESCE(u.std_spend_90d, 0.0) AS std_spend_90d
FROM UserAggregates u;
```

---

## 4. Query Indexing Strategies for Feature Serving

When serving real-time features at inference time, query response latency must be under 10ms:

* **B-Tree Index:** Default for equality and range queries (`timestamp >= ?`, `user_id = ?`).
* **Composite Index:** An index on multiple columns `(user_id, timestamp DESC)`. Crucial for window functions and temporal lookups so the database does not perform an expensive disk sort.
* **Hash Index:** Fast $\mathcal{O}(1)$ equality lookups for exact string keys (e.g., `device_uuid = ?`).
