# Data Contracts, Schema Validation & Feature Stores

In production machine learning systems, silent data bugs (distribution drift, unexpected nulls, schema changes, and temporal lookahead leakage) are far more common than model bugs. This guide covers **Data Quality Contracts** and **Feature Store architectures**.

---

## 1. Data Contracts & Schema Validation

A **Data Contract** is a formal agreement between data producers (upstream software engineers, APIs, IoT sensors) and data consumers (ML engineers). It guarantees column names, types, allowed value ranges, and nullability constraints.

### Why Runtime DataFrame Validation is Essential
Static typing (like mypy) only checks variable types at compile time; it cannot check whether a DataFrame contains invalid values (e.g., negative ages, latitudes > 90, or a sudden spike in missing values).

Using **Pandera**, ML engineers write declarative runtime validation schemas:

```python
import pandera as pa
from pandera import Column, Check

TelemetrySchema = pa.DataFrameSchema(
    columns={
        "device_id": Column(str, nullable=False),
        "temperature": Column(float, Check.in_range(-50.0, 150.0), nullable=True),
        "lat": Column(float, Check.in_range(-90.0, 90.0), nullable=False),
        "lon": Column(float, Check.in_range(-180.0, 180.0), nullable=False),
        "timestamp": Column(pa.DateTime, nullable=False),
    },
    strict=True,  # Reject unrecognized unexpected columns
    coerce=True   # Automatically cast compatible types
)
```

If upstream data violates any check, Pandera halts the training pipeline and outputs a detailed diagnostic report before corrupted data reaches the model.

---

## 2. Temporal Data Leakage & Point-in-Time Correctness

**Temporal leakage** is one of the most subtle and destructive errors in machine learning. It occurs when a model is trained using features that incorporate information from *after* the prediction event occurred.

### The Naive Join Trap
Suppose you are predicting loan default for a customer who applied on **May 1st, 2024**.
If you simply perform:
```sql
SELECT * FROM loan_applications a
LEFT JOIN user_features f ON a.user_id = f.user_id;
```
If `user_features` contains the user's *current* credit score or balance as of **today**, the training set leaks whether the user repaid the loan in the future! The model achieves 99% accuracy in training, but catastrophically fails in production.

### Point-in-Time Correctness (ASOF Join)
To create an unbiased training set, every observation must be joined with the feature values **as they existed strictly prior to or at the observation timestamp**:

```text
Feature Updates:     (t=Jan 1: score=650)   (t=Apr 15: score=680)   (t=Jun 20: score=720)
                                                    │
Observation Event:                          (t=May 1: Loan Applied)
                                                    │
Point-in-Time Match:                                └──> Joined with score=680 (Apr 15)
                                                         (NEVER score=720 from Jun 20!)
```

In Polars and Pandas, this is computed using `join_asof()`.

---

## 3. Modern Feature Store Architecture (Feast)

A **Feature Store** is a centralized data platform that solves two critical problems:
1. **Feature Consistency:** Reusing the exact same feature logic across training and real-time inference.
2. **Dual-Store Architecture:**

```text
┌─────────────────────────────────────────────────────────────┐
│                       Feature Store                         │
├──────────────────────────────┬──────────────────────────────┤
│        Offline Store         │         Online Store         │
│  (Parquet / Snowflake / S3)  │      (Redis / DynamoDB)      │
├──────────────────────────────┼──────────────────────────────┤
│ • Optimized for high-volume  │ • Optimized for sub-10ms     │
│   batch scans                │   key-value lookups          │
│ • Point-in-time ASOF joins   │ • Stores only the latest     │
│ • Used for MODEL TRAINING    │   feature state per entity   │
│                              │ • Used for LIVE INFERENCE    │
└──────────────────────────────┴──────────────────────────────┘
```
