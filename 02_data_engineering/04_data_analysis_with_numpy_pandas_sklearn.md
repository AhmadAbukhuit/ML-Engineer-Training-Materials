# 🐼 Comprehensive Data Analysis with NumPy, Pandas & Scikit-Learn

In modern Machine Learning Engineering, modeling is only as effective as the exploratory data analysis and feature preprocessing that precedes it. Before passing data into complex estimators, an engineer must understand distribution shapes, correlation structures, missingness patterns, and outlier dynamics.

This guide provides an end-to-end reference for **NumPy**, **Pandas**, and **Scikit-Learn**—the holy trinity of data analysis in Python.

---

## 1. The Core Data Stack: Architectural Roles

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          Scikit-Learn                                  │
│   • Preprocessing (Scalers, Encoders, Imputers)                        │
│   • Pipelines & ColumnTransformers (Zero-Leakage Encapsulation)        │
│   • Feature Selection & Dimensionality Analysis                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Operates on / returns
┌───────────────────────────────────▼────────────────────────────────────┐
│                            Pandas                                      │
│   • Heterogeneous 2D Tabular Structures (DataFrame, Series)           │
│   • Exploratory Data Analysis (EDA), Missingness Profiling             │
│   • Grouping, Aggregating, Merging, Pivoting, Reshaping               │
│   • Time Series Resampling, Rolling Windows, Index Alignment           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Built on top of
┌───────────────────────────────────▼────────────────────────────────────┐
│                             NumPy                                      │
│   • N-Dimensional Homogeneous Arrays (`ndarray`)                       │
│   • Vectorized Array Arithmetic & Broadcasting Rules                   │
│   • High-Performance BLAS/LAPACK Mathematical Operations               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. NumPy for High-Performance Data Analysis

NumPy is the foundational layer for all numerical data in Python. It stores data in contiguous memory blocks (buffers) with homogeneous data types, eliminating Python interpreter overhead and pointer indirection.

### 2.1 Vectorization vs. Python Loops
In standard Python, a loop over a list involves dynamic type checking and boxing/unboxing overhead at every iteration:

```python
import numpy as np

# Python loop (Slow):
values = list(range(1_000_000))
squared_py = [x ** 2 for x in values]

# Vectorized NumPy (10x-50x Faster):
arr = np.arange(1_000_000)
squared_np = arr ** 2
```

### 2.2 The Cardinal Rules of Broadcasting
Broadcasting allows NumPy to perform element-wise arithmetic on arrays with different shapes without making unneeded copies in memory. Two dimensions are compatible if:
1. They are **equal**, or
2. One of them is **1**.

```python
# Matrix shape: (100, 5)
X = np.random.randn(100, 5)

# Mean vector shape: (5,) -> Broadcasts across all 100 rows
col_means = np.mean(X, axis=0)
X_centered = X - col_means  # (100, 5) - (5,) -> Valid!
```

### 2.3 Boolean Indexing & Conditional Masking
Boolean masks allow selective data querying and conditional replacements:

```python
arr = np.array([12, -4, 35, -1, 48, 0])

# Filter strictly positive values
positives = arr[arr > 0]  # [12, 35, 48]

# Conditional substitution with np.where
# Syntax: np.where(condition, value_if_true, value_if_false)
clipped = np.where(arr < 0, 0, arr)  # [12, 0, 35, 0, 48, 0]

# Multi-condition selection with np.select
conditions = [arr < 0, arr == 0, arr > 0]
choices = [-1, 0, 1]
signs = np.select(conditions, choices)
```

### 2.4 Statistical Profiling with NumPy
* **Percentiles & Quantiles:** `np.percentile(arr, [25, 50, 75])`
* **NaN-Resistant Functions:** When arrays contain missing values (`np.nan`), standard functions return `nan`. Use `np.nanmean()`, `np.nanstd()`, `np.nanmedian()`.
* **Pairwise Distance & Norms:**
  $$\|u - v\|_2 = \sqrt{\sum (u_i - v_i)^2} \longrightarrow \text{np.linalg.norm}(u - v)$$

---

## 3. Pandas for Exploratory Data Analysis & Wrangling

Pandas provides high-level data structures (`Series` and `DataFrame`) designed for heterogeneous, indexed tabular data.

### 3.1 Initial Dataset Inspection Checklist
When receiving a new raw dataset, always perform this initial audit:

```python
import pandas as pd

df = pd.read_csv("telemetry.csv")

# 1. Structural shape and schema
print(df.shape)
print(df.info())

# 2. Statistical distributions
print(df.describe(include="all"))

# 3. Missing values audit
missing_audit = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": (df.isna().sum() / len(df)) * 100
})
print(missing_audit[missing_audit["missing_count"] > 0])

# 4. Duplicate rows
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
```

### 3.2 Indexing: `loc` vs. `iloc`
* **`loc` (Label-based):** Selects by index labels and column names.
  `df.loc[df["age"] > 30, ["income", "churn"]]`
* **`iloc` (Integer position-based):** Selects by numeric position `[row_idx, col_idx]`.
  `df.iloc[0:10, 0:3]`

### 3.3 Data Cleaning & Imputation
* **Dropping Missing Data:**
  * `df.dropna(subset=["critical_column"])`: Drops rows missing a specific key feature.
  * `df.dropna(thresh=5)`: Retains rows that have at least 5 non-null values.
* **Filling Missing Data:**
  * `df["sensor_temp"].fillna(df["sensor_temp"].median(), inplace=False)`
  * `df["status"].ffill()`: Forward-fill for continuous sensor streams.
* **Type Optimization:**
  * Converting string columns with low cardinality to `category` saves 80%+ memory:
    `df["device_type"] = df["device_type"].astype("category")`

### 3.4 Aggregation & Grouping (`groupby` & `agg`)
The Split-Apply-Combine pattern:

```python
# Multi-feature custom aggregation
summary = df.groupby("device_type").agg(
    avg_temp=("temperature", "mean"),
    max_temp=("temperature", "max"),
    p95_vibration=("vibration", lambda x: np.percentile(x.dropna(), 95)),
    error_count=("status_code", lambda x: (x >= 400).sum())
).reset_index()

# Cross-tabulations for categorical associations
confusion_like = pd.crosstab(df["device_type"], df["failure_flag"], normalize="index")
```

### 3.5 Outlier Detection: The Interquartile Range (IQR) Method
An observation $x$ is flagged as an outlier if:
$$x < Q_1 - 1.5 \times \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \times \text{IQR}$$
Where $\text{IQR} = Q_3 - Q_1$.

```python
def get_iqr_bounds(series: pd.Series, factor: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return lower_bound, upper_bound
```

---

## 4. Scikit-Learn for Preprocessing & Transformation Pipelines

Scikit-learn standardizes feature manipulation using the **Estimator & Transformer API**.

### 4.1 The Three Core Scalers
| Scaler | Transformation Formula | Best Suited For | Outlier Sensitivity |
| :--- | :--- | :--- | :--- |
| **`StandardScaler`** | $z = \frac{x - \mu}{\sigma}$ | Normally distributed features, linear models, neural networks. | **High** (Mean & std distorted by outliers) |
| **`MinMaxScaler`** | $x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Bounded ranges (e.g. $[0, 1]$), image pixels, algorithms requiring positive bounds. | **Very High** (Extreme min/max compress distribution) |
| **`RobustScaler`** | $x_{\text{robust}} = \frac{x - Q_2}{Q_3 - Q_1}$ | Datasets with heavy tails and frequent outliers. Uses Median and IQR. | **Low** (Resilient to extreme values) |

### 4.2 Categorical Encoders
* **`OneHotEncoder`**: Converts discrete categories into binary dummy columns.
  * **Critical Production Flag:** `handle_unknown='ignore'`. When new, unseen categories appear in production requests, this prevents crashing by assigning zeros across all dummy columns.
* **`OrdinalEncoder`**: Maps categories to integers $\{0, 1, 2, \dots\}$. Use only when categories possess a natural ordinal hierarchy (e.g., `["low", "medium", "high"]`).

### 4.3 Composing Zero-Leakage Pipelines
Never call `scaler.fit()` on your entire dataset before splitting train and test sets. Doing so causes **Data Leakage**: the test set's mean and variance inform the training step, generating overly optimistic evaluation metrics that fail in production.

Always encapsulate steps inside `ColumnTransformer` and `Pipeline`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

numeric_cols = ["temperature", "pressure", "vibration"]
categorical_cols = ["region", "device_model"]

# 1. Numeric pipeline: Impute missing with median -> Scale robustly
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])

# 2. Categorical pipeline: Impute with most frequent -> One-hot encode
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# 3. Combine into ColumnTransformer
full_preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ],
    remainder="drop"  # Drop unhandled columns to prevent accidental leakage
)

# 4. Fit ONLY on training data; Transform both safely
X_train_clean = full_preprocessor.fit_transform(X_train)
X_test_clean = full_preprocessor.transform(X_test)
```

---

## 5. Summary Cheat Sheet for Engineers

1. **Vectorize:** Replace iterative loops with NumPy vectorized broadcasts and Pandas vectorized `.apply()` or built-ins.
2. **Audit Missingness Early:** Distinguish between Missing Completely at Random (MCAR) and Missing Informatively before deciding to drop or impute.
3. **Use Robust Statistics:** Default to `median` and `IQR` (or `RobustScaler`) when dealing with uncleaned real-world sensor or user data with outliers.
4. **Isolate Folds:** Fit scalers, encoders, and imputers **strictly** on the training fold, never on the combined dataset.
