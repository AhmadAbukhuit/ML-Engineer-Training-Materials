# 📊 Phase 2: Data Science, Big Data Engineering & Feature Platforms

Real-world machine learning is 70% data engineering. Before a model can be trained, raw data must be extracted, parsed, cleaned, standardized, and validated. A production-ready ML Engineer builds pipelines that handle missing data gracefully, prevent temporal leakage, enforce strict data contracts, and scale across multi-gigabyte columnar formats.

---

## 🎯 Learning Objectives

By the end of this phase, you will:

1. **High-Performance Ingestion:** Process multi-modal datasets (CSV, JSON, XML, Parquet) using `Polars` and `DuckDB`.
2. **Columnar Big Data & In-Process OLAP:** Master Apache Parquet, Apache Arrow memory formats, and predicate pushdown query optimization.
3. **Advanced SQL Feature Engineering:** Write production SQL queries utilizing window functions (`OVER PARTITION BY`), `LAG`/`LEAD`, and Common Table Expressions (CTEs).
4. **Data Quality Contracts:** Define runtime schema contracts using `Pandera` to block corrupted data batches and catch schema drift before model training.
5. **Point-in-Time Correctness (ASOF Joins):** Build temporal merge pipelines that join event observations with feature histories without lookahead data leakage.
6. **Advanced Feature Transformations:** Implement Out-of-Fold (OOF) smoothed Target Encoding for high-cardinality categories and Yeo-Johnson power transforms for skewed distributions.
7. **Computer Vision Pipelines:** Implement aspect-ratio preserving letterboxing and label format conversions (Pascal VOC to normalized YOLO TXT).

---

## 📚 Curated Study Resources

### 📖 Documentation & Guides
* **[DuckDB Documentation](https://duckdb.org/docs/)**: Vectorized SQL query engine for fast Parquet analytics.
* **[Polars User Guide](https://docs.pola.rs/user-guide/)**: Lazy evaluation and multithreaded Arrow execution in Rust.
* **[Pandera Documentation](https://pandera.readthedocs.io/)**: Declarative statistical data validation for DataFrames.
* **[Feast Feature Store Concepts](https://docs.feast.dev/)**: Point-in-time correctness, offline vs online feature serving.
* **[Scikit-Learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)**: Transformers, scalers, and imputers.

### 📄 Included Deep-Dive Guides
* [`01_columnar_storage_and_query_engines.md`](01_columnar_storage_and_query_engines.md): Parquet, Arrow, DuckDB, and Predicate Pushdown.
* [`02_sql_for_feature_engineering.md`](02_sql_for_feature_engineering.md): Window functions, rolling spend, login intervals, and CTEs.
* [`03_data_quality_and_feature_stores.md`](03_data_quality_and_feature_stores.md): Pandera contracts, temporal leakage, and Feature Store architecture.

---

## 🛠️ Practical Coding Tasks

Complete these implementation tasks inside `02_data_engineering/src/` and verify your solutions with tests in `02_data_engineering/tests/`.

---

### Task 2.1: Telemetry Parser & Aggregator (JSON/XML to CSV)
* **File:** `src/telemetry_parser.py`
* **Goal:** Parse hierarchical JSON and XML IoT telemetry logs, flatten them into a unified tabular schema, and export cleaned CSV records using Polars.

---

### Task 2.2: Automated CV Preprocessing & Label Conversion
* **File:** `src/image_pipeline.py`
* **Goal:** Resize raw images to fixed resolutions with aspect-ratio preserving letterbox padding, normalize pixels to `[0, 1]`, and convert Pascal VOC XML labels to normalized YOLO format (`class x_center y_center width height`).

---

### Task 2.3: The Zero-Leakage `scikit-learn` Pipeline
* **File:** `src/feature_engineering.py`
* **Goal:** Build an isolated `ColumnTransformer` with median/most-frequent imputers, `StandardScaler`, and `OneHotEncoder`. Fit strictly on train folds and export `preprocessor.joblib`.

---

### Task 2.4: Columnar Big Data & DuckDB Analytics
* **File:** `src/duckdb_parquet.py`
* **Goal:** Export DataFrames to compressed Apache Parquet and run SQL queries directly on disk with predicate and projection pushdown, plus rolling window statistics.

---

### Task 2.5: Data Quality Contracts via Pandera
* **File:** `src/data_validation.py`
* **Goal:** Implement declarative DataFrame validation schemas checking range limits, null constraints, and required fields. Halt pipeline execution on contract violations with actionable diagnostic reports.

---

### Task 2.6: Advanced Feature Transforms (OOF Target Encoding)
* **File:** `src/advanced_transforms.py`
* **Goal:** Implement Out-of-Fold smoothed Target Encoding with m-estimate smoothing to prevent target leakage on high-cardinality features, plus Yeo-Johnson power transformations.

---

### Task 2.7: Point-in-Time Correct (ASOF) Feature Joins
* **File:** `src/temporal_joins.py`
* **Goal:** Implement temporal backward ASOF merges that join prediction events with the latest feature states strictly on or before event timestamps, guaranteeing zero future leakage.

---

## 📁 Directory Structure for Phase 2

```text
02_data_engineering/
├── README.md
├── 01_columnar_storage_and_query_engines.md
├── 02_sql_for_feature_engineering.md
├── 03_data_quality_and_feature_stores.md
├── notebooks/
│   ├── 01_telemetry_eda.ipynb              # Exploring raw JSON/XML telemetry
│   ├── 02_image_visualization.ipynb         # Visualizing letterboxing & YOLO boxes
│   └── 03_sql_and_parquet_benchmarks.ipynb  # Parquet vs CSV benchmarks & ASOF joins
├── src/
│   ├── __init__.py
│   ├── telemetry_parser.py                 # Task 1.1: Polars IoT parser
│   ├── image_pipeline.py                   # Task 1.2: CV Letterboxing & VOC-to-YOLO
│   ├── feature_engineering.py              # Task 1.3: Zero-leakage ColumnTransformer
│   ├── duckdb_parquet.py                   # Task 1.4: Parquet export & DuckDB queries
│   ├── data_validation.py                  # Task 1.5: Pandera data contracts
│   ├── advanced_transforms.py              # Task 1.6: OOF Target Encoding & Yeo-Johnson
│   └── temporal_joins.py                   # Task 1.7: Point-in-time ASOF merges
└── tests/
    ├── __init__.py
    ├── test_data_engineering.py
    ├── test_duckdb_parquet.py
    ├── test_data_validation.py
    ├── test_advanced_transforms.py
    └── test_temporal_joins.py
```
