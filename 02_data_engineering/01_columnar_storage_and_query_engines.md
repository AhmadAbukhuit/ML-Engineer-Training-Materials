# Columnar Storage & High-Performance Query Engines

In production machine learning, datasets frequently reach tens or hundreds of gigabytes. Storing and querying this data in row-oriented plain text formats like CSV or JSON causes severe I/O bottlenecks and excessive memory consumption. This guide covers **columnar storage**, **Apache Arrow**, and **modern analytical query engines (DuckDB and Polars)**.

---

## 1. Row-Oriented vs. Column-Oriented Storage

| Property | Row-Oriented (CSV, JSON, SQLite) | Column-Oriented (Parquet, ORC) |
| :--- | :--- | :--- |
| **Physical Layout** | Consecutive values of a single row are stored together: `[id1, age1, city1], [id2, age2, city2]` | Consecutive values of a single column are stored together: `[id1, id2], [age1, age2], [city1, city2]` |
| **Optimal Use Case** | OLTP (Online Transaction Processing): Quick row-level writes, updates, single-record lookups. | OLAP (Online Analytical Processing) & ML: Aggregations, scanning subsets of columns across millions of rows. |
| **Compression Ratio** | Poor (mixing strings, floats, ints in sequence degrades compression). | Extremely High (similar data types in sequence compress exceptionally well using Run-Length Encoding and Dictionary Encoding). |
| **I/O Efficiency** | Must read every column from disk even if your model only needs 3 features. | Reads **only the requested feature columns** directly from disk (Projection Pushdown). |

---

## 2. Apache Parquet Internals

Apache Parquet is the industry standard for ML training datasets. Its hierarchical layout consists of:

1. **File Metadata (Footer):** Contains schema definitions, total row count, and pointers to Row Groups.
2. **Row Groups:** Horizontal partitions of the dataset (typically 64MB to 512MB each).
3. **Column Chunks:** The data for a specific column within a Row Group.
4. **Data Pages:** The smallest unit within a Column Chunk, compressed (e.g., using Snappy or ZSTD) and encoded (Dictionary, Bit-Packing).

### The Power of Predicate Pushdown
Parquet stores **summary statistics** (minimum value, maximum value, null count) for every column chunk in its metadata footer.

When a query requests:
```sql
SELECT * FROM telemetry WHERE temperature > 80.0;
```

The storage engine reads the Parquet footer *first*. If a Row Group's metadata indicates that `max(temperature) = 65.0`, the engine skips that entire Row Group on disk without decompressing or loading a single byte into RAM.

---

## 3. Apache Arrow: The In-Memory Standard

While Parquet is an on-disk format, **Apache Arrow** is the standardized, open-source **in-memory columnar memory format**.

* **Traditional Problem:** Passing data between Python (pandas), C++, and Spark historically required slow serialization and deserialization (copying and converting formats in memory).
* **The Arrow Solution:** A shared, zero-copy in-memory format. Systems that speak Arrow can share memory buffers directly across processes without serialization overhead.
* Modern libraries like **Polars**, **DuckDB**, and **PyArrow** use Apache Arrow internally for SIMD (Single Instruction, Multiple Data) vectorized execution.

---

## 4. In-Process OLAP: DuckDB vs. Distributed Clusters (PySpark)

Historically, querying large Parquet datasets required provisioning a distributed cluster (such as Apache Spark). Today, modern analytical engines bring cluster-level performance to a single machine:

* **DuckDB:** Often called "the SQLite for Analytics". It is an in-process vectorized SQL OLAP engine that runs inside your Python process. It can query multi-gigabyte compressed Parquet files directly from local disk or S3 without loading them entirely into RAM.
* **Polars:** A blazingly fast DataFrame library written in Rust, leveraging Apache Arrow and multithreaded query planning with lazy evaluation.
* **When to use PySpark:** Only when your data genuinely cannot fit on a single modern machine's NVMe drive or compute capability (typically > 500GB–1TB of raw data).
