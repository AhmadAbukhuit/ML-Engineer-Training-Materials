# 📊 Phase 1: Data Science & Big Data Engineering

Real-world machine learning is 70% data engineering. Before a model can be trained, raw data must be extracted, parsed, cleaned, and standardized. A production-ready ML Engineer builds pipelines that handle missing data gracefully and transform multi-modal inputs (CSV, JSON, XML, Images) into clean, model-ready tensors.

---

## 🎯 Learning Objectives

By the end of this phase, you will:

1. Process massive structured datasets (CSVs) using high-performance libraries like `Polars` and `pandas`.
2. Flatten deeply nested semi-structured data (JSON/XML telemetry logs) into relational tables.
3. Process unstructured image data using `OpenCV`, applying automated resizing, normalization, and bounding-box label conversion (XML to YOLO TXT).
4. Eliminate data leakage by wrapping imputation, scaling, and encoding steps into reusable `scikit-learn` pipelines.

---

## 📚 Curated Study Resources

### 📖 Documentation & Guides

* **[Polars User Guide](https://docs.pola.rs/user-guide/)**: Essential for understanding lazy evaluation and multi-threaded DataFrame operations.
* **[Pandas to Polars Cheat Sheet](https://docs.pola.rs/user-guide/migration/pandas/)**: Bridging the gap between standard pandas and high-performance rust-based execution.
* **[Scikit-Learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)**: The official guide to transformers, scalers, and imputers.
* **[OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)**: Core image operations, color space conversions, and geometric transformations.

### 🎥 Video Lectures

* **Rob Mulla - Pandas Data Science Tutorial:** Practical techniques for Exploratory Data Analysis (EDA).
* **Sentdex - OpenCV with Python:** Practical walk-throughs on loading, analyzing, and transforming image arrays.

---

## 🛠️ Practical Coding Tasks

Complete these three implementation tasks inside `phase_1_data_engineering/src/` and use the `notebooks/` directory for exploratory data analysis (EDA) before finalizing your scripts.

---

### Task 1.1: Telemetry Parser & Aggregator (JSON/XML to CSV)

* **File:** `src/telemetry_parser.py`
* **Context:** You have a directory of raw sensor logs. Half are in nested JSON formats, and half are in XML. You need a unified tabular format for model training.
* **Requirements:**
  1. Write a script that iterates through a `data/raw/telemetry/` directory.
  2. Parse nested JSON logs (e.g., extracting `sensor_1.temperature` and `gps.coordinates.lat`).
  3. Parse XML logs (e.g., `<reading><sensor id="2"><value>45.2</value></sensor></reading>`) using Python's `xml.etree.ElementTree`.
  4. Merge all parsed records into a single high-performance `Polars` DataFrame.
  5. Handle missing timestamps and drop duplicate entries.
  6. Export the final cleaned table to `data/processed/unified_telemetry.csv`.
* **Deliverable:** A robust, type-hinted Python script that executes this pipeline in under 5 seconds for 100,000+ records.

---

### Task 1.2: Automated CV Preprocessing & Label Conversion

* **File:** `src/image_pipeline.py`
* **Context:** You are preparing a dataset for an object detection model (like YOLOv8). The raw dataset contains ultra-high-resolution RGB and thermal imagery, but the annotations are in Pascal VOC (XML) format.
* **Requirements:**
  1. Use `OpenCV` (`cv2`) to load raw images from `data/raw/images/`.
  2. Resize all images to a standard model input resolution (e.g., 640x640). Maintain aspect ratios using letterboxing (padding with black pixels) to prevent distortion.
  3. Normalize the pixel values to a `[0, 1]` range.
  4. Write a parser that reads Pascal VOC XML files and converts the bounding box coordinates into the normalized YOLO TXT format: `<class_id> <center_x> <center_y> <width> <height>`.
  5. Save the resized images to `data/processed/images/` and the converted labels to `data/processed/labels/`.
* **Deliverable:** A pipeline script that guarantees every output image has a strictly matching label file with correctly scaled coordinates.

---

### Task 1.3: The Zero-Leakage `scikit-learn` Pipeline

* **File:** `src/feature_engineering.py`
* **Context:** You are preparing tabular features for a regression model. If you scale or impute data *before* splitting your train and test sets, you cause data leakage. 
* **Requirements:**
  1. Load a tabular dataset using `pandas`.
  2. Split the data into strictly isolated training and testing sets using `train_test_split`.
  3. Build a `sklearn.compose.ColumnTransformer` with two distinct branches:
     * **Numeric branch:** Apply `SimpleImputer` (strategy='median') followed by `StandardScaler`.
     * **Categorical branch:** Apply `SimpleImputer` (strategy='most_frequent') followed by `OneHotEncoder` (drop='first').
  4. Fit the pipeline *only* on the training data, then transform both the training and testing sets.
  5. Save the fitted pipeline object to disk using `joblib` so it can be reloaded during the production API phase.
* **Deliverable:** The Python script and the exported `preprocessor.joblib` artifact.

---

## 📁 Directory Structure for Phase 1

```text
phase_1_data_engineering/
├── README.md
├── notebooks/
│   ├── 01_telemetry_eda.ipynb      # Exploring the raw JSON/XML data
│   └── 02_image_visualization.ipynb # Plotting OpenCV bounding boxes 
└── src/
    ├── __init__.py
    ├── telemetry_parser.py         # Task 1.1 implementation
    ├── image_pipeline.py           # Task 1.2 implementation
    └── feature_engineering.py      # Task 1.3 implementation
```
