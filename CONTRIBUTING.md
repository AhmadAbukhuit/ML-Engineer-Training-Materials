# Contributing to the ML Engineer Training Materials

First off, thank you for considering contributing to this repository! This guide is meant to be a living document, and community contributions are what keep the Machine Learning Engineer Roadmap accurate, comprehensive, and useful for everyone learning AI, Data Engineering, and MLOps.

Whether you are fixing a typo in a mathematical derivation, adding a new script for a custom autograd engine, or sharing an updated deployment method for FastAPI, your help is appreciated.

## 📋 Table of Contents

- [Contributing to the ML Engineer Training Materials](#contributing-to-the-ml-engineer-training-materials)
  - [📋 Table of Contents](#-table-of-contents)
  - [🛠️ How Can I Contribute?](#️-how-can-i-contribute)
  - [🔄 Submission Workflow](#-submission-workflow)
  - [📝 Style Guide \& Formatting](#-style-guide--formatting)
  - [🔄 Keeping Your Fork Synced](#-keeping-your-fork-synced)
  - [🤝 Code of Conduct](#-code-of-conduct)

---

## 🛠️ How Can I Contribute?

There are several ways you can contribute to this project:

- **Add New Practice Modules:** Have a great exercise for teaching linear algebra, a new unstructured dataset to parse in Data Engineering, or a script for machine learning deployment? We want it.
- **Update Existing Content:** AI frameworks evolve rapidly. If a syntax is deprecated (e.g., in `pandas`, `PyTorch`, or `FastAPI`) or a setup guide is out of date, please submit an update.
- **Fix Formatting or Typos:** Clean, readable documentation, accurate math equations, and well-commented code are key for learners. Minor corrections are always welcome.
- **Suggest Topics:** If you don't have the time to write a module but want to request one, feel free to open an Issue.

---

## 🔄 Submission Workflow

To submit a contribution, please follow the standard GitHub Pull Request (PR) workflow:

1. **Fork the Repository:** Click the "Fork" button at the top right of the repository page.
2. **Clone Your Fork:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/ML-Engineer-Training-Materials.git
    cd ML-Engineer-Training-Materials
    ```

3. **Create a Branch:** Create a uniquely named branch for your feature or fix.

    ```bash
    git checkout -b add-yolo-deployment-module
    ```

4. **Make Your Changes:** Add or edit the Markdown, Jupyter Notebook (`.ipynb`), or Python (`.py`) files in the appropriate phase directory (e.g., `phase_3_deep_learning/src/`).

5. **Commit Your Changes:** Write a clear, concise commit message.

    ```bash
    git commit -m "docs: add practice module for YOLO object detection in phase 3"
    ```

6. **Push to Your Fork:**

    ```bash
    git push origin add-yolo-deployment-module
    ```

7. **Open a Pull Request:** Navigate to the original repository and click "Compare & pull request." Provide a brief description of what you added or fixed.

## 📝 Style Guide & Formatting

To keep the repository clean and easily scannable for students, please adhere to the following formatting guidelines when writing your files:

1. **File Naming**
    Use lowercase letters and underscores for Python scripts, and hyphens for markdown files if preferred.
    - Good: `telemetry_parser.py` or `01_autograd_xor.ipynb`
    - Bad: `Telemetry Parser.py` or `01 Autograd.ipynb`

2. **Headings and Structure (for Markdown)**
    Start every new document with a single H1 (`#`) title, followed by a brief description. Use H2 (`##`) and H3 (`###`) for subsequent sections. For math equations, use standard LaTeX syntax (e.g. `$$ \nabla L(\theta) $$`).

3. **Code Blocks**
    Always use syntax highlighting for code blocks in documentation. Specify the language (e.g., python, bash, json).

    Example:

    ```python
    import polars as pl

    # Drop missing timestamps and remove duplicates
    df = df.drop_nulls(subset=["timestamp"])
    df = df.unique(subset=["timestamp", "device_id"])
    ```

4. **Context is Key**
    When adding a new concept or code snippet, briefly explain what it does and why it is useful for the learner. Include docstrings in all Python functions explaining the mathematical or practical ML concept.
    - Good: `model.zero_grad()` — This resets all gradients to zero. It is strictly required before starting a new backward pass to prevent gradient accumulation from previous iterations.
    - Bad: Run `model.zero_grad()`.

5. **Categorization**
    Place your code or note in the most relevant phase folder (`phase_0`, `phase_1`, `phase_2`, `phase_3`, `phase_4`). Within the phase, separate concepts into `notebooks/` (for EDA and tutorials) and `src/` (for clean implementation code).

## 🔄 Keeping Your Fork Synced

Before creating a new branch or opening a pull request, please ensure your fork is up to date with the original repository to avoid merge conflicts.

1. **Add the upstream repository** (You only need to do this once):

    ```bash
    git remote add upstream https://github.com/AhmadAbukhuit/ML-Engineer-Training-Materials.git
    ```

2. **Fetch and merge the latest updates:**

    ```bash
    # Download the latest changes from the original repo
    git fetch upstream

    # Ensure you are on your local main branch
    git checkout main

    # Merge the updates into your local main branch
    git merge upstream/main

    # Push the synced changes up to your GitHub fork
    git push origin main
    ```

## 🤝 Code of Conduct

This project is an open and welcoming environment designed to help people learn Machine Learning Engineering. Please be respectful, encouraging, and constructive in your PR descriptions, issue comments, and code reviews.
