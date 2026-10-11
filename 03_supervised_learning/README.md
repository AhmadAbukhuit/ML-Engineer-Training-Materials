# Phase 3: Supervised Learning & Model Evaluation

Welcome to the **Supervised Learning** phase of the Machine Learning Engineering curriculum. This phase covers regression, classification, non-parametric methods, ensembles, time series modeling, and rigorous evaluation metrics with complete mathematical and production implementations.

---

## 📚 Curriculum & Reference Guides

- [`01_supervised_models_catalog.md`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/01_supervised_models_catalog.md): Comprehensive mathematical catalog of all supervised learning algorithms (Linear Models, Ridge, Lasso, ElasticNet, Logistic Regression, SVM, Naive Bayes, k-NN, CART Trees, Bagging, Random Forests, AdaBoost, XGBoost, LightGBM, CatBoost, Stacking, and Time Series Forecasting).
- [`02_evaluation_metrics_guide.md`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/02_evaluation_metrics_guide.md): Exhaustive guide to machine learning evaluation metrics (Confusion Matrix, Accuracy Paradox, Precision, Recall, Specificity, F1-Score, $F_\beta$, Balanced Accuracy, MCC, ROC-AUC vs PR-AUC, Log Loss, Brier Score, MAE, MSE, RMSE, MAPE, $R^2$, and retrieval metrics).

---

## 🛠️ Code Implementations (`src/`)

- [`src/evaluation_metrics.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/evaluation_metrics.py): Complete evaluation metrics implemented from scratch in pure NumPy (Accuracy, Precision, Recall, F1, $F_\beta$, Specificity, Balanced Accuracy, MCC, Brier Score, ROC-AUC, PR-AUC, MAE, MSE, RMSE, $R^2$, MAPE, and Classification Report).
- [`src/linear_models.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/linear_models.py): Vectorized OLS Regression, Ridge Regression ($L_2$), and Logistic Regression with Gradient Descent.
- [`src/svm.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/svm.py): Linear Support Vector Machine using soft-margin hinge loss optimization.
- [`src/naive_bayes.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/naive_bayes.py): Gaussian Naive Bayes classifier with log-probability numerical stability.
- [`src/knn.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/knn.py): K-Nearest Neighbors Classifier and Regressor from scratch with Euclidean, Manhattan, and Minkowski distance metrics.
- [`src/decision_tree.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/decision_tree.py): CART Decision Tree Classifier from scratch using Gini impurity.
- [`src/ensemble_benchmark.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/ensemble_benchmark.py): Comparative benchmark of Random Forest, LightGBM, and XGBoost with latency profiling.
- [`src/time_series.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/time_series.py): Lag feature generation and Walk-Forward cross-validation without temporal lookahead leakage.
- [`src/train_eval.py`](file:///home/ahmad/ML-Engineer-Training-Materials/03_supervised_learning/src/train_eval.py): Production pipeline for scikit-learn models with Stratified K-Fold metric reporting and model export.

---

## 🧪 Unit Tests (`tests/`)

Run the test suite using pytest:

```bash
pytest 03_supervised_learning/tests
```
