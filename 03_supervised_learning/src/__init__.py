"""Supervised Learning Module: Algorithms, Ensembles, Time Series, and Evaluation Metrics."""

from .decision_tree import DecisionTreeClassifierScratch
from .ensemble_benchmark import compare_ensembles
from .evaluation_metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mae_score,
    mape_score,
    matthews_corrcoef,
    mse_score,
    precision_score,
    r2_score,
    recall_score,
    rmse_score,
    roc_auc_score,
    roc_curve,
    specificity_score,
)
from .knn import KNNClassifier, KNNRegressor
from .linear_models import LinearRegressionGD, LogisticRegressionGD, RidgeRegression
from .naive_bayes import GaussianNaiveBayes
from .svm import LinearSVM
from .time_series import create_lag_features, walk_forward_cv_mae
from .train_eval import evaluate_model_cross_validation, train_and_export_baseline

__all__ = [
    "DecisionTreeClassifierScratch",
    "GaussianNaiveBayes",
    "KNNClassifier",
    "KNNRegressor",
    "LinearRegressionGD",
    "LinearSVM",
    "LogisticRegressionGD",
    "RidgeRegression",
    "accuracy_score",
    "average_precision_score",
    "balanced_accuracy_score",
    "brier_score_loss",
    "classification_report",
    "compare_ensembles",
    "confusion_matrix",
    "create_lag_features",
    "evaluate_model_cross_validation",
    "f1_score",
    "fbeta_score",
    "mae_score",
    "mape_score",
    "matthews_corrcoef",
    "mse_score",
    "precision_score",
    "r2_score",
    "recall_score",
    "rmse_score",
    "roc_auc_score",
    "roc_curve",
    "specificity_score",
    "train_and_export_baseline",
    "walk_forward_cv_mae",
]
