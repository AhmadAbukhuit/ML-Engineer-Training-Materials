"""Supervised Learning Module: Algorithms, Ensembles, Time Series, and Evaluation."""

from .decision_tree import DecisionTreeClassifierScratch
from .ensemble_benchmark import compare_ensembles
from .knn import KNNClassifier, KNNRegressor
from .linear_models import LinearRegressionGD, LogisticRegressionGD, RidgeRegression
from .naive_bayes import GaussianNaiveBayes
from .svm import LinearSVM
from .time_series import create_lag_features, walk_forward_cv_mae
from .train_eval import evaluate_classifier, train_baseline

__all__ = [
    "DecisionTreeClassifierScratch",
    "GaussianNaiveBayes",
    "KNNClassifier",
    "KNNRegressor",
    "LinearRegressionGD",
    "LinearSVM",
    "LogisticRegressionGD",
    "RidgeRegression",
    "compare_ensembles",
    "create_lag_features",
    "evaluate_classifier",
    "train_baseline",
    "walk_forward_cv_mae",
]
