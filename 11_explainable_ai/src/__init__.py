"""Explainable AI (XAI) Module: SHAP, LIME, Permutation Importance, and PDP/ICE."""

from .lime_explainer import LimeTabularExplainer
from .partial_dependence import individual_conditional_expectation, partial_dependence
from .permutation_importance import permutation_feature_importance
from .shapley_values import ExactShapleyExplainer, SamplingShapleyExplainer

__all__ = [
    "ExactShapleyExplainer",
    "LimeTabularExplainer",
    "SamplingShapleyExplainer",
    "individual_conditional_expectation",
    "partial_dependence",
    "permutation_feature_importance",
]
