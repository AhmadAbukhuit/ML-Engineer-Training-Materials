"""Algorithmic Fairness and Bias Auditing Metrics."""

from dataclasses import dataclass
import numpy as np


@dataclass
class FairnessReport:
    demographic_parity_diff: float
    disparate_impact_ratio: float
    equal_opportunity_diff: float
    equalized_odds_diff: float
    group_0_selection_rate: float
    group_1_selection_rate: float
    group_0_tpr: float
    group_1_tpr: float
    group_0_fpr: float
    group_1_fpr: float


class FairnessAuditor:
    """Evaluates Algorithmic Fairness Across Sensitive Demographic Attributes.

    References:
    - Hardt et al., "Equality of Opportunity in Supervised Learning", NeurIPS 2016.
    - Feldman et al., "Certifying and Removing Disparate Impact", KDD 2015.
    """

    @staticmethod
    def audit(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> FairnessReport:
        """Audit classification predictions for group disparity.

        Parameters
        ----------
        y_true : np.ndarray of shape (N,)
            Binary ground truth labels in {0, 1}.
        y_pred : np.ndarray of shape (N,)
            Binary predicted labels in {0, 1}.
        sensitive_attr : np.ndarray of shape (N,)
            Binary sensitive / protected group indicator A in {0, 1}.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)
        a = np.asarray(sensitive_attr, dtype=int)

        mask_0 = a == 0
        mask_1 = a == 1

        if np.sum(mask_0) == 0 or np.sum(mask_1) == 0:
            raise ValueError("Both sensitive groups A=0 and A=1 must have at least one sample.")

        # Selection rates: P(Y_hat = 1 | A = a)
        sel_0 = float(np.mean(y_p[mask_0] == 1))
        sel_1 = float(np.mean(y_p[mask_1] == 1))
        dp_diff = abs(sel_0 - sel_1)

        # Disparate impact ratio: min(sel_0, sel_1) / max(sel_0, sel_1)
        max_sel = max(sel_0, sel_1)
        if max_sel == 0.0:
            di_ratio = 1.0
        else:
            di_ratio = min(sel_0, sel_1) / max_sel

        # True Positive Rates: P(Y_hat = 1 | A = a, Y = 1)
        pos_0 = mask_0 & (y_t == 1)
        pos_1 = mask_1 & (y_t == 1)
        tpr_0 = float(np.mean(y_p[pos_0] == 1)) if np.sum(pos_0) > 0 else 0.0
        tpr_1 = float(np.mean(y_p[pos_1] == 1)) if np.sum(pos_1) > 0 else 0.0
        eq_opp_diff = abs(tpr_0 - tpr_1)

        # False Positive Rates: P(Y_hat = 1 | A = a, Y = 0)
        neg_0 = mask_0 & (y_t == 0)
        neg_1 = mask_1 & (y_t == 0)
        fpr_0 = float(np.mean(y_p[neg_0] == 1)) if np.sum(neg_0) > 0 else 0.0
        fpr_1 = float(np.mean(y_p[neg_1] == 1)) if np.sum(neg_1) > 0 else 0.0

        # Equalized odds difference: max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)
        eq_odds_diff = max(abs(tpr_0 - tpr_1), abs(fpr_0 - fpr_1))

        return FairnessReport(
            demographic_parity_diff=dp_diff,
            disparate_impact_ratio=di_ratio,
            equal_opportunity_diff=eq_opp_diff,
            equalized_odds_diff=eq_odds_diff,
            group_0_selection_rate=sel_0,
            group_1_selection_rate=sel_1,
            group_0_tpr=tpr_0,
            group_1_tpr=tpr_1,
            group_0_fpr=fpr_0,
            group_1_fpr=fpr_1,
        )
