import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn import metrics as skm

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation_metrics import (
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
    specificity_score,
)


def test_confusion_matrix_and_accuracy():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    cm_scratch = confusion_matrix(y_true, y_pred)
    cm_sk = skm.confusion_matrix(y_true, y_pred)
    np.testing.assert_array_equal(cm_scratch, cm_sk)

    acc_scratch = accuracy_score(y_true, y_pred)
    acc_sk = skm.accuracy_score(y_true, y_pred)
    assert acc_scratch == pytest.approx(acc_sk)


def test_precision_recall_f1_binary():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0])

    p_scratch = precision_score(y_true, y_pred, average="binary")
    p_sk = skm.precision_score(y_true, y_pred, average="binary")
    assert p_scratch == pytest.approx(p_sk)

    r_scratch = recall_score(y_true, y_pred, average="binary")
    r_sk = skm.recall_score(y_true, y_pred, average="binary")
    assert r_scratch == pytest.approx(r_sk)

    f1_scratch = f1_score(y_true, y_pred, average="binary")
    f1_sk = skm.f1_score(y_true, y_pred, average="binary")
    assert f1_scratch == pytest.approx(f1_sk)

    f2_scratch = fbeta_score(y_true, y_pred, beta=2.0, average="binary")
    f2_sk = skm.fbeta_score(y_true, y_pred, beta=2.0, average="binary")
    assert f2_scratch == pytest.approx(f2_sk)


def test_multiclass_averaging():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2, 2, 1])
    y_pred = np.array([0, 2, 1, 0, 1, 2, 1, 2, 0, 1])

    for avg in ["macro", "micro", "weighted"]:
        f1_scratch = f1_score(y_true, y_pred, average=avg)
        f1_sk = skm.f1_score(y_true, y_pred, average=avg)
        assert f1_scratch == pytest.approx(f1_sk)

        p_scratch = precision_score(y_true, y_pred, average=avg)
        p_sk = skm.precision_score(y_true, y_pred, average=avg)
        assert p_scratch == pytest.approx(p_sk)

        r_scratch = recall_score(y_true, y_pred, average=avg)
        r_sk = skm.recall_score(y_true, y_pred, average=avg)
        assert r_scratch == pytest.approx(r_sk)


def test_specificity_balanced_accuracy_mcc_brier():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.85, 0.4, 0.95])

    spec = specificity_score(y_true, y_pred)
    assert spec == pytest.approx(3 / 4)

    bal_acc_scratch = balanced_accuracy_score(y_true, y_pred)
    bal_acc_sk = skm.balanced_accuracy_score(y_true, y_pred)
    assert bal_acc_scratch == pytest.approx(bal_acc_sk)

    mcc_scratch = matthews_corrcoef(y_true, y_pred)
    mcc_sk = skm.matthews_corrcoef(y_true, y_pred)
    assert mcc_scratch == pytest.approx(mcc_sk)

    brier_scratch = brier_score_loss(y_true, y_prob)
    brier_sk = skm.brier_score_loss(y_true, y_prob)
    assert brier_scratch == pytest.approx(brier_sk)


def test_roc_auc_and_pr_auc():
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=100, p=[0.7, 0.3])
    # Correlated probability scores
    y_score = y_true * 0.6 + rng.uniform(0, 0.4, size=100)

    roc_scratch = roc_auc_score(y_true, y_score)
    roc_sk = skm.roc_auc_score(y_true, y_score)
    assert roc_scratch == pytest.approx(roc_sk, rel=1e-3)

    pr_scratch = average_precision_score(y_true, y_score)
    pr_sk = skm.average_precision_score(y_true, y_score)
    assert pr_scratch == pytest.approx(pr_sk, rel=1e-2)


def test_regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    mae = mae_score(y_true, y_pred)
    mae_sk = skm.mean_absolute_error(y_true, y_pred)
    assert mae == pytest.approx(mae_sk)

    mse = mse_score(y_true, y_pred)
    mse_sk = skm.mean_squared_error(y_true, y_pred)
    assert mse == pytest.approx(mse_sk)

    rmse = rmse_score(y_true, y_pred)
    assert rmse == pytest.approx(np.sqrt(mse_sk))

    r2 = r2_score(y_true, y_pred)
    r2_sk = skm.r2_score(y_true, y_pred)
    assert r2 == pytest.approx(r2_sk)

    mape = mape_score(y_true, y_pred)
    mape_sk = skm.mean_absolute_percentage_error(y_true, y_pred) * 100.0
    assert mape == pytest.approx(mape_sk)


def test_classification_report_scratch():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    report = classification_report(y_true, y_pred)
    assert "0" in report
    assert "1" in report
    assert "macro avg" in report
    assert "weighted avg" in report
    assert report["accuracy"]["accuracy"] == 0.75
