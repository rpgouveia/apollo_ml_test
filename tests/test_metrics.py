import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from src.metrics import calculate_binary_roc


def test_roc_auc_continuous():
    """Ensures the accuracy of the ROC and AUC in a scenario with distinct continuous values."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    auc_sklearn = roc_auc_score(y_true, y_scores)

    assert auc_custom == pytest.approx(auc_sklearn)


def test_roc_auc_discrete():
    """Ensures that scores with ties converge with sklearn."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)

    # Rounding to multiples of 0.1 to force ties and constant blocks
    y_scores = np.round(np.random.rand(100), 1)

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    auc_sklearn = roc_auc_score(y_true, y_scores)

    assert auc_custom == pytest.approx(auc_sklearn)


def test_roc_auc_all_equal():
    """Ensures that if all scores are identical, the AUC returns the baseline of 0.5."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    auc_sklearn = roc_auc_score(y_true, y_scores)

    assert auc_custom == pytest.approx(0.5)
    assert auc_custom == pytest.approx(auc_sklearn)


def test_roc_auc_perfect_separation():
    """Ensures that a perfectly correct separation achieves an AUC of 1.0."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    auc_sklearn = roc_auc_score(y_true, y_scores)

    assert auc_custom == pytest.approx(1.0)
    assert auc_custom == pytest.approx(auc_sklearn)
