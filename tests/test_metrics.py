import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score

from src.metrics import (
    calculate_binary_roc,
    calculate_macro_f1_score,
    calculate_per_class_f1_score,
    calculate_top_k_accuracy,
)


def test_roc_auc_continuous():
    """Ensures the accuracy of the ROC and AUC with distinct continuous values (with signal)."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_scores = rng.random(100) + 0.3 * y_true

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    auc_sklearn = roc_auc_score(y_true, y_scores)
    assert auc_custom == pytest.approx(auc_sklearn)


def test_roc_auc_discrete():
    """Ensures that scores with ties converge with sklearn."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_scores = np.round(rng.random(100) + 0.3 * y_true, 1)

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    auc_sklearn = roc_auc_score(y_true, y_scores)
    assert auc_custom == pytest.approx(auc_sklearn)


def test_roc_auc_order_invariance():
    """Ensures that identical blocks of tied scores yield the same AUC regardless of sample order."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_scores = np.round(rng.random(100) + 0.3 * y_true, 1)

    auc_original, _, _ = calculate_binary_roc(y_true, y_scores)

    # Shuffle the dataset
    indices = rng.permutation(len(y_true))
    auc_shuffled, _, _ = calculate_binary_roc(y_true[indices], y_scores[indices])
    assert auc_original == pytest.approx(auc_shuffled)


def test_roc_auc_all_equal():
    """Ensures that if all scores are identical, the AUC returns the baseline of 0.5."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    assert auc_custom == pytest.approx(0.5)


def test_roc_auc_perfect_separation():
    """Ensures that a perfectly correct separation achieves an AUC of 1.0."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    auc_custom, _, _ = calculate_binary_roc(y_true, y_scores)
    assert auc_custom == pytest.approx(1.0)


def test_f1_macro():
    """Ensures custom macro F1 matches sklearn implementation."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 10, 100)
    y_pred = rng.integers(0, 10, 100)

    f1_custom = calculate_macro_f1_score(y_true, y_pred)
    f1_sklearn = f1_score(y_true, y_pred, average="macro")
    assert f1_custom == pytest.approx(f1_sklearn)


def test_f1_macro_missing_class():
    """Ensures union1d correctly handles classes present in y_pred but missing in y_true."""
    y_true = np.array([0, 0, 1, 1])
    # Class 2 is predicted but does not exist in y_true
    y_pred = np.array([0, 2, 1, 1]) 
    
    f1_custom = calculate_macro_f1_score(y_true, y_pred)
    f1_sklearn = f1_score(y_true, y_pred, average='macro')
    assert f1_custom == pytest.approx(f1_sklearn)


def test_per_class_f1_score():
    """Ensures custom per-class F1 matches sklearn implementation (average=None)."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 10, 100)
    y_pred = rng.integers(0, 10, 100)
    
    classes = np.union1d(y_true, y_pred)
    
    f1_custom_dict = calculate_per_class_f1_score(y_true, y_pred)
    f1_sklearn = f1_score(y_true, y_pred, average=None, labels=classes)
    
    for idx, label in enumerate(classes):
        assert f1_custom_dict[int(label)] == pytest.approx(f1_sklearn[idx])


@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_k_accuracy(k):
    """Ensures custom Top-K accuracy matches sklearn (using continuous scores to avoid tie-breaks)."""
    rng = np.random.default_rng(42)
    n_classes = 10
    y_true = rng.integers(0, n_classes, 100)

    # Continuous random scores to ensure deterministic ranking for both functions
    y_scores = rng.random((100, n_classes))

    # Rank the predictions for the custom implementation descendingly (highest score first)
    y_pred_ranked = np.argsort(y_scores, axis=1)[:, ::-1]

    top_k_custom = calculate_top_k_accuracy(y_true, y_pred_ranked, k=k)
    
    # sklearn requires the full continuous score matrix and all possible labels
    top_k_sklearn = top_k_accuracy_score(y_true, y_scores, k=k, labels=np.arange(n_classes))
    
    assert top_k_custom == pytest.approx(top_k_sklearn)
