import numpy as np
from numpy.typing import NDArray


def calculate_top_k_accuracy(
        y_true: NDArray, 
        y_pred_ranked: NDArray, 
        k: int = 1
) -> float:
    """
    Calculates Top-K accuracy.
    Checks if the true class is among the top k predictions.
    https://ml-compiled.readthedocs.io/en/latest/metrics.html
    https://k-dm.work/en/eval/classification/topk-accuracy/topk-accuracy/

    Args:
        y_true: 1D array containing the true labels (n_samples,).
        y_pred_ranked: 2D array containing the predicted classes 
        ordered by distance/score (n_samples, n_classes).
        k: The number of top predictions to consider for accuracy
        calculation.
    """
    top_k_predictions: NDArray = y_pred_ranked[:, :k]
    matches: NDArray = np.any(top_k_predictions == y_true[:, None], axis=1)
    return float(np.mean(matches))

def calculate_macro_f1_score(y_true: NDArray, y_pred: NDArray) -> float:
    """
    Calculates the macro F1 score.
    The macro F1 score is the unweighted mean of F1 scores for each class.
    It treats all classes equally, regardless of their support (number of true instances).
    https://ml-compiled.readthedocs.io/en/latest/metrics.html

    Args:
        y_true: 1D array containing the true labels.
        y_pred: 1D array containing the predicted labels.
    """
    classes: NDArray = np.unique(y_true)
    f1_scores: list[float] = []

    for label in classes:
        tp: int = np.sum((y_pred == label) & (y_true == label))
        fp: int = np.sum((y_pred == label) & (y_true != label))
        fn: int = np.sum((y_pred != label) & (y_true == label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))

def calculate_binary_roc(
        y_true_binary: NDArray, 
        y_scores: NDArray
) -> tuple[float, list, list]:
    """
    Calculates the ROC curve for a binary classification problem.
    Returns the AUC (Area Under the Curve), FPR (False Positive Rate), and TPR (True Positive Rate).
    https://ml-compiled.readthedocs.io/en/latest/metrics.html

    Args:
        y_true_binary: 1D array containing the true binary labels.
        y_scores: 1D array containing the predicted scores.

    Returns:
        A tuple containing the AUC, FPR, and TPR.
    """
    desc_score_indices: NDArray = np.argsort(y_scores)[::-1]
    y_true_sorted: NDArray = y_true_binary[desc_score_indices]

    tpr_list: list[float] = [0.0]
    fpr_list: list[float] = [0.0]
    num_pos: int = np.sum(y_true_sorted == 1)
    num_neg: int = np.sum(y_true_sorted == 0)

    tp: int = 0
    fp: int = 0
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / num_pos if num_pos > 0 else 0.0)
        fpr_list.append(fp / num_neg if num_neg > 0 else 0.0)

    # Trapezoidal integration to calculate the AUC
    #https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html
    auc: float = np.trapezoid(tpr_list, fpr_list)
    return auc, fpr_list, tpr_list

