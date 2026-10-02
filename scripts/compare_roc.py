import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.metrics import calculate_binary_roc


def calculate_binary_roc_buggy(y_true_binary, y_scores):
    """Buggy version without tie handling, kept exclusively to generate report evidence."""
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true_binary[desc_score_indices]

    tpr_list, fpr_list = [0.0], [0.0]
    num_pos = np.sum(y_true_sorted == 1)
    num_neg = np.sum(y_true_sorted == 0)

    tp, fp = 0, 0
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / num_pos if num_pos > 0 else 0.0)
        fpr_list.append(fp / num_neg if num_neg > 0 else 0.0)

    return float(np.trapezoid(tpr_list, fpr_list))

def generate_comparison_table() -> None:
    """Generates a Before/After comparison table for ROC AUC to be used in the final report."""
    rng = np.random.default_rng(42)
    
    # 1. Continuous scores with added signal (rand + 0.3 * y)
    y_true = rng.integers(0, 2, 100)
    y_scores_cont = rng.random(100) + 0.3 * y_true
    
    # 2. Discrete scores (KNN-like ties)
    y_scores_disc = np.round(y_scores_cont, 1)
    
    # 3. Same data, grouped by class
    idx_grouped = np.argsort(y_true)
    y_true_grouped = y_true[idx_grouped]
    y_scores_grouped = y_scores_disc[idx_grouped]
    
    # 4. Grouped in reverse order
    idx_rev = np.argsort(y_true)[::-1]
    y_true_rev = y_true[idx_rev]
    y_scores_rev = y_scores_disc[idx_rev]
    
    # 5. All scores equal
    y_true_eq = np.array([1, 1, 0, 0, 1, 0])
    y_scores_eq = np.array([0.5] * 6)

    scenarios = [
        ("Continuous scores", y_true, y_scores_cont),
        ("Discrete scores (KNN-like)", y_true, y_scores_disc),
        ("Same data, grouped by class", y_true_grouped, y_scores_grouped),
        ("Grouped in reverse order", y_true_rev, y_scores_rev),
        ("All scores equal", y_true_eq, y_scores_eq)
    ]

    results = []
    for name, yt, ys in scenarios:
        buggy_auc = calculate_binary_roc_buggy(yt, ys)
        custom_auc, _, _ = calculate_binary_roc(yt, ys)
        sklearn_auc = roc_auc_score(yt, ys)
        
        results.append({
            "Scenario": name,
            "Buggy (Before)": f"{buggy_auc:.4f}",
            "Fixed (After)": f"{custom_auc:.4f}",
            "Sklearn": f"{sklearn_auc:.4f}",
            "Fixed Match?": "Yes" if abs(custom_auc - sklearn_auc) < 1e-7 else "No"
        })

    df = pd.DataFrame(results)
    
    print("\n=== ROC AUC Bug Fix Evidence ===")
    print(df.to_string(index=False))
    print("================================\n")
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "roc_bugfix_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Evidence table exported to: {csv_path}\n")

if __name__ == "__main__":
    generate_comparison_table()