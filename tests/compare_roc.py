import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.metrics import calculate_binary_roc


def generate_comparison_table() -> None:
    """Generates a comparison table between the custom implementation and sklearn, saving it to a CSV."""
    np.random.seed(42)

    # 1. Continuous scores
    y_true = np.random.randint(0, 2, 100)
    y_scores_cont = np.random.rand(100)

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
        ("All scores equal", y_true_eq, y_scores_eq),
    ]

    results = []
    for name, yt, ys in scenarios:
        custom_auc, _, _ = calculate_binary_roc(yt, ys)
        sklearn_auc = roc_auc_score(yt, ys)

        results.append(
            {
                "Scenario": name,
                "Custom Function": f"{custom_auc:.4f}",
                "Sklearn": f"{sklearn_auc:.4f}",
                "Status": "Match" if abs(custom_auc - sklearn_auc) < 1e-7 else "Diff",
            }
        )

    df = pd.DataFrame(results)

    # Terminal output without tabulate
    print("\n=== ROC AUC Implementation Comparison ===")
    print(df.to_string(index=False))

    # Export to outputs folder
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "roc_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Comparison table exported to: {csv_path}\n")


if __name__ == "__main__":
    generate_comparison_table()
