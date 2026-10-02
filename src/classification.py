import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from src.data_processing import extract_features
from src.metrics import (
    calculate_macro_f1_score,
    calculate_top_k_accuracy,
    calculate_binary_roc,
)


def print_fold_distribution(
    y_train: np.ndarray, y_test: np.ndarray, fold_idx: int
) -> None:
    """Prints the class distribution for a specific fold to verify stratification."""
    print(f"\n--- Class Distribution for Fold {fold_idx} ---")
    train_counts = pd.Series(y_train).value_counts(normalize=True) * 100
    test_counts = pd.Series(y_test).value_counts(normalize=True) * 100
    dist_df = (
        pd.DataFrame({"Train (%)": train_counts, "Test (%)": test_counts})
        .fillna(0)
        .round(2)
    )
    print(dist_df.sort_index().to_string())
    print("--------------------------------------\n")


def run_knn_pipeline(
    dataframe: DataFrame, max_k: int = 15, n_splits: int = 10
) -> DataFrame:
    """Runs 10-fold Stratified Group CV for KNN across multiple configs."""
    X = extract_features(dataframe)

    le = LabelEncoder()
    y = le.fit_transform(dataframe["syndrome_id"].values)
    groups = dataframe["subject_id"].values

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pre-compute folds to guarantee exact same splits across all configurations
    # and save redundant computation inside the loops.
    folds = list(cv.split(X, y, groups))

    scalers = {
        "Raw Data": None,
        "StandardScaler": StandardScaler(),
        "L2 Normalizer": Normalizer(norm="l2"),
    }
    metrics_list = ["cosine", "euclidean"]
    results = []

    print(f"\n=== Starting {n_splits}-Fold CV KNN Evaluation ===")

    for scaler_name, scaler_obj in scalers.items():
        for metric in metrics_list:
            print(f"\nEvaluating: [ {scaler_name} | {metric.upper()} ]")

            for k in range(1, max_k + 1):
                fold_f1, fold_top1, fold_top5, fold_auc = [], [], [], []

                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    if (
                        fold_idx == 0
                        and k == 1
                        and metric == metrics_list[0]
                        and scaler_name == list(scalers.keys())[0]
                    ):
                        print_fold_distribution(y_train, y_test, fold_idx + 1)

                    if scaler_obj is not None:
                        X_train = scaler_obj.fit_transform(X_train)
                        X_test = scaler_obj.transform(X_test)

                    # weights='distance' solves arbitrary index-based tie-breaking for small k (e.g., k=2)
                    knn = KNeighborsClassifier(
                        n_neighbors=k, metric=metric, weights="distance"
                    )
                    knn.fit(X_train, y_train)

                    y_probs = knn.predict_proba(X_test)

                    # Single source of truth for predictions.
                    # Stable sort descending (-y_probs with mergesort) safely mimics argmax tie-breaking behavior.
                    sorted_indices = np.argsort(-y_probs, axis=1, kind="mergesort")
                    y_pred_ranked = knn.classes_[sorted_indices]

                    # Force F1 to use strictly the top ranked class from our custom matrix
                    y_pred = y_pred_ranked[:, 0]

                    f1 = calculate_macro_f1_score(y_test, y_pred)
                    top1 = calculate_top_k_accuracy(y_test, y_pred_ranked, k=1)
                    top5 = calculate_top_k_accuracy(y_test, y_pred_ranked, k=5)

                    auc_classes = []
                    for idx, cls in enumerate(knn.classes_):
                        y_true_bin = (y_test == cls).astype(int)
                        if np.sum(y_true_bin) > 0:
                            auc, _, _ = calculate_binary_roc(
                                y_true_bin, y_probs[:, idx]
                            )
                            auc_classes.append(auc)

                    macro_auc = np.mean(auc_classes) if auc_classes else 0.0

                    fold_f1.append(f1)
                    fold_top1.append(top1)
                    fold_top5.append(top5)
                    fold_auc.append(macro_auc)

                results.append(
                    {
                        "Scaler": scaler_name,
                        "Distance": metric,
                        "k": k,
                        "Top-1 Acc (Mean)": np.mean(fold_top1),
                        "Top-1 Acc (Std)": np.std(fold_top1),
                        "Top-5 Acc (Mean)": np.mean(fold_top5),
                        "Top-5 Acc (Std)": np.std(fold_top5),
                        "Macro F1 (Mean)": np.mean(fold_f1),
                        "Macro F1 (Std)": np.std(fold_f1),
                        "Macro AUC (Mean)": np.mean(fold_auc),
                        "Macro AUC (Std)": np.std(fold_auc),
                    }
                )

                print(
                    f"k={k:02d} | F1: {np.mean(fold_f1):.4f}±{np.std(fold_f1):.4f} | AUC: {np.mean(fold_auc):.4f}±{np.std(fold_auc):.4f}"
                )

    return pd.DataFrame(results)


def save_and_display_best_results(
    results_df: DataFrame, output_dir: str = "outputs"
) -> None:
    """Finds the optimal k for each configuration based on Macro F1, displays summary, and saves to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "knn_evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n=== Optimal Model Configurations ===")
    print(
        "* Optimization Criterion: Macro F1 Score (to account for class imbalance) *\n"
    )

    combinations = results_df[["Scaler", "Distance"]].drop_duplicates()

    for _, row in combinations.iterrows():
        scaler = row["Scaler"]
        metric = row["Distance"]
        subset = results_df[
            (results_df["Scaler"] == scaler) & (results_df["Distance"] == metric)
        ]

        # Select best k using Macro F1
        best_row = subset.loc[subset["Macro F1 (Mean)"].idxmax()]

        print(f"[{scaler}] + [{metric.upper()}]:")
        print(f" - Optimal k : {best_row['k']}")
        print(
            f" - Macro F1  : {best_row['Macro F1 (Mean)']:.4f} ± {best_row['Macro F1 (Std)']:.4f}"
        )
        print(
            f" - Top-1 Acc : {best_row['Top-1 Acc (Mean)']:.4f} ± {best_row['Top-1 Acc (Std)']:.4f}"
        )
        print(
            f" - Top-5 Acc : {best_row['Top-5 Acc (Mean)']:.4f} ± {best_row['Top-5 Acc (Std)']:.4f}"
        )
        print(
            f" - Macro AUC : {best_row['Macro AUC (Mean)']:.4f} ± {best_row['Macro AUC (Std)']:.4f}\n"
        )

    print(f"[+] Full evaluation results saved to: {csv_path}\n")
