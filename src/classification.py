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
    """Runs 10-fold Stratified Group CV for KNN across multiple k values, distances, and scalers."""
    X = extract_features(dataframe)

    le = LabelEncoder()
    y = le.fit_transform(dataframe["syndrome_id"].values)
    groups = dataframe["subject_id"].values

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scalers = {
        "StandardScaler": StandardScaler(),
        "L2 Normalizer": Normalizer(norm="l2"),
    }
    metrics_list = ["cosine", "euclidean"]
    results = []

    print(f"\n=== Starting {n_splits}-Fold CV KNN Evaluation ===")
    print(
        f"Testing k from 1 to {max_k}, Distances: {metrics_list}, Scalers: {list(scalers.keys())}"
    )

    for scaler_name, scaler_obj in scalers.items():
        for metric in metrics_list:
            print(f"\nEvaluating: [ {scaler_name} | {metric.upper()} ]")

            for k in range(1, max_k + 1):
                fold_f1, fold_top1, fold_top5, fold_auc = [], [], [], []

                for fold_idx, (train_idx, test_idx) in enumerate(
                    cv.split(X, y, groups)
                ):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    if (
                        fold_idx == 1
                        and k == 1
                        and metric == metrics_list[0]
                        and scaler_name == list(scalers.keys())[0]
                    ):
                        print_fold_distribution(y_train, y_test, fold_idx)

                    # Scaling
                    X_train_scaled = scaler_obj.fit_transform(X_train)
                    X_test_scaled = scaler_obj.transform(X_test)

                    # Train KNN
                    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                    knn.fit(X_train_scaled, y_train)

                    # Predict
                    y_pred = knn.predict(X_test_scaled)
                    y_probs = knn.predict_proba(X_test_scaled)

                    # Rank classes for Top-K
                    sorted_indices = np.argsort(y_probs, axis=1)[:, ::-1]
                    y_pred_ranked = knn.classes_[sorted_indices]

                    # Custom metrics
                    f1 = calculate_macro_f1_score(y_test, y_pred)
                    top1 = calculate_top_k_accuracy(y_test, y_pred_ranked, k=1)
                    top5 = calculate_top_k_accuracy(y_test, y_pred_ranked, k=5)

                    # OVR Macro AUC
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
                        "Top-1 Acc": np.mean(fold_top1),
                        "Top-5 Acc": np.mean(fold_top5),
                        "Macro F1": np.mean(fold_f1),
                        "Macro AUC": np.mean(fold_auc),
                    }
                )

                print(
                    f"k={k:02d} | F1: {np.mean(fold_f1):.4f} | AUC: {np.mean(fold_auc):.4f}"
                )

    results_df = pd.DataFrame(results)
    return results_df


def save_and_display_best_results(
    results_df: DataFrame, output_dir: str = "outputs"
) -> None:
    """Finds the optimal k for each configuration, displays summary, and saves to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "knn_evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n=== Optimal Model Configurations ===")

    # Group by both Scaler and Distance
    combinations = results_df[["Scaler", "Distance"]].drop_duplicates()

    for _, row in combinations.iterrows():
        scaler = row["Scaler"]
        metric = row["Distance"]
        subset = results_df[
            (results_df["Scaler"] == scaler) & (results_df["Distance"] == metric)
        ]

        # Selecting best k based on Top-1 Accuracy
        best_row = subset.loc[subset["Top-1 Acc"].idxmax()]

        print(f"\n[{scaler}] + [{metric.upper()}]:")
        print(f" - Optimal k : {best_row['k']}")
        print(f" - Top-1 Acc : {best_row['Top-1 Acc']:.4f}")
        print(f" - Top-5 Acc : {best_row['Top-5 Acc']:.4f}")
        print(f" - Macro F1  : {best_row['Macro F1']:.4f}")
        print(f" - Macro AUC : {best_row['Macro AUC']:.4f}")

    print(f"\n[+] Full evaluation results saved to: {csv_path}\n")
