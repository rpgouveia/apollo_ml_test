import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
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

                    # weights='distance' solves arbitrary index-based tie-breaking for small k
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

                # ddof adjusted for sample standard deviation
                # https://numpy.org/doc/stable/reference/generated/numpy.std.html
                # fold_f1 saved for pairwise comparison
                results.append({
                    'Scaler': scaler_name,
                    'Distance': metric,
                    'k': k,
                    'Top-1 Acc (Mean)': np.mean(fold_top1),
                    'Top-1 Acc (Std)': np.std(fold_top1, ddof=1),
                    'Top-5 Acc (Mean)': np.mean(fold_top5),
                    'Top-5 Acc (Std)': np.std(fold_top5, ddof=1),
                    'Macro F1 (Mean)': np.mean(fold_f1),
                    'Macro F1 (Std)': np.std(fold_f1, ddof=1),
                    'Macro AUC (Mean)': np.mean(fold_auc),
                    'Macro AUC (Std)': np.std(fold_auc, ddof=1),
                    'Fold F1s': fold_f1
                })

                print(
                    f"k={k:02d} | F1: {np.mean(fold_f1):.4f}±{np.std(fold_f1):.4f} | AUC: {np.mean(fold_auc):.4f}±{np.std(fold_auc):.4f}"
                )

    return pd.DataFrame(results)


def plot_f1_vs_k(results_df: DataFrame, output_dir: str = 'outputs') -> None:
    """Generates the F1 stabilization plot as a function of k, with standard deviation bands."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    raw_df = results_df[results_df['Scaler'] == 'Raw Data']
    
    plt.figure(figsize=(10, 6))
    colors = {'cosine': 'blue', 'euclidean': 'orange'}
    
    for metric in ['cosine', 'euclidean']:
        sub = raw_df[raw_df['Distance'] == metric]
        k_vals = sub['k'].values
        f1_mean = sub['Macro F1 (Mean)'].values
        f1_std = sub['Macro F1 (Std)'].values
        
        plt.plot(k_vals, f1_mean, label=f'{metric.capitalize()}', color=colors[metric], marker='o')
        plt.fill_between(k_vals, f1_mean - f1_std, f1_mean + f1_std, color=colors[metric], alpha=0.2)
        
    plt.title('Macro F1 Score vs. k Neighbors (Raw Data)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Macro F1 Score')
    plt.xticks(range(1, 16))
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(output_dir, 'f1_vs_k_raw_data.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] F1 vs k plot saved to: {output_path}")


def evaluate_best_models_roc(dataframe: DataFrame, best_k_cos: int, best_k_euc: int, output_dir: str = 'outputs') -> None:
    """Evaluates the best models on the raw data to generate the interpolated ROC curve and per-class metrics."""
    X = extract_features(dataframe)
    le = LabelEncoder()
    y = le.fit_transform(dataframe['syndrome_id'].values)
    groups = dataframe['subject_id'].values
    
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(cv.split(X, y, groups))
    
    configs = {'Cosine': best_k_cos, 'Euclidean': best_k_euc}
    mean_fpr = np.linspace(0, 1, 100)
    roc_data = {}
    f1_fold_history = {}
    
    for metric_name, k in configs.items():
        tprs = []
        fold_f1_scores = []
        
        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric_name.lower(), weights='distance')
            knn.fit(X_train, y_train)
            y_probs = knn.predict_proba(X_test)
            y_pred = knn.classes_[np.argsort(-y_probs, axis=1, kind='mergesort')[:, 0]]
            
            fold_f1_scores.append(calculate_macro_f1_score(y_test, y_pred))
            
            # OVR ROC Calculation for the Fold
            fold_tprs = []
            for idx, cls in enumerate(knn.classes_):
                y_true_bin = (y_test == cls).astype(int)
                if np.sum(y_true_bin) > 0:
                    auc_val, fpr_val, tpr_val = calculate_binary_roc(y_true_bin, y_probs[:, idx])
                    # Interpolation to ensure the same X-axis (FPR) across all folds
                    interp_tpr = np.interp(mean_fpr, fpr_val, tpr_val)
                    interp_tpr[0] = 0.0
                    fold_tprs.append(interp_tpr)
                    
            # Average across classes (Macro) for the current fold
            tprs.append(np.mean(fold_tprs, axis=0))
        
        f1_fold_history[metric_name] = fold_f1_scores
        roc_data[metric_name] = np.mean(tprs, axis=0) # Average of the 10 folds
    
    # 1. Paired Comparison (Paired Win-Rate)
    wins_cos = sum(1 for c, e in zip(f1_fold_history['Cosine'], f1_fold_history['Euclidean']) if c > e)
    ties = sum(1 for c, e in zip(f1_fold_history['Cosine'], f1_fold_history['Euclidean']) if c == e)
    wins_euc = 10 - wins_cos - ties
    
    print(f"\n=== Paired Comparison (Fold-by-Fold on Raw Data) ===")
    print(f"Cosine wins: {wins_cos}/10 folds")
    print(f"Euclidean wins: {wins_euc}/10 folds")
    print(f"Ties: {ties}/10 folds")
    
    # 2. Plotting the Averaged ROC Curve
    plt.figure(figsize=(10, 8))
    for metric_name, mean_tpr in roc_data.items():
        mean_tpr[-1] = 1.0 # Ensures convergence at the top right
        macro_auc = np.trapezoid(mean_tpr, mean_fpr)
        plt.plot(mean_fpr, mean_tpr, label=f'{metric_name} (Macro AUC = {macro_auc:.4f})', lw=2)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random Guess')
    plt.title('10-Fold CV Averaged ROC Curves (OVR Macro)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(output_dir, 'roc_curves_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] ROC curves plot saved to: {output_path}\n")


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