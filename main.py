import sys
import argparse
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from src.data_processing import (
    data_loader,
    flatten_data,
    clean_and_validate_data,
    extract_features,
)
from src.eda import generate_eda_report
from src.visualization import plot_tsne
from src.classification import (
    run_knn_pipeline,
    save_and_display_best_results,
    plot_f1_vs_k,
    evaluate_best_models_roc
)


def main():
    parser = argparse.ArgumentParser(description="Load and process data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/mini_gm_public_v0.1.p",
        help="Path to the data file.",
    )
    args = parser.parse_args()

    print(f"Loading data from file: {args.data_path}")
    try:
        data: dict = data_loader(args.data_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File '{args.data_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        sys.exit(1)

    flattened_data: list = flatten_data(data)
    dataframe: DataFrame = pd.DataFrame(flattened_data)
    print("Data successfully flattened and converted to DataFrame.")

    validated_df: DataFrame = clean_and_validate_data(dataframe)
    generate_eda_report(validated_df)
    plot_tsne(validated_df)
    results_df = run_knn_pipeline(validated_df, max_k=15, n_splits=10)
    save_and_display_best_results(results_df)
    plot_f1_vs_k(results_df)

    X = extract_features(validated_df)
    y = LabelEncoder().fit_transform(validated_df["syndrome_id"].values)
    groups = validated_df["subject_id"].values
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    static_folds = list(cv.split(X, y, groups))

    evaluate_best_models_roc(
        validated_df,
        best_k_cos=14,
        best_k_euc=14,
        folds=static_folds,
    )


if __name__ == "__main__":
    main()