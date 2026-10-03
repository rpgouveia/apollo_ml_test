import argparse
import sys

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.classification import (
    evaluate_best_models_roc,
    plot_f1_vs_k,
    run_knn_pipeline,
    save_and_display_best_results,
)
from src.data_processing import (
    clean_and_validate_data,
    data_loader,
    extract_features,
    flatten_data,
)
from src.eda import generate_eda_report
from src.visualization import plot_tsne


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
    except ValueError as e:
        print(f"An error occurred while loading the data: {e}")
        sys.exit(1)

    # Flatten the data and convert it to a DataFrame
    flattened_data: list = flatten_data(data)
    dataframe: DataFrame = pd.DataFrame(flattened_data)
    print("Data successfully flattened and converted to DataFrame.")

    # Clean and validate the data, then generate EDA report and t-SNE plot
    validated_df: DataFrame = clean_and_validate_data(dataframe)
    generate_eda_report(validated_df)
    plot_tsne(validated_df)

    # Create a Single Source of Truth (SSOT) for the Folds
    print("\n=== Preparing Cross-Validation Folds ===")
    X = extract_features(validated_df)
    y = LabelEncoder().fit_transform(validated_df['syndrome_id'].values)
    groups = validated_df['subject_id'].values
    
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    static_folds = list(cv.split(X, y, groups))

    # Inject the folds into the pipeline and retrieve the dictionary with the best k values
    results_df = run_knn_pipeline(validated_df, folds=static_folds, max_k=15)
    best_ks = save_and_display_best_results(results_df)

    # Extraction of dynamic insights for the report
    plot_f1_vs_k(results_df)

    # Final evaluation using the rigorously determined optimal k values from the pipeline
    evaluate_best_models_roc(
        validated_df, 
        best_k_cos=best_ks['cosine'], 
        best_k_euc=best_ks['euclidean'], 
        folds=static_folds
    )


if __name__ == "__main__":
    main()