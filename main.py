import sys
import argparse
import pandas as pd
from pandas import DataFrame
from src.classification import run_knn_pipeline, save_and_display_best_results
from src.data_processing import data_loader, flatten_data, clean_and_validate_data
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


if __name__ == "__main__":
    main()
