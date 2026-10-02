import pickle
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

def data_loader(file_path: str) -> dict:
    """Loads data from a pickle file."""
    with open(file_path, 'rb') as file:
        data: dict = pickle.load(file)
    return data

def flatten_data(data: dict) -> list:
    """Flattens the data structure into a list of embeddings."""
    flattened_data: list = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                flattened_data.append({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    'embedding': embedding
                })
    return flattened_data

def remove_missing_values(dataframe: DataFrame) -> DataFrame:
    """Removes rows with missing values in the DataFrame."""
    return dataframe.dropna().copy()

def remove_duplicate_images(dataframe: DataFrame) -> DataFrame:
    """Removes records with duplicate 'image_id'."""
    return dataframe.drop_duplicates(subset=['image_id']).copy()

def filter_by_embedding_dimension(dataframe: DataFrame, expected_dimension: int = 320) -> DataFrame:
    """Keeps only rows where the embedding has the expected dimension."""
    mask: Series[bool] = dataframe['embedding'].apply(lambda x: len(x) == expected_dimension)
    return dataframe[mask].copy()

def is_valid_vector(vector: list) -> bool:
    """Checks if there are NaN or Infinite values within the vectors."""
    array: NDArray = np.array(vector)
    return not (np.isnan(array).any() or np.isinf(array).any())

def remove_corrupted_embeddings(dataframe: DataFrame) -> DataFrame:
    """Removes rows with corrupted embeddings."""
    mask: Series[bool] = dataframe['embedding'].apply(is_valid_vector)
    return dataframe[mask].copy()

def clean_and_validate_data(dataframe: DataFrame) -> DataFrame:
    """Applies all data cleaning and validation functions step-by-step."""
    print("\n=== Data Integrity Validation ===")
    initial_rows: int = len(dataframe)
    
    df_no_missing: DataFrame = remove_missing_values(dataframe)
    missing_removed: int = initial_rows - len(df_no_missing)
    
    df_no_duplicates: DataFrame = remove_duplicate_images(df_no_missing)
    duplicates_removed: int = len(df_no_missing) - len(df_no_duplicates)
    
    df_valid_dim: DataFrame = filter_by_embedding_dimension(df_no_duplicates)
    dimension_removed: int = len(df_no_duplicates) - len(df_valid_dim)
    
    df_clean: DataFrame = remove_corrupted_embeddings(df_valid_dim)
    corrupted_removed: int = len(df_valid_dim) - len(df_clean)
    
    final_rows: int = len(df_clean)
    
    print(f"Total original records: {initial_rows}")
    print(f" - Missing values removed: {missing_removed}")
    print(f" - Duplicate 'image_id' removed: {duplicates_removed}")
    print(f" - Invalid dimensions (not 320) removed: {dimension_removed}")
    print(f" - Corrupted embeddings (NaN/Inf) removed: {corrupted_removed}")
    print(f"Total valid records: {final_rows}")
    
    return df_clean