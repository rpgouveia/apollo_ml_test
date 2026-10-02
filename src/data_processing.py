import pickle
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

def data_loader(file_path: str) -> dict:
    """Carrega dados de um arquivo pickle."""
    with open(file_path, 'rb') as file:
        data: dict = pickle.load(file)
    return data

def flatten_data(data: dict) -> list:
    """Achata a estrutura de dados para uma lista de embeddings."""
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
    """Remove linhas com valores nulos no DataFrame."""
    return dataframe.dropna().copy()

def remove_duplicate_images(dataframe: DataFrame) -> DataFrame:
    """Remove registros com 'image_id' duplicados."""
    return dataframe.drop_duplicates(subset=['image_id']).copy()

def filter_by_embedding_dimension(dataframe: DataFrame, expected_dimension: int = 320) -> DataFrame:
    """Mantém apenas as linhas cujo embedding tenha a dimensão esperada."""
    mask: Series[bool] = dataframe['embedding'].apply(lambda x: len(x) == expected_dimension)
    return dataframe[mask].copy()

def is_valid_vector(vector: list) -> bool:
    """Verifica se há valores NaN ou Infinitos dentro dos vetores."""
    array: NDArray = np.array(vector)
    return not (np.isnan(array).any() or np.isinf(array).any())

def remove_corrupted_embeddings(dataframe: DataFrame) -> DataFrame:
    """Remove linhas com embeddings corrompidos."""
    mask: Series[bool] = dataframe['embedding'].apply(is_valid_vector)
    return dataframe[mask].copy()

def clean_and_validate_data(dataframe: DataFrame) -> DataFrame:
    """Aplica todas as funções de limpeza e validação de dados detalhadamente."""
    print("\n=== Validação de Integridade ===")
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
    
    print(f"Total de registros originais: {initial_rows}")
    print(f" - Valores nulos removidos: {missing_removed}")
    print(f" - 'image_id' duplicados removidos: {duplicates_removed}")
    print(f" - Dimensão diferente de 320 removidos: {dimension_removed}")
    print(f" - Embeddings com NaN/Inf removidos: {corrupted_removed}")
    print(f"Total de registros íntegros: {final_rows}")
    return df_clean