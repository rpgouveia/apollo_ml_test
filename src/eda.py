import pandas as pd
from pandas import DataFrame, Series


def generate_dataset_statistics(dataframe: DataFrame):
    """Gera estatísticas descritivas do DataFrame."""
    print("Estatísticas do conjunto de dados:")
    total_syndrome = dataframe['syndrome_id'].nunique()
    total_subjects = dataframe['subject_id'].nunique()
    total_images = dataframe['image_id'].nunique()

    print("Visão geral:")
    print(f"Total de síndromes únicas: {total_syndrome}")
    print(f"Total de sujeitos únicos: {total_subjects}")
    print(f"Total de imagens únicas: {total_images}\n")

    print("Imagens por síndrome:")
    images_per_syndrome: Series[int] = dataframe['syndrome_id'].value_counts()
    print(images_per_syndrome.to_string())
    print("\nResumo: Imagens por síndrome")
    print(f"Média: {images_per_syndrome.mean():.2f}")
    print(f"Mediana: {images_per_syndrome.median()}")
    print(f"Desvio padrão: {images_per_syndrome.std():.2f}")
    print(f"Mínimo: {images_per_syndrome.min()}")
    print(f"Máximo: {images_per_syndrome.max()}\n")

    print("Indivíduos por síndrome:")
    subjects_per_syndrome = dataframe.groupby('syndrome_id')['subject_id'].nunique().sort_values(ascending=False)
    print(subjects_per_syndrome.to_string())


def check_class_balance(dataframe: DataFrame, target_column: str = 'syndrome_id') -> DataFrame:
    """Verifica o balanceamento das classes no DataFrame por frequência."""
    count: Series[int] = dataframe[target_column].value_counts()
    percentage: Series[float] = dataframe[target_column].value_counts(normalize=True) * 100

    summary: DataFrame = pd.DataFrame({
        'Quantidade': count,
        'Percentual (%)': percentage
    })
    print(summary)

    majority_class: str = summary['Percentual (%)'].max()
    minority_class: str = summary['Percentual (%)'].min()
    print("\nResumo:")
    print(f"Classe mais frequente tem: {majority_class:.2f}% dos dados.")
    print(f"Classe menos frequente tem: {minority_class:.2f}% dos dados.\n")

    if (majority_class / minority_class) > 2:
        print("Atenção: O conjunto de dados parece estar desbalanceado!\n")
    else:
        print("O conjunto de dados parece estar relativamente bem balanceado.\n")

    return summary