import pandas as pd
from pandas import DataFrame, Series


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