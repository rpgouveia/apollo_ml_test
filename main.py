import argparse
import pandas as pd
from pandas import DataFrame
from src.data_processing import data_loader, flatten_data, clean_and_validate_data
from src.eda import generate_eda_report

def main():
    parser = argparse.ArgumentParser(description='Carregar e processar dados.')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/mini_gm_public_v0.1.p',
        help='Caminho para o arquivo de dados'
    )
    args = parser.parse_args()

    print(f"Carregando dados do arquivo: {args.data_path}")
    try:
        data: dict = data_loader(args.data_path)
        print("Dados carregados com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{args.data_path}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao carregar os dados: {e}")
        return

    flattened_data: list = flatten_data(data)
    dataframe: DataFrame = pd.DataFrame(flattened_data)
    print("Dados achatados e convertidos para DataFrame com sucesso.")

    validated_df: DataFrame = clean_and_validate_data(dataframe)
    generate_eda_report(validated_df)

if __name__ == '__main__':
    main()