import argparse
import pandas as pd
from src.data_processing import data_loader, flatten_data


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
        data = data_loader(args.data_path)
        print("Dados carregados com sucesso.")
        flattened_data = flatten_data(data)
        df = pd.DataFrame(flattened_data)
        print("Dados achatados e convertidos para DataFrame com sucesso.\n")

        print("Estrutura do DataFrame:")
        print(df.head())

    except FileNotFoundError:
        print(f"Erro: O arquivo '{args.data_path}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao carregar os dados: {e}")

if __name__ == '__main__':
    main()

