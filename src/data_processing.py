import pickle


def data_loader(file_path: str):
    """Carrega dados de um arquivo pickle."""
    with open(file_path, 'rb') as file:
        data: dict = pickle.load(file)
    return data

