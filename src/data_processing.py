import pickle


def data_loader(file_path: str):
    """Carrega dados de um arquivo pickle."""
    with open(file_path, 'rb') as file:
        data: dict = pickle.load(file)
    return data

def flatten_data(data):
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