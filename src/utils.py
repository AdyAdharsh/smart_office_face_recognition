import pickle
import os

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        