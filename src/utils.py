import pickle
import os

def load_embeddings(file_path):
    # 1. Load the raw dictionary data
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # If the file doesn't exist, start with an empty dictionary
        data = {}

    # 2. Extract keys (names) and values (embeddings) from the dictionary
    # The keys of the dictionary are the names, and the values are the embeddings (list of arrays)
    known_names = list(data.keys())
    known_embeddings = list(data.values())

    # 3. Return the two required lists
    return known_embeddings, known_names

def save_embeddings(file_path, data):
    # This function should save the dictionary (data) you use for registration
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)