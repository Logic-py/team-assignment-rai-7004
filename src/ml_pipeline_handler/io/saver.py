"""Data Saver Module."""
import pickle

def save_model(model, file_name) -> None:
    with open(file_name+'.pkl', 'wb') as f:
        pickle.dump(model, f)
    return None