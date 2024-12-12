"""Data Saver Module."""
import pickle

def save_model(model, file_name) -> None:
    """Save the model as a pickle file.

    Args:
        model: The pipeline to be saved
        file_name: The target file name, with optional directory and mandatory .pkl extension

    Returns:
        None

    """
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
