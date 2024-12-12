"""Data Saver Module."""

import pickle

from src.ml_pipeline_handler.pipeline.base_pipeline import BasePipeline


def save_model(model: BasePipeline, file_name: str) -> None:
    """Save the model as a pickle file.

    Args:
        model: The pipeline to be saved
        file_name: The target file name, with optional directory and mandatory .pkl extension

    Returns:
        None

    """
    with open(file_name, "w") as f:
        pickle.dump(model, f)
