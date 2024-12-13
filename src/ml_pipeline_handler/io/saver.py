"""Data Saver Module."""

import pickle
from pathlib import Path

from src.ml_pipeline_handler.pipeline.base_pipeline import BasePipeline


def save_model(model: BasePipeline, file_name: str) -> None:
    """Save the model as a pickle file.

    Args:
        model: The pipeline to be saved
        file_name: The target file name, with optional directory and mandatory .pkl extension

    Returns:
        None

    """
    file_path = Path(file_name)
    with file_path.open(mode="wb") as f:
        pickle.dump(obj=model, file=f)
