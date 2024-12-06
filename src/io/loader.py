"""Data Loader Module."""

from pandas import DataFrame, Series, read_csv


def load_data(data_path: str, target_column: str) -> tuple[DataFrame, Series]:
    """Load the data as a Dataframe and split it into features and target column.

    Args:
        data_path: The path of the data to load
        target_column: The target column of the dataset.

    Returns:
        A tuple of a DataFrame and a Series representing the features and target column.

    """
    data = read_csv(filepath_or_buffer=data_path)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    x = data.drop(columns=[target_column])
    y = data[target_column]
    return x, y
