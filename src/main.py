"""Main Module."""

from loguru import logger

from algorithm import AlgorithmType


def main(
    data_path: str,
    target_column: str,
    algorithm: AlgorithmType,
    out_file: str,
    random_state: int = 42,
    num_folds: int = 5,
) -> None:
    """Entry point for the ML pipeline application.

    Args:
        data_path: str, Path to the CSV file.
        target_column: str, Name of the target column in the dataset.
        algorithm: str, Name of the ML algorithm.
        out_file: str, Name of the outfile for model pipelines.
        random_state: int, Random seed for reproducibility.
        num_folds: int, Number of cross-validation folds.

    Returns:
        None

    """
    logger.info(
        f"Starting Main function with: {data_path}, {target_column}, {algorithm}, {out_file},"
        f" {random_state}, {num_folds}"
    )

    # TODO: from src.io.saver import save_data
    # pipeline, metrics = build_pipeline(
    #     data_path=data_path,
    #     target_column=target_column,
    #     algorithm=algorithm,
    #     random_state=random_state,
    #     num_folds=num_folds
    # )

    # TODO
    # save_model(pipeline, "model.pkl")

    # logger.info(f"Metrics: {metrics}")
