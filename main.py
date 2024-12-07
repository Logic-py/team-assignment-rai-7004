"""Main Module."""

from loguru import logger
from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory


def main(
    data_path: str,
    features: list[str],
    target_column: str,
    algorithm: AlgorithmType,
    out_file: str,
    random_state: int = 42,
    num_folds: int = 5,
) -> None:
    """Entry point for the ML pipeline application.

    Args:
        data_path: str, Path to the CSV file.
        features: list[str], The list of features to analyze.
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

    pipeline_config = PipelineConfig(
        data_path=data_path,
        features=features,
        target_column=target_column,
        algorithm=algorithm,
        out_file=out_file,
        random_state=random_state,
        num_folds=num_folds,
    )

    pipeline = PipelineFactory.build_pipeline(config=pipeline_config)
    prediction = pipeline.predict()

    metrics = pipeline.compute_metrics(prediction=prediction)
    logger.info(f"Metrics: {metrics}")

    # TODO: from src.io.saver import save_data
    # save_model(pipeline, "model.pkl")


if __name__ == "__main__":
    main(
        data_path="src/data/housing.csv",
        features=[
            "OverallQual",
            "GrLivArea",
            "GarageCars",
            "GarageArea",
            "TotalBsmtSF",
        ],
        target_column="SalePrice",
        algorithm=AlgorithmType.LINEAR_REGRESSION,
        out_file="",
        random_state=42,
        num_folds=5,
    )
