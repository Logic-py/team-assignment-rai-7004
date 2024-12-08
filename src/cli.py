"""CLI module."""

import argparse

from loguru import logger

from ml_pipeline_handler.pipeline.base_config import PipelineConfig
from ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory


def cli() -> None:
    """CLI wrapper for the main application.

    Example usage:
        python src/cli.py --data_path=src/data/housing.csv --features "OverallQual" "GrLivArea" "GarageCars"
         "GarageArea" "TotalBsmtSF" --target_column=SalePrice --algorithm=linear_regression --random_state=42
          --num_folds=5 --out_file=whatever.pkl

    Returns:
        None

    """
    logger.info("[START] CLI")

    parser = argparse.ArgumentParser(
        description="ML Pipeline Application",
    )
    parser.add_argument("--data_path", required=True, help="Path to the CSV file")
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="List of features to include in the model (e.g., 'feature1 feature2 feature3')",
    )
    parser.add_argument("--target_column", required=True, help="Name of the target column")
    parser.add_argument("--algorithm", required=True, help="Algorithm name (e.g., 'RandomForest')")
    parser.add_argument("--out_file", required=True, help="Where to store the output file")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    logger.info(
        "Starting Main function with: %s, %s, %s, %s, %s, %s, %s",
        args.data_path,
        args.features,
        args.target_column,
        args.algorithm,
        args.out_file,
        args.random_state,
        args.num_folds,
    )

    pipeline_config = PipelineConfig(
        data_path=args.data_path,
        features=args.features,
        target_column=args.target_column,
        algorithm=args.algorithm,
        out_file=args.out_file,
        random_state=args.random_state,
        num_folds=args.num_folds,
    )

    pipeline = PipelineFactory.build_pipeline(config=pipeline_config)
    prediction = pipeline.predict()

    metrics = pipeline.compute_metrics(prediction=prediction)
    logger.info(f"Metrics: {metrics}")

    # TODO: from src.io.saver import save_data -> save_model(pipeline, "model.pkl")

    logger.info("[END] CLI")


if __name__ == "__main__":
    cli()
