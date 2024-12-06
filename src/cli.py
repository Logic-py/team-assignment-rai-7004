"""CLI module."""

import argparse

from loguru import logger

from main import main


def cli() -> None:
    """CLI wrapper for the main application.

    Example usage:
        python src/cli.py --data_path=whatever --target_column=whatever --algorithm=whatever --random_state=42 --num_folds=5 --out_file=daswhaw

    Returns:
        None

    """
    logger.info("[START] CLI")

    parser = argparse.ArgumentParser(
        description="ML Pipeline Application",
    )
    parser.add_argument("--data_path", required=True, help="Path to the CSV file")
    parser.add_argument("--target_column", required=True, help="Name of the target column")
    parser.add_argument("--algorithm", required=True, help="Algorithm name (e.g., 'RandomForest')")
    parser.add_argument("--out_file", required=True, help="Where to store the output file")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        target_column=args.target_column,
        algorithm=args.algorithm,
        out_file=args.out_file,
        random_state=args.random_state,
        num_folds=args.num_folds,
    )

    logger.info("[END] CLI")


if __name__ == "__main__":
    cli()
