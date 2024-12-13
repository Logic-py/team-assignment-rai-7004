"""Example in python code."""

from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory

if __name__ == "__main__":
    data_path: str = "src/data/housing.csv"
    features: list[str] = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF"]
    target_column: str = "SalePrice"
    scale_standard: list[str] = ["OverallQual"]
    scale_robust: list[str] = ["TotalBsmtSF"]
    algorithm = AlgorithmType.LINEAR_REGRESSION
    out_file: str = "example out file.pkl"
    random_state: int = 42

    pipeline_config = PipelineConfig(
        data_path=data_path,
        features=features,
        scale_standard=scale_standard,
        scale_robust=scale_robust,
        target_column=target_column,
        algorithm=algorithm,
        out_file=out_file,
        random_state=random_state,
    )

    pipeline = PipelineFactory.build_pipeline(config=pipeline_config)
    prediction, probability = pipeline.predict()
    pipeline.compute_metrics(prediction=prediction)
