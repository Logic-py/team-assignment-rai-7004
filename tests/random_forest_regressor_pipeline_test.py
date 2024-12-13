from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory


def test_random_forest_regressor_pipeline():
    config = PipelineConfig(
        data_path="data/housing.csv",
        features=["BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "GarageCars", "OpenPorchSF"],
        target_column="SalePrice",
        algorithm=AlgorithmType.RANDOM_FOREST_REGRESSOR,
        out_file="models/random_forest_regressor.pkl",
        random_state=42,
        num_folds=5,
    )
    pipeline = PipelineFactory.build_pipeline(config=config)
    assert pipeline is not None