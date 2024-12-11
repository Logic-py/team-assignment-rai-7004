from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory


def test_random_forest_pipeline():
    config = PipelineConfig(
        data_path="data/test_data.csv",
        features=["feature1", "feature2"],
        target_column="target",
        algorithm=AlgorithmType.RANDOM_FOREST,
        out_file="models/random_forest_classifier.pkl",
        random_state=42,
        num_folds=5,
    )
    pipeline = PipelineFactory.build_pipeline(config=config)
    assert pipeline is not None


