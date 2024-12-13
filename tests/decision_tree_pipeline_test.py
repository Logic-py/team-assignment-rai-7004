from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory


def test_decision_tree_classifier_pipeline():
    config = PipelineConfig(
        data_path="data/test_data.csv",
        features=["feature1", "feature2"],
        target_column="target",
        algorithm=AlgorithmType.DECISION_TREE_CLASSIFIER,
        out_file="models/decision_tree_classifier.pkl",
        random_state=42,
    )
    pipeline = PipelineFactory.build_pipeline(config=config)
    assert pipeline is not None


def test_decision_tree_regressor_pipeline():
    config = PipelineConfig(
        data_path="data/test_data.csv",
        features=["feature1", "feature2"],
        target_column="target",
        algorithm=AlgorithmType.DECISION_TREE_REGRESSOR,
        out_file="models/decision_tree_regressor.pkl",
        random_state=42,
    )
    pipeline = PipelineFactory.build_pipeline(config=config)
    assert pipeline is not None
