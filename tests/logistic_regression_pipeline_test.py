from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory

# Have been replaced - data_path="data/test_data.csv",
# Have been replaced - features=["feature1", "feature2"],
# Have been replaced - target_column="target",


def test_logistic_regression_classifier_pipeline():
    config = PipelineConfig(
        data_path="data/Iris.csv",
        features=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        target_column="Species",
        algorithm=AlgorithmType.LOGISTIC_REGRESSION,
        out_file="models/logistic_regression.pkl",
        random_state=42,
    )
    pipeline = PipelineFactory.build_pipeline(config=config)
    assert pipeline is not None
