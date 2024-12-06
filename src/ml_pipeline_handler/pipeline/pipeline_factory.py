"""Pipeline Factory Module."""

from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.base_pipeline import BasePipeline
from src.ml_pipeline_handler.pipeline.linear_regression_pipeline import LinearRegressionPipeline


class PipelineFactory:
    """Pipeline Factory class."""

    @classmethod
    def build_pipeline(cls, config: PipelineConfig) -> BasePipeline:
        """Build a pipeline based on the configuration.

        Args:
            config: PipelineConfig, contains configuration information for a pipeline.

        Returns:
            BasePipeline object that contains a specific model type.

        """
        if config.algorithm == AlgorithmType.LINEAR_REGRESSION:
            return LinearRegressionPipeline(config=config)

        # TODO: the other model types.
        raise NotImplementedError
