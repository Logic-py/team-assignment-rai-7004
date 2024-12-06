"""Pipeline Factory Module."""

from ..algorithm import AlgorithmType
from ..pipeline.base_config import PipelineConfig
from ..pipeline.base_pipeline import BasePipeline
from ..pipeline.linear_regression_pipeline import LinearRegressionPipeline


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
