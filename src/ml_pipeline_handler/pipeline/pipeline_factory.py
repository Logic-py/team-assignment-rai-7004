"""Pipeline Factory Module.

This module implements a factory class for constructing specific pipelines
based on the provided configuration. The pipelines support algorithms such as
linear regression, classification, and regression using decision trees.
"""

from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.base_pipeline import BasePipeline
from src.ml_pipeline_handler.pipeline.decision_tree_classifier_pipeline import (
    DecisionTreeClassifierPipeline,
)
from src.ml_pipeline_handler.pipeline.decision_tree_regressor_pipeline import (
    DecisionTreeRegressorPipeline,
)
from src.ml_pipeline_handler.pipeline.linear_regression_pipeline import (
    LinearRegressionPipeline,
)


class PipelineFactory:
    """Pipeline Factory class."""

    @classmethod
    def build_pipeline(cls, config: PipelineConfig) -> BasePipeline:
        """Build a pipeline based on the configuration.

        Args:
            config: PipelineConfig, contains configuration information for a pipeline.

        Returns:
            BasePipeline object that contains a specific model type.

        Raises:
            NotImplementedError: If the specified algorithm is not supported.

        """
        if config.algorithm == AlgorithmType.LINEAR_REGRESSION:
            return LinearRegressionPipeline(config=config)
        if config.algorithm == AlgorithmType.DECISION_TREE_CLASSIFIER:
            return DecisionTreeClassifierPipeline(config=config)
        if config.algorithm == AlgorithmType.DECISION_TREE_REGRESSOR:
            return DecisionTreeRegressorPipeline(config=config)

        raise NotImplementedError(f"Algorithm '{config.algorithm}' is not implemented.")
