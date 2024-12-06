from ..pipeline.decision_tree_classifier_pipeline import DecisionTreeClassifierPipeline
from ..pipeline.decision_tree_regressor_pipeline import DecisionTreeRegressorPipeline

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
        elif config.algorithm == AlgorithmType.DECISION_TREE_CLASSIFIER:
            return DecisionTreeClassifierPipeline(config=config)
        elif config.algorithm == AlgorithmType.DECISION_TREE_REGRESSOR:
            return DecisionTreeRegressorPipeline(config=config)

        raise NotImplementedError(f"Algorithm '{config.algorithm}' is not implemented.")
