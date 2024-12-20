"""Metrics Factory Module."""

from loguru import logger

from src.ml_pipeline_handler.metric.base_metric_handler import BaseMetricsHandler
from src.ml_pipeline_handler.metric.classification.handler import ClassificationMetricHandler
from src.ml_pipeline_handler.metric.model_type import ModelType
from src.ml_pipeline_handler.metric.regression.handler import RegressionMetricHandler


class MetricFactory:
    """Metrics Factory pattern Class."""

    @classmethod
    def get_metrics_handler(cls, model_type: ModelType) -> BaseMetricsHandler:
        """Instantiate the concrete Metrics Handler Class based on the model type.

        Args:
            model_type: Type of Machine Learning Model.

        Returns:
            The Concrete Metrics Handler Class based on given model type.

        """
        if model_type == ModelType.REGRESSION:
            return RegressionMetricHandler(model_type=model_type)
        if model_type == ModelType.CLASSIFICATION:
            return ClassificationMetricHandler(model_type=model_type)
        logger.error(f"Model Type: {model_type} is not valid.")
        raise NotImplementedError
