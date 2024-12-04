"""Metrics test module."""

from src.metric.base_metric_handler import BaseMetricsHandler
from src.metric.classification.handler import ClassificationMetricHandler
from src.metric.metric_factory import MetricFactory
from src.metric.model_type import ModelType
from src.metric.regression.handler import RegressionMetricHandler


def test_get_metrics_handler_regression() -> None:
    """Test that the factory returns a RegressionMetricHandler for regression models.

    Returns:
        None
    """
    handler = MetricFactory.get_metrics_handler(model_type=ModelType.REGRESSION)
    assert isinstance(handler, RegressionMetricHandler), "Expected a RegressionMetricHandler instance."
    assert isinstance(handler, BaseMetricsHandler), "Expected handler to be a subclass of BaseMetricsHandler."


def test_get_metrics_handler_classification() -> None:
    """Test that the factory returns a ClassificationMetricHandler for classification models.

    Returns:
        None
    """

    handler = MetricFactory.get_metrics_handler(model_type=ModelType.CLASSIFICATION)
    assert isinstance(handler, ClassificationMetricHandler), "Expected a ClassificationMetricHandler instance."
    assert isinstance(handler, BaseMetricsHandler), "Expected handler to be a subclass of BaseMetricsHandler."
