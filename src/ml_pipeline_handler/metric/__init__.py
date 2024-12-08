"""Initialization module for metrics.

This package includes handlers, results, and utilities for metrics
computation in classification and regression models.
"""

from src.ml_pipeline_handler.metric.base_metric_handler import BaseMetricsHandler
from src.ml_pipeline_handler.metric.base_result import BaseMetricResult
from src.ml_pipeline_handler.metric.classification.handler import (
    ClassificationMetricHandler,
)
from src.ml_pipeline_handler.metric.classification.result import (
    ClassificationMetricResult,
)
from src.ml_pipeline_handler.metric.metric_factory import MetricFactory
from src.ml_pipeline_handler.metric.model_type import ModelType
from src.ml_pipeline_handler.metric.regression.handler import RegressionMetricHandler
from src.ml_pipeline_handler.metric.regression.result import RegressionMetricResult

__all__ = [
    "BaseMetricResult",
    "BaseMetricsHandler",
    "ClassificationMetricHandler",
    "ClassificationMetricResult",
    "MetricFactory",
    "ModelType",
    "RegressionMetricHandler",
    "RegressionMetricResult",
]
