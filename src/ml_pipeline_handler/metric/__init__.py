from .base_metric_handler import BaseMetricsHandler
from .classification.handler import ClassificationMetricHandler
from .classification.result import ClassificationMetricResult  # Aggiunto
from .regression.handler import RegressionMetricHandler
from .regression.result import RegressionMetricResult  # Se necessario
from .model_type import ModelType
from .base_result import BaseMetricResult
from .metric_factory import MetricFactory 

__all__ = [
    "BaseMetricResult",
    "BaseMetricsHandler",
    "ClassificationMetricHandler",
    "ClassificationMetricResult",  # Aggiunto
    "RegressionMetricHandler",
    "RegressionMetricResult",  # Se necessario
    "ModelType",
    "MetricFactory",
]
