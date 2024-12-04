"""Classification Metric results module."""

from dataclasses import dataclass

from src.metric.base_result import BaseMetricResult


@dataclass
class ClassificationMetricResult(BaseMetricResult):
    """Classification Metric results."""

    accuracy: float
    roc_auc: float