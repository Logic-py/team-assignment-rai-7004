"""Regression Metric results module."""

from dataclasses import dataclass

from ...metric.base_result import BaseMetricResult


@dataclass
class RegressionMetricResult(BaseMetricResult):
    """Regression Metric results."""

    mean_absolute_error: float
    mean_squared_error: float
    r_square: float
