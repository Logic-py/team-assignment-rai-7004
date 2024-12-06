"""Base Metric Handler Abstract Class Module."""

from abc import ABC, abstractmethod
from typing import Optional

from numpy import ndarray

from src.metric.base_result import BaseMetricResult
from src.metric.model_type import ModelType


class BaseMetricsHandler(ABC):
    """Base Metric Handler."""

    def __init__(self, model_type: ModelType) -> None:
        """Initialize Class with given model type.

        Args:
            model_type: Type of Machine Learning Model.

        """
        self.model_type = model_type

    @abstractmethod
    def compute_metrics(self, y_true: ndarray, y_pred: ndarray, y_proba: Optional[ndarray] = None) -> BaseMetricResult:
        """Compute all relevant metrics for a given task and return them in a structured datamodel.

        Args:
            y_true (ndarray): The ground truth (actual) values.
            y_pred (ndarray): The predicted values.
            y_proba (Optional[ndarray]): The predicted probabilities for the positive class (classification only).

        Returns:
            MetricsResults: A datamodel containing all computed metrics.

        """
        raise NotImplementedError
