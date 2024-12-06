from typing import Optional
from loguru import logger
from numpy import ndarray
from pandas import Series
from sklearn.metrics import accuracy_score, roc_auc_score

from ...metric.base_metric_handler import BaseMetricsHandler
from .result import ClassificationMetricResult


class ClassificationMetricHandler(BaseMetricsHandler):
    """Classification Metric Handler."""

    @staticmethod
    def compute_accuracy_score(y_true: Series, y_pred: ndarray) -> float:
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def compute_area_under_curve_score(y_true: Series, y_proba: ndarray) -> float:
        return roc_auc_score(y_true=y_true, y_score=y_proba)

    def compute_metrics(
        self, y_true: Series, y_pred: ndarray, y_proba: Optional[ndarray] = None
    ) -> ClassificationMetricResult:
        accuracy = ClassificationMetricHandler.compute_accuracy_score(y_true=y_true, y_pred=y_pred)

        if y_proba is None:
            logger.error("Variable [y_proba] is mandatory in ClassificationMetricHandler")
            raise ValueError

        roc_auc = ClassificationMetricHandler.compute_area_under_curve_score(y_true=y_true, y_proba=y_proba)
        return ClassificationMetricResult(accuracy=accuracy, roc_auc=roc_auc)
