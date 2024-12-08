"""Classification Metrics Handler Module."""

from typing import Optional

from loguru import logger
from numpy import ndarray
from pandas import Series
from sklearn.metrics import accuracy_score, roc_auc_score

from src.ml_pipeline_handler.metric.base_metric_handler import BaseMetricsHandler
from src.ml_pipeline_handler.metric.classification.result import (
    ClassificationMetricResult,
)


class ClassificationMetricHandler(BaseMetricsHandler):
    """Classification Metric Handler."""

    @staticmethod
    def compute_accuracy_score(y_true: Series, y_pred: ndarray) -> float:
        """Compute the accuracy score for a classification task.

        Args:
            y_true (Series): The ground truth (actual) labels.
            y_pred (ndarray): The predicted labels.

        Returns:
            float: The accuracy score, representing the proportion of correct predictions out of the total number of
            predictions.

        """
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def compute_area_under_curve_score(y_true: Series, y_proba: ndarray) -> float:
        """Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

        Args:
            y_true (Series): The ground truth (actual) binary labels.
            y_proba (ndarray): The predicted probabilities for the positive class.

        Returns:
            float: The ROC AUC score, which measures the ability of the model to distinguish between the positive and
            negative classes.

        """
        return roc_auc_score(y_true=y_true, y_score=y_proba)

    def compute_metrics(
        self, y_true: Series, y_pred: ndarray, y_proba: Optional[ndarray] = None
    ) -> ClassificationMetricResult:
        """Compute all relevant metrics for a given task and return them in a structured datamodel.

        Args:
            y_true (Series): The ground truth (actual) values.
            y_pred (ndarray): The predicted values.
            y_proba (Optional[ndarray]): The predicted probabilities for the positive class (classification only).

        Returns:
            MetricsResults: A datamodel containing all computed metrics.

        """
        accuracy = ClassificationMetricHandler.compute_accuracy_score(y_true=y_true, y_pred=y_pred)

        if y_proba is None:
            logger.error("Variable [y_proba] is mandatory in ClassificationMetricHandler")
            raise ValueError

        roc_auc = ClassificationMetricHandler.compute_area_under_curve_score(y_true=y_true, y_proba=y_proba)

        return ClassificationMetricResult(accuracy=accuracy, roc_auc=roc_auc)
