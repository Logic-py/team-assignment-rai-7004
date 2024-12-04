"""Regression Metrics Handler Module."""

from typing import Optional

from numpy import ndarray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.metric.base_metric_handler import BaseMetricsHandler
from src.metric.regression.result import RegressionMetricResult


class RegressionMetricHandler(BaseMetricsHandler):
    """Regression Metric Handler."""

    @staticmethod
    def compute_mean_absolute_error(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute the Mean Absolute Error (MAE) for regression tasks.

        Args:
            y_true (ndarray): The ground truth (actual) values.
            y_pred (ndarray): The predicted values.

        Returns:
            float: The mean absolute error, representing the average absolute difference between the true and
            predicted values.

        """
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def compute_mean_squared_error(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute the Mean Squared Error (MSE) for regression tasks.

        Args:
            y_true (ndarray): The ground truth (actual) values.
            y_pred (ndarray): The predicted values.

        Returns:
            float: The mean squared error, representing the average of the squared differences between the true and
            predicted values.

        """
        return mean_squared_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def compute_r_square_score(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute the R-squared (R²) score for regression tasks.

        Args:
            y_true (ndarray): The ground truth (actual) values.
            y_pred (ndarray): The predicted values.

        Returns:
            float: The R² score, representing the proportion of variance in the dependent variable that is predictable
            from the independent variables.

        """
        return r2_score(y_true=y_true, y_pred=y_pred)

    def compute_metrics(
        self, y_true: ndarray, y_pred: ndarray, y_proba: Optional[ndarray] = None
    ) -> RegressionMetricResult:
        """Compute all relevant metrics for a given task and return them in a structured datamodel.

        Args:
            y_true (ndarray): The ground truth (actual) values.
            y_pred (ndarray): The predicted values.
            y_proba (Optional[ndarray]): The predicted probabilities for the positive class (classification only).

        Returns:
            MetricsResults: A datamodel containing all computed metrics.

        """
        del y_proba  # Argument is not used by Regression.
        mae = RegressionMetricHandler.compute_mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse = RegressionMetricHandler.compute_mean_squared_error(y_true=y_true, y_pred=y_pred)
        r_square = RegressionMetricHandler.compute_r_square_score(y_true=y_true, y_pred=y_pred)
        return RegressionMetricResult(mean_absolute_error=mae, mean_squared_error=mse, r_square=r_square)
