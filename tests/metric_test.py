"""Metrics test module."""

import numpy as np
import pytest

from src.ml_pipeline_handler.metric.base_metric_handler import BaseMetricsHandler
from src.ml_pipeline_handler.metric.classification.handler import ClassificationMetricHandler
from src.ml_pipeline_handler.metric.classification.result import ClassificationMetricResult
from src.ml_pipeline_handler.metric.metric_factory import MetricFactory
from src.ml_pipeline_handler.metric.model_type import ModelType
from src.ml_pipeline_handler.metric.regression.handler import RegressionMetricHandler
from src.ml_pipeline_handler.metric.regression.result import RegressionMetricResult


# region Factory Tests


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


# endregion

# region Classification Tests
classification_y_true = np.array([0, 1, 1, 0, 1, 0])
classification_y_pred = np.array([0, 1, 1, 0, 0, 1])
classification_y_proba = np.array(
    [
        [0.2, 0.8, 0.9, 0.3, 0.4, 0.6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
    ]
)


def test_compute_accuracy_score() -> None:
    """ "Test computation of accuracy score."

    Returns:
        None

    """
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    accuracy = handler.compute_accuracy_score(y_true=classification_y_true, y_pred=classification_y_pred)
    expected_accuracy = 4 / 6
    assert accuracy == pytest.approx(expected=expected_accuracy), f"Expected {expected_accuracy}, got {accuracy}"


def test_compute_area_under_curve_score() -> None:
    """Test computation of ROC AUC score.

    Returns:
        None
    """
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    roc_auc = handler.compute_area_under_curve_score(y_true=classification_y_true, y_proba=classification_y_proba)
    expected_roc_auc = 0.67
    assert roc_auc == pytest.approx(expected=expected_roc_auc, rel=1e-2), f"Expected {expected_roc_auc}, got {roc_auc}"


def test_compute_classification_metrics() -> None:
    """Test computation of all metrics.

    Returns:
        None
    """
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    result = handler.compute_metrics(
        y_true=classification_y_true,
        y_pred=classification_y_pred,
        y_proba=classification_y_proba,
    )
    assert isinstance(
        result, ClassificationMetricResult
    ), "Expected result to be an instance of ClassificationMetricResult"
    assert result.accuracy == pytest.approx(expected=4 / 6), f"Expected accuracy to be {4 / 6}, got {result.accuracy}"
    assert result.roc_auc == pytest.approx(
        expected=0.67, rel=1e-2
    ), f"Expected ROC AUC to be approximately 0.67, got {result.roc_auc}"


def test_compute_metrics_missing_y_proba() -> None:
    """Test that compute_metrics raises ValueError if y_proba is missing.

    Returns:
        None
    """
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    with pytest.raises(ValueError):
        handler.compute_metrics(y_true=classification_y_true, y_pred=classification_y_pred)


# endregion

# region Regression Tests
regression_y_true = np.array([3.0, -0.5, 2.0, 7.0])
regression_y_pred = np.array([2.5, 0.0, 2.0, 8.0])


def test_compute_mean_absolute_error() -> None:
    """Test computation of Mean Absolute Error (MAE).

    Returns:
        None
    """
    handler = RegressionMetricHandler(model_type=ModelType.REGRESSION)
    mae = handler.compute_mean_absolute_error(y_true=regression_y_true, y_pred=regression_y_pred)
    expected_mae = 0.5  # Example based on test data
    assert mae == pytest.approx(expected=expected_mae), f"Expected {expected_mae}, got {mae}"


def test_compute_mean_squared_error() -> None:
    """Test computation of Mean Squared Error (MSE).

    Returns:
        None
    """
    handler = RegressionMetricHandler(model_type=ModelType.REGRESSION)
    mse = handler.compute_mean_squared_error(y_true=regression_y_true, y_pred=regression_y_pred)
    expected_mse = 0.375  # Example based on test data
    assert mse == pytest.approx(expected=expected_mse), f"Expected {expected_mse}, got {mse}"


def test_compute_r_square_score() -> None:
    """Test computation of R-squared (R²).

    Returns:
        None
    """
    handler = RegressionMetricHandler(model_type=ModelType.REGRESSION)
    r_square = handler.compute_r_square_score(y_true=regression_y_true, y_pred=regression_y_pred)
    expected_r_square = 0.948608137  # Example based on test data
    assert r_square == pytest.approx(
        expected=expected_r_square, rel=1e-5
    ), f"Expected {expected_r_square}, got {r_square}"


def test_compute_regression_metrics() -> None:
    """Test computation of all regression metrics.

    Returns:
        None
    """
    handler = RegressionMetricHandler(model_type=ModelType.REGRESSION)
    result = handler.compute_metrics(y_true=regression_y_true, y_pred=regression_y_pred)
    assert isinstance(result, RegressionMetricResult), "Expected result to be an instance of RegressionMetricResult"
    assert result.mean_absolute_error == pytest.approx(
        expected=0.5
    ), f"Expected MAE to be 0.5, got {result.mean_absolute_error}"
    assert result.mean_squared_error == pytest.approx(
        expected=0.375
    ), f"Expected MSE to be 0.375, got {result.mean_squared_error}"
    assert result.r_square == pytest.approx(
        expected=0.948608137, rel=1e-5
    ), f"Expected R² to be 0.948608137, got {result.r_square}"


def test_compute_metrics_ignores_y_proba() -> None:
    """Test that compute_metrics ignores y_proba in regression context.

    Returns:
        None
    """
    handler = RegressionMetricHandler(model_type=ModelType.REGRESSION)
    y_proba_dummy = np.array([0.1, 0.2, 0.3, 0.4])  # Should be ignored
    result = handler.compute_metrics(y_true=regression_y_true, y_pred=regression_y_pred, y_proba=y_proba_dummy)
    assert result.mean_absolute_error == pytest.approx(expected=0.5), "y_proba should not affect MAE computation"
    assert result.mean_squared_error == pytest.approx(expected=0.375), "y_proba should not affect MSE computation"
    assert result.r_square == pytest.approx(expected=0.948608137, rel=1e-5), "y_proba should not affect R² computation"


# endregion
