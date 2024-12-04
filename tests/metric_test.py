"""Metrics test module."""

import numpy as np
import pytest

from src.metric.base_metric_handler import BaseMetricsHandler
from src.metric.classification.handler import ClassificationMetricHandler
from src.metric.classification.result import ClassificationMetricResult
from src.metric.metric_factory import MetricFactory
from src.metric.model_type import ModelType
from src.metric.regression.handler import RegressionMetricHandler


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
classification_y_proba = np.array([0.2, 0.8, 0.9, 0.3, 0.4, 0.6])


def test_compute_accuracy_score():
    """Test computation of accuracy score."""
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    accuracy = handler.compute_accuracy_score(y_true=classification_y_true, y_pred=classification_y_pred)
    expected_accuracy = 4 / 6
    assert accuracy == pytest.approx(expected=expected_accuracy), f"Expected {expected_accuracy}, got {accuracy}"


def test_compute_area_under_curve_score():
    """Test computation of ROC AUC score."""
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    roc_auc = handler.compute_area_under_curve_score(y_true=classification_y_true, y_proba=classification_y_proba)
    expected_roc_auc = 0.89
    assert roc_auc == pytest.approx(expected=expected_roc_auc, rel=1e-2), f"Expected {expected_roc_auc}, got {roc_auc}"


def test_compute_metrics():
    """Test computation of all metrics."""
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    result = handler.compute_metrics(
        y_true=classification_y_true, y_pred=classification_y_pred, y_proba=classification_y_proba
    )
    assert isinstance(
        result, ClassificationMetricResult
    ), "Expected result to be an instance of ClassificationMetricResult"
    assert result.accuracy == pytest.approx(expected=4 / 6), f"Expected accuracy to be {4 / 6}, got {result.accuracy}"
    assert result.roc_auc == pytest.approx(
        expected=0.89, rel=1e-2
    ), f"Expected ROC AUC to be approximately 0.75, got {result.roc_auc}"


def test_compute_metrics_missing_y_proba():
    """Test that compute_metrics raises ValueError if y_proba is missing."""
    handler = ClassificationMetricHandler(model_type=ModelType.CLASSIFICATION)
    with pytest.raises(ValueError):
        handler.compute_metrics(y_true=classification_y_true, y_pred=classification_y_pred)


# endregion
