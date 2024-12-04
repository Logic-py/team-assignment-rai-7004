"""Model Type Module."""

from enum import Enum


class ModelType(str, Enum):
    """Machine Learning Model type enum."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
