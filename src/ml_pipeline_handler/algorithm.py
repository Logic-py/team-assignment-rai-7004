"""Algorithm Type Module."""

from enum import Enum


class AlgorithmType(str, Enum):
    """Algorithm Type."""

    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
