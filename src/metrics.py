"""Metrics module."""

from numpy import ndarray
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


class Metrics:
    """The purpose of the metrics module is to expose metric computations for Machine Learning Models."""

    @staticmethod
    def compute_accuracy_score(y_true: ndarray, y_pred: ndarray) -> float:
        """Compute the accuracy score for a classification task.

        Args:
            y_true (ndarray): The ground truth (actual) labels.
            y_pred (ndarray): The predicted labels.

        Returns:
            float: The accuracy score, representing the proportion of correct predictions out of the total number of
            predictions.

        """
        return accuracy_score(y_true=y_true, y_pred=y_pred)

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
    def compute_area_under_curve_score(y_true: ndarray, y_proba: ndarray) -> float:
        """Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

        Args:
            y_true (ndarray): The ground truth (actual) binary labels.
            y_proba (ndarray): The predicted probabilities for the positive class.

        Returns:
            float: The ROC AUC score, which measures the ability of the model to distinguish between the positive and
            negative classes.

        """
        return roc_auc_score(y_true=y_true, y_score=y_proba)

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
