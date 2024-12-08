"""Base Pipeline module."""

from abc import ABC, abstractmethod

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from src.ml_pipeline_handler.io.loader import load_data
from src.ml_pipeline_handler.metric.base_result import BaseMetricResult
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig


class BasePipeline(ABC):
    """Base Pipeline Class."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize any Pipeline.

        Args:
            config: PipelineConfig, contains configuration information for a pipeline.

        """
        self.config = config

    def load_data_set(self) -> tuple[DataFrame, Series]:
        """Call load_data function from io module.

        Returns:
            tuple[DataFrame, Series] of the features and target.

        """
        return load_data(data_path=self.config.data_path, target_column=self.config.target_column)

    def create_training_set(
        self, features: DataFrame, target: Series, test_size: float
    ) -> tuple[ndarray, ndarray, Series, Series]:
        """Create training set based on features and target series and test size.

        Args:
            features: DataFrame containing all features.
            target: Series containing the target.
            test_size: float of the test size [0.0 - 1.0]

        Returns:
            tuple[ndarray, ndarray, Series, Series]: x_train, x_test, y_train, y_test.

        """
        features = features[self.config.features]
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=self.config.random_state
        )
        return x_train, x_test, y_train, y_test

    @abstractmethod
    def predict(self) -> ndarray:
        """Predict the target values based on the features.

        Returns:
            ndarray, of the prediction results.

        """

    @abstractmethod
    def compute_metrics(self, prediction: ndarray) -> BaseMetricResult:
        """Compute the metrics of the given model.

        Args:
            prediction: ndarray, the prediction of the model.

        Returns:
            BaseMetricResult, containing metric information.

        """
