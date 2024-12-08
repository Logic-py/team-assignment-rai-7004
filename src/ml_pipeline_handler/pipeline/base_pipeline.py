"""Base Pipeline module."""

from abc import ABC, abstractmethod

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..io.loader import load_data
from ..metric.base_result import BaseMetricResult
from ..pipeline.base_config import PipelineConfig


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

    def get_pre_processor(self) -> ColumnTransformer:
        """Assemble and return a ColumnTransformer based on the pipeline config.

        Returns:
            ColumnTransformer to transform the data.

        """
        scaler_standard = Pipeline(steps=[("scaler", StandardScaler())])
        scaler_robust = Pipeline(steps=[("scaler", RobustScaler())])
        scaler_minmax = Pipeline(steps=[("scaler", MinMaxScaler())])

        return ColumnTransformer(
            transformers=[
                ("scaler_standard", scaler_standard, self.config.scale_standard),
                ("scaler_robust", scaler_robust, self.config.scale_robust),
                ("scaler_minmax", scaler_minmax, self.config.scale_minmax),
            ]
        )

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
