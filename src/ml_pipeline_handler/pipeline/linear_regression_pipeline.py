"""Linear Regression Module."""

from typing import Optional

from numpy import ndarray
from pandas import Series
from sklearn.linear_model import LinearRegression

from ..metric.base_result import BaseMetricResult
from ..metric.metric_factory import MetricFactory
from ..metric.model_type import ModelType
from ..pipeline.base_config import PipelineConfig
from ..pipeline.base_pipeline import BasePipeline


class LinearRegressionPipeline(BasePipeline):
    """Linear Regression Pipeline Class."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the Linear Regression Pipeline.

        Args:
            config: PipelineConfig, contains configuration information for a pipeline.

        """
        super().__init__(config=config)
        self.model = LinearRegression()
        self.metric_handler = MetricFactory.get_metrics_handler(model_type=ModelType.REGRESSION)

        self.x_train: Optional[ndarray] = None
        self.x_test: Optional[ndarray] = None
        self.y_train: Optional[Series] = None
        self.y_test: Optional[Series] = None

    def predict(self) -> ndarray:
        """Predict the target values based on the features.

        Returns:
            ndarray, of the prediction results.

        """
        features, target = self.load_data_set()
        self.x_train, self.x_test, self.y_train, self.y_test = self.create_training_set(
            features=features, target=target, test_size=0.3
        )

        model = LinearRegression()
        model.fit(X=self.x_train, y=self.y_train)
        return model.predict(X=self.x_test)

    def compute_metrics(self, prediction: ndarray) -> BaseMetricResult:
        """Compute the metrics of the given model.

        Args:
            prediction: ndarray, the prediction of the model.

        Returns:
            BaseMetricResult, containing metric information.

        """
        return self.metric_handler.compute_metrics(y_true=self.y_test, y_pred=prediction)
