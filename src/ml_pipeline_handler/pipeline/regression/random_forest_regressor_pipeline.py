"""Random Forest Regressor Module."""

from typing import Optional

from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor

from src.ml_pipeline_handler.metric.base_result import BaseMetricResult
from src.ml_pipeline_handler.metric.metric_factory import MetricFactory
from src.ml_pipeline_handler.metric.model_type import ModelType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.base_pipeline import BasePipeline


class RandomForestRegressorPipeline(BasePipeline):
    """Random Forest Regressor Pipeline Class."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the Random Forest Regressor Pipeline.

        Args:
            config: PipelineConfig, contains configuration information for a pipeline.

        """
        super().__init__(config=config)
        self.model = RandomForestRegressor(random_state=config.random_state)
        self.metric_handler = MetricFactory.get_metrics_handler(model_type=ModelType.REGRESSION)
        self.pre_processor = self.get_pre_processor()

    def predict(self) -> tuple[ndarray, Optional[ndarray]]:
        """Predict the target values based on the features.

        Returns:
            ndarray, of the prediction results.

        """
        features, target = self.load_data_set()
        self.x_train, self.x_test, self.y_train, self.y_test = self.create_training_set(
            features=features, target=target, test_size=0.3
        )

        x_train_to_use, x_test_to_use = self.pre_process_data(x_train=self.x_train, x_test=self.x_test)

        self.model.fit(X=x_train_to_use, y=self.y_train)
        return self.model.predict(X=x_test_to_use), None

    def compute_metrics(self, prediction: ndarray, probability: Optional[ndarray] = None) -> BaseMetricResult:
        """Compute the metrics of the given model.

        Args:
            prediction: ndarray, the prediction of the model.
            probability: Optional[ndarray], used for probability in classification models.

        Returns:
            BaseMetricResult, containing metric information.

        """
        return self.metric_handler.compute_metrics(y_true=self.y_test, y_pred=prediction, y_proba=probability)