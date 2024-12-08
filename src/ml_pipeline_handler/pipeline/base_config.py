"""Base Pipeline Config Module."""

from dataclasses import dataclass, field

from src.ml_pipeline_handler.algorithm import AlgorithmType


@dataclass
class PipelineConfig:
    """Pipeline Config Dataclass."""

    data_path: str
    features: list[str]
    target_column: str
    algorithm: AlgorithmType
    out_file: str
    random_state: int
    num_folds: int
    scale_standard: list[str] = field(default_factory=list)
    scale_robust: list[str] = field(default_factory=list)
    scale_minmax: list[str] = field(default_factory=list)

    def has_pre_processing(self) -> bool:
        """Check whether the pipeline config contains pre_processing.

        Returns:
            Boolean representing the presence of preprocessing.

        """
        return any([self.scale_standard, self.scale_robust, self.scale_minmax])
