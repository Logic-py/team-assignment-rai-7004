"""Base Pipeline Config Module."""

from dataclasses import dataclass

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
