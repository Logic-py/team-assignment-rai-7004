# Example CLI usage:

```shell
python src/cli.py --data_path=src/data/housing.csv --features="OverallQual" "GrLivArea" "GarageCars" "GarageArea" "TotalBsmtSF" --target_column=SalePrice --algorithm=linear_regression --random_state=42 --num_folds=5 --out_file=whatever.pkl
```

# Example Jupyter usage:
```python
from src.ml_pipeline_handler.algorithm import AlgorithmType
from src.ml_pipeline_handler.pipeline.base_config import PipelineConfig
from src.ml_pipeline_handler.pipeline.pipeline_factory import PipelineFactory

data_path: str = "src/data/housing.csv"
features: list[str] = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF"]
target_column: str = "SalePrice"
algorithm = AlgorithmType.LINEAR_REGRESSION
out_file: str = "example out file.pkl"
random_state: int = 42
num_folds: int = 5

pipeline_config = PipelineConfig(
    data_path=data_path,
    features=features,
    target_column=target_column,
    algorithm=algorithm,
    out_file=out_file,
    random_state=random_state,
    num_folds=num_folds,
)

pipeline = PipelineFactory.build_pipeline(config=pipeline_config)

prediction = pipeline.predict()

pipeline.compute_metrics(prediction=prediction)
```