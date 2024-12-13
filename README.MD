# README
A Configurable Machine Learning Pipeline

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Features](#features)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Pipeline Details](#pipeline-details)
7. [Testing](#testing)
8. [License](#license)

---

## Introduction
This project provides a configurable and modular machine learning pipeline that supports various preprocessing options and machine learning algorithms. It is designed to streamline the process of data ingestion, preprocessing, model training, and evaluation.

## Project Overview
### Code Structure
The repository is organized using a modular approach, enabling ease of maintenance, readability, and extensibility. The implementation leverages the **Factory Design Pattern** to create machine learning pipelines dynamically, ensuring flexibility and reusability across different algorithms and configurations. Additionally, a dedicated metrics module facilitates model evaluation by calculating various performance metrics.

The project structure is as follows:
```
project/
├── src/                                    # Main source directory
│   ├── cli.py                                  # Command-line interface for pipeline execution
│   ├── data/                                   # Example CSV file for testing purposes
│   ├── ml_pipeline_handler/                    # Core machine learning pipeline components
│   │   ├── algorithm.py                            # Definition of supported algorithms
│   │   ├── io/                                     # Input/Output utilities
│   │   ├── metric/                                 # Metrics module for model evaluation
│   │   ├── pipeline/                               # Pipeline implementation
│   │   │   ├── base_config.py                          # Pipeline configuration dataclass
│   │   │   ├── base_pipeline.py                        # Base pipeline logic
│   │   │   ├── pipeline_factory.py                     # Factory class for pipeline creation
│   │   │   ├── classification/                         # Classification pipeline implementation
│   │   │   └── regression/                             # Regression pipeline implementation
│   └── __init__.py                             # Marks the src folder as a Python package
├── tests/                                  # Test suite for the project
│   └── decision_tree_pipeline_test.py          # Unit tests for pipeline functionality
├── main.py                                 # Entry point for executing the pipeline
├── requirements.txt                        # Python dependencies
├── pyproject.toml                          # Project configuration for Poetry
├── README.md                               # Project documentation
├── LICENSE                                 # Project license
└── .gitignore                              # Files to exclude from version control
```

### Factory Design Pattern
The project uses the **Factory Design Pattern** to dynamically generate pipelines based on user-defined configurations. The `PipelineFactory` class in `pipeline_factory.py` accepts a configuration object (e.g., `PipelineConfig`) and returns the appropriate pipeline instance (e.g., Classification or Regression). This design ensures:
- Flexibility to add new pipeline types without modifying the existing codebase.
- Cleaner, modular implementation for pipeline instantiation.

### Metrics Module
The metrics module located in `ml_pipeline_handler/metric` provides tools for evaluating model performance. It supports:
- Standard metrics like accuracy, precision, recall, and F1-score for classification tasks.
- Metrics for regression such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
- Seamless integration with the pipeline for automatic evaluation during cross-validation.

## Features
- **Input Flexibility**: Supports CSV-based datasets.
- **Feature Scaling**: Configurable options for Standard, Robust, and MinMax scalers.
- **ML Algorithms**: Choose from various predefined algorithms.
- **Cross-Validation**: Configurable k-fold cross-validation for robust model evaluation.
- **Logging**: Detailed logging using `loguru` for improved traceability.
- **Pipeline Outputs**: Saves trained pipelines/models for later use.

## Configuration
### Parameters
| Parameter         | Description                         | Required |
|-------------------|-------------------------------------|----------|
| `data_path`       | Path to the CSV dataset             | Yes      |
| `features`        | List of feature column names        | Yes      |
| `target_column`   | Name of the target column           | Yes      |
| `scale_standard`  | Features for standard scaling       | No       |
| `scale_robust`    | Features for robust scaling         | No       |
| `scale_minmax`    | Features for MinMax scaling         | No       |
| `algorithm`       | Machine learning algorithm to use   | Yes      |
| `out_file`        | Output file to save the pipeline    | Yes      |
| `random_state`    | Random seed for reproducibility     | No       |
| `num_folds`       | Number of cross-validation folds    | No       |

### Example Configuration
```python
PipelineConfig(
    data_path="data.csv",
    features=["feature1", "feature2"],
    target_column="target",
    scale_standard=["feature1"],
    scale_minmax=["feature2"],
    algorithm=AlgorithmType.CLASSIFICATION,
    out_file="model_pipeline.pkl",
    random_state=42,
    num_folds=5
)
```

## Usage
### Example CLI usage:

```shell
python src/cli.py --data_path=src/data/housing.csv --features="OverallQual" "GrLivArea" "GarageCars" "GarageArea" "TotalBsmtSF" --target_column=SalePrice --algorithm=linear_regression --random_state=42 --num_folds=5 --out_file=whatever.pkl
```

### Example Jupyter usage:
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
## Pipeline Details
The pipeline includes the following stages:
1. **Data Loading**: Reads data from the specified CSV file.
2. **Preprocessing**:
   - Scaling using Standard, Robust, or MinMax scalers.
3. **Model Training**:
   - Supports configurable ML algorithms.
4. **Cross-Validation**:
   - k-fold cross-validation for evaluation.
5. **Model Saving**:
   - Saves the trained pipeline to the specified output file.

## Testing
To run tests:
```bash
pytest tests/
```