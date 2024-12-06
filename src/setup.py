"""Setup module."""

from setuptools import find_packages, setup

setup(
    name="ml_pipeline_builder",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["loguru", "scikit-learn ", "numpy", "pandas"],
    entry_points={
        "console_scripts": [
            "ml_pipeline-cli = cli:cli",
        ],
    },
    package_dir={"": "src"},
)
