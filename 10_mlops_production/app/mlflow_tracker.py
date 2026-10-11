"""
Standardized MLflow Experiment Tracking and Model Lifecycle Interface.

This module provides:
1. MLflowTracker: Helper class encapsulating experiment creation, parameter logging,
   metric logging, tag tracking, and artifact logging.
2. Best Run Querying: Retrieves top-performing model candidate runs based on specified evaluation metrics.
"""

from collections.abc import Generator
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


class MLflowTracker:
    """
    Wrapper for MLflow experiment tracking ensuring standardized metadata capture.
    """

    def __init__(self, experiment_name: str, tracking_uri: str | Path | None = None):
        self.experiment_name = experiment_name
        if tracking_uri is not None:
            uri_str = str(tracking_uri)
            # If tracking_uri is a local folder path, convert to file URI
            if not uri_str.startswith(("http://", "https://", "file://", "sqlite://")):
                Path(uri_str).mkdir(parents=True, exist_ok=True)
                uri_str = Path(uri_str).resolve().as_uri()
            mlflow.set_tracking_uri(uri_str)

        self.experiment = mlflow.set_experiment(experiment_name)

    @contextmanager
    def run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> Generator[Any, None, None]:
        """
        Context manager for tracking an experiment run.
        Automatically finalizes run status on completion or error.
        """
        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            yield active_run

    def log_parameters(self, params: dict[str, Any]) -> None:
        """Logs hyperparameters for the current active run."""
        sanitized = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in params.items()}
        mlflow.log_params(sanitized)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs scalar evaluation metrics for the current active run."""
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Uploads an artifact (e.g., config, serialized weights, plot) to MLflow artifact storage."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def get_best_run(self, metric_name: str, ascending: bool = False) -> dict[str, Any] | None:
        """
        Queries all runs in the experiment and identifies the best run according to metric_name.

        Args:
            metric_name: Monitored metric name (e.g. 'accuracy', 'val_loss').
            ascending: Set to True for losses (min is best), False for accuracies (max is best).

        Returns:
            Dictionary containing run_id, metrics, params, and tags.
        """
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1,
        )

        if runs.empty:
            return None

        best_row = runs.iloc[0]
        run_id = str(best_row["run_id"])
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id).data

        return {
            "run_id": run_id,
            "metrics": dict(run_data.metrics),
            "params": dict(run_data.params),
            "tags": dict(run_data.tags),
        }
