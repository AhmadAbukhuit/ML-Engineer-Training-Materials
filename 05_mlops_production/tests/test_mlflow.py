import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mlflow_tracker import MLflowTracker


def test_mlflow_tracker_logging_and_querying(tmp_path):
    tracking_dir = tmp_path / "mlruns"
    tracker = MLflowTracker(experiment_name="test_experiment", tracking_uri=tracking_dir)

    # Run 1: lower accuracy
    with tracker.run(run_name="baseline_run"):
        tracker.log_parameters({"learning_rate": 0.01, "batch_size": 32})
        tracker.log_metrics({"val_accuracy": 0.82, "val_loss": 0.35})

    # Run 2: higher accuracy
    with tracker.run(run_name="tuned_run"):
        tracker.log_parameters({"learning_rate": 0.001, "batch_size": 64})
        tracker.log_metrics({"val_accuracy": 0.94, "val_loss": 0.12})

        # Save and log artifact
        artifact_file = tmp_path / "summary.txt"
        artifact_file.write_text("Model converged successfully.")
        tracker.log_artifact(artifact_file)

    # Query best run by val_accuracy
    best_acc_run = tracker.get_best_run(metric_name="val_accuracy", ascending=False)
    assert best_acc_run is not None
    assert best_acc_run["metrics"]["val_accuracy"] == 0.94
    assert best_acc_run["params"]["batch_size"] == "64"

    # Query best run by val_loss (ascending=True)
    best_loss_run = tracker.get_best_run(metric_name="val_loss", ascending=True)
    assert best_loss_run is not None
    assert best_loss_run["metrics"]["val_loss"] == 0.12
