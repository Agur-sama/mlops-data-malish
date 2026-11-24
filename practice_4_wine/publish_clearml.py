"""Публикация лучшего MLflow-запуска как ClearML Task и OutputModel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from clearml import OutputModel, Task


def validate_official_summary(summary: dict, verification: dict) -> None:
    """Не дать опубликовать локальный preflight вместо официального запуска."""

    dataset_id = str(summary.get("dataset_id", ""))
    if dataset_id.startswith("local-preflight:") or not re.fullmatch(r"[0-9a-f]{32}", dataset_id):
        raise ValueError("ClearML publication requires an official 32-character Dataset ID")
    if verification.get("dataset_id") != dataset_id:
        raise ValueError("Summary and verification refer to different datasets")
    if verification.get("dataset_sha256") != summary.get("dataset_sha256"):
        raise ValueError("Summary and verification contain different dataset SHA-256 values")


def publish(artifacts_dir: Path, project_name: str) -> dict[str, str]:
    summary = json.loads((artifacts_dir / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((artifacts_dir / "verification.json").read_text(encoding="utf-8"))
    validate_official_summary(summary, verification)
    comparison = pd.read_csv(artifacts_dir / "comparison.csv")

    task = Task.init(
        project_name=project_name,
        task_name="wine-best-model",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=True,
    )
    best_row = comparison.loc[comparison["run_id"] == summary["best_run_id"]].iloc[0]
    task.connect(best_row.to_dict(), name="best_configuration")
    task.connect(
        {
            "mlflow_run_id": summary["best_run_id"],
            "clearml_dataset_id": summary["dataset_id"],
            "dataset_sha256": summary["dataset_sha256"],
        },
        name="provenance",
    )
    task.get_logger().report_table(
        title="MLflow runs",
        series="validation comparison",
        iteration=0,
        table_plot=comparison,
    )
    task.get_logger().report_scalar(
        "metrics", "validation_f1", float(summary["validation_f1"]), iteration=0
    )
    task.get_logger().report_scalar(
        "metrics", "test_f1", float(summary["test_f1_weighted"]), iteration=0
    )
    task.upload_artifact("verification", verification, wait_on_upload=True)
    task.upload_artifact("comparison", comparison, wait_on_upload=True)

    labels = {name: index for index, name in enumerate(["class_0", "class_1", "class_2"])}
    model = OutputModel(
        task=task,
        name="wine-best-model",
        framework="scikit-learn",
        label_enumeration=labels,
        config_dict=best_row.to_dict(),
    )
    model.set_metadata("mlflow_run_id", summary["best_run_id"])
    model.set_metadata("clearml_dataset_id", summary["dataset_id"])
    model.set_metadata("dataset_sha256", summary["dataset_sha256"])
    uploaded_uri = model.update_weights(
        weights_filename=str(artifacts_dir / "best_model.joblib"),
        upload_uri=task.output_uri,
        auto_delete_file=False,
        async_enable=False,
    )
    if not uploaded_uri.startswith(("https://", "http://", "s3://", "gs://", "azure://")):
        raise RuntimeError(f"ClearML model was not uploaded to remote storage: {uploaded_uri}")
    model.publish()
    OutputModel.wait_for_uploads(timeout=300)
    result = {
        "clearml_task_id": task.id,
        "clearml_model_id": model.id,
        "clearml_task_url": task.get_output_log_web_page(),
        "clearml_model_url": model.url,
        "model_storage_uri": uploaded_uri,
    }
    task.upload_artifact("published_ids", result, wait_on_upload=True)
    task.close()
    (artifacts_dir / "clearml_ids.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--project", default="MIIT/MLOps/practice_4/malish")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    publish(arguments.artifacts_dir.resolve(), arguments.project)
