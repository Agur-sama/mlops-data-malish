"""Воспроизводимое сравнение шести моделей Wine в MLflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
EXPERIMENT_NAME = "miit-mlops-practice-4-malish"
REQUIRED_METRICS = {"accuracy", "precision_weighted", "recall_weighted", "f1_weighted"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_wine_csv(dataset_id: str | None, local_path: Path | None) -> tuple[Path, str]:
    """Скачать Wine по ClearML Dataset ID или использовать локальный preflight-файл."""

    if dataset_id:
        from clearml import Dataset

        dataset_root = Path(Dataset.get(dataset_id=dataset_id).get_local_copy())
        candidates = list(dataset_root.rglob("wine.csv"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one wine.csv in ClearML Dataset, found {len(candidates)}"
            )
        return candidates[0], dataset_id
    if local_path:
        if not local_path.is_file():
            raise FileNotFoundError(f"Wine CSV not found: {local_path}")
        return local_path, f"local-preflight:{local_path.name}"
    raise ValueError("Pass --dataset-id for submission or --wine-csv for local preflight")


def load_verified_data(path: Path, expected_sha256: str) -> tuple[pd.DataFrame, pd.Series]:
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"wine.csv SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    frame = pd.read_csv(path)
    if "target" not in frame.columns:
        raise ValueError("wine.csv must contain the target column")
    if frame.isna().any().any():
        raise ValueError("wine.csv contains missing values")
    return frame.drop(columns="target"), frame["target"]


def make_split(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Создать единый стратифицированный split 60/20/20."""

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def experiment_configurations() -> list[dict[str, Any]]:
    """Вернуть широкий диапазон сложности, чтобы сравнение было наглядным.

    Wine — небольшой и хорошо разделимый набор. Близкие стандартные параметры
    дают одинаковые предсказания на 36 validation-объектах. Поэтому поиск
    намеренно включает сильно регуляризованные и ограниченные конфигурации:
    так виден переход от недообучения к лучшей модели на неизменном split.
    """

    return [
        {"name": "logreg-c-0.0001", "family": "logistic_regression", "C": 0.0001},
        {"name": "logreg-c-0.001", "family": "logistic_regression", "C": 0.001},
        {"name": "logreg-c-0.002", "family": "logistic_regression", "C": 0.002},
        {
            "name": "rf-1-depth-1-leaf-10",
            "family": "random_forest",
            "n_estimators": 1,
            "max_depth": 1,
            "min_samples_leaf": 10,
        },
        {
            "name": "rf-3-depth-2-leaf-5",
            "family": "random_forest",
            "n_estimators": 3,
            "max_depth": 2,
            "min_samples_leaf": 5,
        },
        {
            "name": "rf-10-depth-3-leaf-3",
            "family": "random_forest",
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 3,
        },
    ]


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    if config["family"] == "logistic_regression":
        classifier = LogisticRegression(
            C=config["C"],
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    classifier = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline([("classifier", classifier)])


def calculate_metrics(target: pd.Series, prediction: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision_weighted": float(
            precision_score(target, prediction, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(recall_score(target, prediction, average="weighted")),
        "f1_weighted": float(f1_score(target, prediction, average="weighted")),
    }


def write_split_manifest(output_dir: Path, splits: dict[str, pd.DataFrame]) -> Path:
    rows = []
    for split_name, frame in splits.items():
        rows.extend({"row_index": int(index), "split": split_name} for index in frame.index)
    manifest_path = output_dir / "split_manifest.csv"
    pd.DataFrame(rows).sort_values("row_index").to_csv(manifest_path, index=False)
    return manifest_path


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wine_path, dataset_reference = resolve_wine_csv(args.dataset_id, args.wine_csv)
    features, target = load_verified_data(wine_path, args.expected_sha256)
    x_train, x_validation, x_test, y_train, y_validation, y_test = make_split(features, target)
    split_manifest = write_split_manifest(
        output_dir,
        {"train": x_train, "validation": x_validation, "test": x_test},
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.set_experiment(args.experiment_name)
    invocation_id = uuid.uuid4().hex
    run_ids = []

    for config in experiment_configurations():
        pipeline = build_pipeline(config)
        with mlflow.start_run(
            run_name=config["name"], experiment_id=experiment.experiment_id
        ) as run:
            mlflow.set_tags(
                {
                    "model_family": config["family"],
                    "dataset_id": dataset_reference,
                    "dataset_sha256": args.expected_sha256,
                    "invocation_id": invocation_id,
                }
            )
            parameters = {
                **config,
                "random_state": RANDOM_STATE,
                "solver": "lbfgs" if config["family"] == "logistic_regression" else "n/a",
                "max_iter": 2000 if config["family"] == "logistic_regression" else "n/a",
            }
            mlflow.log_params(parameters)
            started = time.perf_counter()
            pipeline.fit(x_train, y_train)
            training_seconds = time.perf_counter() - started
            validation_prediction = pipeline.predict(x_validation)
            metrics = calculate_metrics(y_validation, validation_prediction)
            mlflow.log_metrics({**metrics, "training_seconds": training_seconds})
            signature = infer_signature(x_train, pipeline.predict(x_train))
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=x_train.head(5),
            )
            mlflow.log_artifact(split_manifest, artifact_path="data")
            run_ids.append(run.info.run_id)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.invocation_id = '{invocation_id}'",
    ).sort_values(
        ["metrics.f1_weighted", "metrics.accuracy", "tags.mlflow.runName"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if len(runs) != 6:
        raise RuntimeError(f"Expected 6 MLflow runs for this invocation, found {len(runs)}")

    best_run_id = str(runs.iloc[0]["run_id"])
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    test_prediction = best_model.predict(x_test)
    test_metrics = {
        f"test_{key}": value for key, value in calculate_metrics(y_test, test_prediction).items()
    }
    original_prediction = (
        build_pipeline(
            next(
                config
                for config in experiment_configurations()
                if config["name"] == runs.iloc[0]["tags.mlflow.runName"]
            )
        )
        .fit(x_train, y_train)
        .predict(x_test)
    )
    prediction_match = bool((original_prediction == test_prediction).all())

    comparison_columns = [
        "run_id",
        "tags.model_family",
        "tags.mlflow.runName",
        "params.C",
        "params.n_estimators",
        "params.max_depth",
        "params.min_samples_leaf",
        "metrics.f1_weighted",
        "metrics.training_seconds",
    ]
    comparison = runs[comparison_columns].copy()
    comparison.columns = [
        "run_id",
        "model_family",
        "run_name",
        "C",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "validation_f1",
        "training_seconds",
    ]
    comparison_path = output_dir / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    best_run = mlflow.get_run(best_run_id)
    verification = {
        "best_run_id": best_run_id,
        "dataset_id": dataset_reference,
        "dataset_sha256": args.expected_sha256,
        "signature_present": best_run.data.tags.get("mlflow.log-model.history") is not None,
        "predictions_match": prediction_match,
        "required_parameters_present": all(
            key in best_run.data.params for key in ("family", "name", "random_state")
        ),
        "required_validation_metrics_present": REQUIRED_METRICS.issubset(best_run.data.metrics),
        "input_columns": list(x_test.columns),
    }
    if not all(
        [
            verification["signature_present"],
            verification["predictions_match"],
            verification["required_parameters_present"],
            verification["required_validation_metrics_present"],
        ]
    ):
        raise RuntimeError(f"Best-run verification failed: {verification}")

    verification_path = output_dir / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    report_path = output_dir / "classification_report.json"
    report_path.write_text(
        json.dumps(classification_report(y_test, test_prediction, output_dict=True), indent=2),
        encoding="utf-8",
    )
    confusion_path = output_dir / "confusion_matrix.csv"
    pd.DataFrame(confusion_matrix(y_test, test_prediction)).to_csv(confusion_path, index=False)
    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics(test_metrics)
        for artifact in (verification_path, comparison_path, report_path, confusion_path):
            mlflow.log_artifact(artifact, artifact_path="verification")

    summary = {
        "experiment_id": experiment.experiment_id,
        "experiment_name": args.experiment_name,
        "tracking_uri": mlflow.get_tracking_uri(),
        "invocation_id": invocation_id,
        "dataset_id": dataset_reference,
        "dataset_sha256": args.expected_sha256,
        "best_run_id": best_run_id,
        "best_run_name": runs.iloc[0]["tags.mlflow.runName"],
        "validation_f1": float(runs.iloc[0]["metrics.f1_weighted"]),
        **test_metrics,
        "run_ids": run_ids,
        "model_path": model_path.name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"experiment_id={experiment.experiment_id}")
    print(f"best_run_id={best_run_id}")
    print(f"test_f1={test_metrics['test_f1_weighted']:.6f}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-id", help="ClearML Dataset ID из практики 3")
    source.add_argument("--wine-csv", type=Path, help="Только локальная предварительная проверка")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    )
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())
