from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from clearml import Dataset
from sklearn.datasets import load_iris, load_wine

from prepare_data import save_frame


PROJECT = "MIIT/MLOps/practice_3/malish"
DATASET_NAME = "sklearn-tabular-malish"


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание трёх версий ClearML Dataset")
    parser.add_argument("--output", type=Path, default=Path("clearml_ids.json"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="practice3-clearml-") as temp_dir:
        temp = Path(temp_dir)

        base_file = temp / "base" / "iris.csv"
        wine_file = temp / "wine" / "wine.csv"
        features_file = temp / "iris_features" / "iris.csv"
        iris = load_iris(as_frame=True).frame
        iris_features = iris.copy()
        iris_features["sepal_area"] = (
            iris_features["sepal length (cm)"] * iris_features["sepal width (cm)"]
        )
        iris_features["petal_area"] = (
            iris_features["petal length (cm)"] * iris_features["petal width (cm)"]
        )
        save_frame(iris, base_file)
        save_frame(load_wine(as_frame=True).frame, wine_file)
        save_frame(iris_features, features_file)

        base = Dataset.create(
            dataset_project=PROJECT,
            dataset_name=DATASET_NAME,
            dataset_version="1.0.0",
            dataset_tags=["base"],
            description="Базовая версия с исходным sklearn Iris.",
        )
        base.add_files(path=base_file)
        base.upload()
        base.finalize()

        wine = Dataset.create(
            dataset_project=PROJECT,
            dataset_name=DATASET_NAME,
            dataset_version="1.1.0",
            parent_datasets=[base.id],
            dataset_tags=["wine"],
            description="Дочерняя версия: к Iris добавлен sklearn Wine.",
        )
        wine.add_files(path=wine_file)
        wine.upload()
        wine.finalize()

        features = Dataset.create(
            dataset_project=PROJECT,
            dataset_name=DATASET_NAME,
            dataset_version="2.0.0",
            parent_datasets=[base.id],
            dataset_tags=["iris-features"],
            description="Отдельная ветвь от базы: Iris с sepal_area и petal_area.",
        )
        features.add_files(path=features_file)
        features.upload()
        features.finalize()

    ids = {
        "BASE_ID": base.id,
        "WINE_ID": wine.id,
        "IRIS_FEATURES_ID": features.id,
    }
    args.output.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ids, indent=2))


if __name__ == "__main__":
    main()
