from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from clearml import Dataset

from prepare_data import DATA_DIR, prepare_base, prepare_iris_features, prepare_wine


PROJECT = "MIIT/MLOps/practice_3/malish"
DATASET_NAME = "sklearn-tabular-malish"


def finalized_dataset(**kwargs: object) -> Dataset:
    dataset = Dataset.create(**kwargs)
    dataset.upload()
    dataset.finalize()
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание трёх версий ClearML Dataset")
    parser.add_argument("--output", type=Path, default=Path("clearml_ids.json"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="practice3-clearml-") as temp_dir:
        temp = Path(temp_dir)

        prepare_base()
        base_file = temp / "base" / "iris.csv"
        base_file.parent.mkdir()
        shutil.copy2(DATA_DIR / "iris.csv", base_file)
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

        prepare_wine()
        wine = Dataset.create(
            dataset_project=PROJECT,
            dataset_name=DATASET_NAME,
            dataset_version="1.1.0",
            parent_datasets=[base.id],
            dataset_tags=["wine"],
            description="Дочерняя версия: к Iris добавлен sklearn Wine.",
        )
        wine.add_files(path=DATA_DIR / "wine.csv")
        wine.upload()
        wine.finalize()

        prepare_base()
        prepare_iris_features()
        features = Dataset.create(
            dataset_project=PROJECT,
            dataset_name=DATASET_NAME,
            dataset_version="2.0.0",
            parent_datasets=[base.id],
            dataset_tags=["iris-features"],
            description="Отдельная ветвь от базы: Iris с sepal_area и petal_area.",
        )
        features.add_files(path=DATA_DIR / "iris.csv")
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
