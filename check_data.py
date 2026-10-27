from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_series_equal
from sklearn.datasets import load_iris, load_wine


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BASE_IRIS_COLUMNS = list(load_iris(as_frame=True).frame.columns)
WINE_COLUMNS = list(load_wine(as_frame=True).frame.columns)
FEATURE_COLUMNS = BASE_IRIS_COLUMNS + ["sepal_area", "petal_area"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_common(path: Path, expected_rows: int, expected_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ожидаемый файл отсутствует: {path.relative_to(ROOT)}")

    frame = pd.read_csv(path)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        raise ValueError(f"{path.name}: найден безымянный индексный столбец: {unnamed}")
    if len(frame) != expected_rows:
        raise ValueError(f"{path.name}: ожидалось {expected_rows} строк, получено {len(frame)}")
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{path.name}: неверный порядок или состав столбцов: {list(frame.columns)}"
        )
    return frame


def validate_iris(path: Path, with_features: bool) -> pd.DataFrame:
    columns = FEATURE_COLUMNS if with_features else BASE_IRIS_COLUMNS
    frame = validate_common(path, len(load_iris().data), columns)
    if with_features:
        assert_series_equal(
            frame["sepal_area"],
            frame["sepal length (cm)"] * frame["sepal width (cm)"],
            check_names=False,
            rtol=1e-9,
            atol=1e-9,
        )
        assert_series_equal(
            frame["petal_area"],
            frame["petal length (cm)"] * frame["petal width (cm)"],
            check_names=False,
            rtol=1e-9,
            atol=1e-9,
        )
    return frame


def describe(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns_count": len(frame.columns),
        "columns": list(frame.columns),
        "missing_values": int(frame.isna().sum().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка состава и содержимого данных")
    parser.add_argument("mode", choices=("base", "wine", "iris_features", "final"))
    args = parser.parse_args()

    expected_files = {
        "base": {"iris.csv"},
        "wine": {"iris.csv", "wine.csv"},
        "iris_features": {"iris.csv"},
        "final": {"iris.csv", "wine.csv"},
    }[args.mode]
    actual_files = {path.name for path in DATA_DIR.glob("*.csv")} if DATA_DIR.exists() else set()
    if actual_files != expected_files:
        raise ValueError(f"Ожидались файлы {sorted(expected_files)}, найдены {sorted(actual_files)}")

    iris_features = args.mode in {"iris_features", "final"}
    results = [describe(DATA_DIR / "iris.csv", validate_iris(DATA_DIR / "iris.csv", iris_features))]
    if "wine.csv" in expected_files:
        wine_path = DATA_DIR / "wine.csv"
        results.append(describe(wine_path, validate_common(wine_path, len(load_wine().data), WINE_COLUMNS)))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
