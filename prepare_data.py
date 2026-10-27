from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_OPTIONS = {
    "index": False,
    "lineterminator": "\n",
    "float_format": "%.10g",
}


def sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, deterministically ordered copy of a dataset."""
    return frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_frame(frame).to_csv(path, **CSV_OPTIONS)


def prepare_base() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in DATA_DIR.glob("*.csv"):
        csv_path.unlink()
    save_frame(load_iris(as_frame=True).frame, DATA_DIR / "iris.csv")


def prepare_wine() -> None:
    iris_path = DATA_DIR / "iris.csv"
    if not iris_path.exists():
        raise FileNotFoundError("Сначала создайте базовую версию: python prepare_data.py base")
    save_frame(load_wine(as_frame=True).frame, DATA_DIR / "wine.csv")


def prepare_iris_features() -> None:
    iris = load_iris(as_frame=True).frame.copy()
    iris["sepal_area"] = iris["sepal length (cm)"] * iris["sepal width (cm)"]
    iris["petal_area"] = iris["petal length (cm)"] * iris["petal width (cm)"]
    save_frame(iris, DATA_DIR / "iris.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание версий Iris и Wine")
    parser.add_argument("mode", choices=("base", "wine", "iris_features"))
    args = parser.parse_args()

    actions = {
        "base": prepare_base,
        "wine": prepare_wine,
        "iris_features": prepare_iris_features,
    }
    actions[args.mode]()


if __name__ == "__main__":
    main()
