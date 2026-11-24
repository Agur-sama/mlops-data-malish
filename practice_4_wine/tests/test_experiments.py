import hashlib

import pandas as pd
import pytest
from sklearn.datasets import load_wine

from publish_clearml import validate_official_summary
from run_experiments import (
    build_pipeline,
    calculate_metrics,
    experiment_configurations,
    load_verified_data,
    make_split,
    sha256_file,
)


@pytest.fixture
def wine_csv(tmp_path):
    dataset = load_wine(as_frame=True)
    path = tmp_path / "wine.csv"
    dataset.frame.to_csv(path, index=False)
    return path


def test_sha256_and_verified_load(wine_csv):
    expected = hashlib.sha256(wine_csv.read_bytes()).hexdigest()
    features, target = load_verified_data(wine_csv, expected)
    assert len(features) == len(target) == 178
    assert "target" not in features.columns
    assert sha256_file(wine_csv) == expected


def test_hash_mismatch_fails_before_training(wine_csv):
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_verified_data(wine_csv, "0" * 64)


def test_split_is_stratified_reproducible_and_60_20_20(wine_csv):
    expected = sha256_file(wine_csv)
    features, target = load_verified_data(wine_csv, expected)
    first = make_split(features, target)
    second = make_split(features, target)
    assert [len(part) for part in first[:3]] == [106, 36, 36]
    assert [list(part.index) for part in first[:3]] == [list(part.index) for part in second[:3]]
    for labels in first[3:]:
        assert set(labels.unique()) == {0, 1, 2}


def test_six_meaningfully_different_configurations():
    configs = experiment_configurations()
    assert len(configs) == 6
    assert sum(config["family"] == "logistic_regression" for config in configs) == 3
    assert sum(config["family"] == "random_forest" for config in configs) == 3
    assert len({config["name"] for config in configs}) == 6
    assert [config["C"] for config in configs[:3]] == [0.0001, 0.001, 0.002]
    assert [config["n_estimators"] for config in configs[3:]] == [1, 3, 10]


def test_configurations_produce_informative_validation_comparison(wine_csv):
    expected = sha256_file(wine_csv)
    features, target = load_verified_data(wine_csv, expected)
    x_train, x_validation, _, y_train, y_validation, _ = make_split(features, target)

    validation_f1 = []
    for config in experiment_configurations():
        model = build_pipeline(config).fit(x_train, y_train)
        metrics = calculate_metrics(y_validation, model.predict(x_validation))
        validation_f1.append(round(metrics["f1_weighted"], 6))

    assert len(set(validation_f1)) >= 5
    assert validation_f1.count(max(validation_f1)) == 1


def test_target_column_is_required(tmp_path):
    path = tmp_path / "wine.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target"):
        load_verified_data(path, sha256_file(path))


def test_clearml_publication_rejects_local_preflight():
    summary = {"dataset_id": "local-preflight:wine.csv", "dataset_sha256": "a" * 64}
    with pytest.raises(ValueError, match="official"):
        validate_official_summary(summary, dict(summary))


def test_clearml_publication_accepts_matching_official_dataset():
    summary = {"dataset_id": "9a2b28de618840fc89baf3d5daf64b20", "dataset_sha256": "a" * 64}
    validate_official_summary(summary, dict(summary))
