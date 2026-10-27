# Практическая работа №3

Версионирование наборов Iris и Wine с помощью Git, DVC и ClearML Dataset.

## Запуск

```bash
poetry install
poetry run python prepare_data.py base
poetry run python check_data.py base
```

Команды для остальных состояний: `wine`, `iris_features`; итоговое объединённое
состояние проверяется командой `poetry run python check_data.py final`.

ClearML credentials не хранятся в репозитории. После `clearml-init` версии
создаются и проверяются командами:

```bash
poetry run python create_clearml_datasets.py
poetry run python verify_clearml_datasets.py
```
