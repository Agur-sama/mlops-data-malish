# Практика 4 — управление экспериментами Wine

Проект сравнивает шесть конфигураций моделей на одном и том же Wine Dataset:
три Logistic Regression и три Random Forest. Данные загружаются по ClearML
Dataset ID, перед обучением проверяется SHA-256, а все параметры, метрики,
signature, input example и модель записываются в MLflow.

## Установка

```bash
poetry install --with test --no-root
poetry run python --version
poetry run python -m pip check
poetry run pytest -q
```

Требуется Python 3.12. Значения ClearML API credentials хранятся только в
`~/clearml.conf` или переменных окружения и никогда не добавляются в Git.

## Формальный запуск для сдачи

Возьмите `WINE_ID` и `WINE_SHA256` из отчёта практики 3:

```bash
poetry run python run_experiments.py \
  --dataset-id "$WINE_ID" \
  --expected-sha256 "$WINE_SHA256" \
  --tracking-uri file:./mlruns
```

Скрипт завершится до обучения, если Dataset ID неверен, в Dataset нет ровно
одного `wine.csv` либо SHA-256 не совпал. Повторный запуск использует тот же
experiment и создаёт новые runs.

После проверки MLflow лучшая модель публикуется в ClearML:

```bash
poetry run python publish_clearml.py \
  --project MIIT/MLOps/practice_4/malish
```

Команда создаёт Task `wine-best-model`, загружает таблицу сравнения,
`verification.json`, test-метрики и публикует `OutputModel` с provenance.
Публикация локального `local-preflight` запрещена проверкой в коде.

Фактический официальный результат:

- Dataset ID `9a2b28de618840fc89baf3d5daf64b20`;
- Experiment ID `212918486938020074`;
- best run `4ba8030412de4058ae9cc0bfc63e0437`;
- validation F1 `0.972263`, test F1 `0.972474`;
- ClearML Task `902c3e4ad01f400986abecc0d772c78f`;
- ClearML Model `151f3ebe430a48f8afc99ab3e7ea0257`.

## MLflow UI

```bash
poetry run mlflow ui --backend-store-uri ./mlruns \
  --host 127.0.0.1 --port 5000
```

Откройте <http://127.0.0.1:5000>. В experiment должны быть шесть runs с
одинаковым split и одинаковым набором названий validation-метрик. Значения
различаются из-за разной сложности моделей. Только у лучшего run есть метрики
с префиксом `test_`.

## Локальная предварительная проверка

Ключ `--wine-csv` существует только для проверки кода без ClearML. Такой запуск
не заменяет формальный запуск по Dataset ID и помечается тегом
`local-preflight:*`.

Официальные шесть runs находятся только в `mlruns/` и имеют настоящий ClearML
Dataset ID в tags. Технический preflight и кэши удалены из рабочего проекта.

## Что создаётся

- `artifacts/comparison.csv` — сравнение шести runs;
- `artifacts/verification.json` — результат повторной загрузки и проверки;
- `artifacts/best_model.joblib` — копия лучшей модели для ClearML;
- `artifacts/classification_report.json` и `confusion_matrix.csv` — бонусные артефакты;
- `artifacts/split_manifest.csv` — неизменный состав train/validation/test;
- `artifacts/summary.json` — идентификаторы и итоговые метрики.

Детальное объяснение для защиты находится в `LAB4_EXPLANATION.md`.
