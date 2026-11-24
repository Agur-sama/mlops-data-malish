# Отчёт по практике 4

## Официальный источник данных

- GitHub практики 3: <https://github.com/Agur-sama/mlops-data-malish>.
- ClearML project: `MIIT/MLOps/practice_3/malish`.
- Dataset: `sklearn-tabular-malish`, version `1.1.0`.
- WINE_ID: `9a2b28de618840fc89baf3d5daf64b20`.
- WINE_SHA256: `38860ba04ace4d402242340bcf40be4bd6f4ebde9144a3dc89dc12c20f7f8e60`.
- Dataset URL: <https://app.clear.ml/datasets/simple/7735890b414747d493153662254d5051/experiments/9a2b28de618840fc89baf3d5daf64b20>.

ClearML SDK действительно скачал Dataset. В нём найдены `iris.csv` и
`wine.csv`; форма Wine — 178 строк и 14 столбцов. Рассчитанный SHA-256 полностью
совпал с REPORT практики 3. Результат сохранён в
`artifacts/dataset_verification.json`.

## Окружение и split

- Python 3.12.5; MLflow 2.19.0; ClearML SDK 1.16.5; scikit-learn 1.5.2.
- Tracking URI: `file:./mlruns`.
- Experiment: `miit-mlops-practice-4-malish`.
- Experiment ID: `212918486938020074`.

Данные один раз разделены со стратификацией и `random_state=42`: 106 train,
36 validation и 36 test. Test не участвовал в сравнении конфигураций. Индексы
зафиксированы в `artifacts/split_manifest.csv`.

## Шесть официальных MLflow runs

| Run ID | Family | Конфигурация | Validation F1 | Training, s |
|---|---|---|---:|---:|
| `4ba8030412de4058ae9cc0bfc63e0437` | Random Forest | 10, depth=3, leaf=3 | 0.972263 | 0.00568 |
| `420edde16ff049a89141843748d13360` | Logistic Regression | C=0.002 | 0.944961 | 0.00356 |
| `a5bbe58ae6a64ad5b0c4625bfa636395` | Random Forest | 3, depth=2, leaf=5 | 0.944269 | 0.00882 |
| `7938a52332f44c74808664111132acae` | Logistic Regression | C=0.001 | 0.725530 | 0.00311 |
| `7534e00e863548d4af3e310347e7fd67` | Random Forest | 1, depth=1, leaf=10 | 0.464646 | 0.00303 |
| `2fa7039d914a473d963c8adb8bb84ac8` | Logistic Regression | C=0.0001 | 0.217778 | 0.00822 |

Все runs содержат одинаковый набор метрик, dataset ID/SHA tags, signature,
input example, параметры и model artifact. Значения различаются: намеренно
широкий диапазон сложности показывает недообучение слабых конфигураций и даёт
однозначного победителя. Test-метрики записаны только в выбранный run:

- accuracy: `0.972222`;
- precision weighted: `0.975000`;
- recall weighted: `0.972222`;
- F1 weighted: `0.972474`.

Модель повторно загружена по `runs:/4ba8030412de4058ae9cc0bfc63e0437/model`.
Предсказания до и после загрузки совпали. Все проверки в
`artifacts/verification.json` имеют значение `true`.

## ClearML Task и опубликованная модель

- Task ID: `902c3e4ad01f400986abecc0d772c78f`.
- Task: <https://app.clear.ml/projects/9e2b5232e1df4f90aec6c7754396d8dc/experiments/902c3e4ad01f400986abecc0d772c78f/output/log>.
- Model ID: `151f3ebe430a48f8afc99ab3e7ea0257`.
- Загруженные веса: <https://files.clear.ml/MIIT/MLOps/practice_4/malish/wine-best-model.902c3e4ad01f400986abecc0d772c78f/models/best_model.joblib>.

SDK-проверка подтвердила: Task имеет статус `completed`, Model —
`published=True`, а metadata модели содержит официальный Dataset ID, SHA-256 и
лучший MLflow run ID. Файл весов находится на ClearML files server, а не по
локальному пути.

`publish_clearml.py` отдельно запрещает публикацию любого результата с
`dataset_id=local-preflight:*` и проверяет согласованность summary/verification.

## Повторение и демонстрация

```bash
poetry install --with test --no-root
poetry run pytest -q
CLEARML_CONFIG_FILE="$HOME/clearml.conf" poetry run python run_experiments.py \
  --dataset-id 9a2b28de618840fc89baf3d5daf64b20 \
  --expected-sha256 38860ba04ace4d402242340bcf40be4bd6f4ebde9144a3dc89dc12c20f7f8e60 \
  --tracking-uri file:./mlruns
poetry run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Официальный каталог `mlruns/` сохранён вместе с проектом. Старые технические
preflight-запуски и кэши не входят в рабочий проект и не используются как
доказательство сдачи.
