# Отчёт по практической работе №3

## Окружение

- Python: `3.12.5`
- DVC: `3.67.1`
- ClearML: `2.1.12`
- scikit-learn: `1.9.1`
- pandas: `3.1.0`
- Git-репозиторий: <https://github.com/Agur-sama/mlops-data-malish>

Все прямые зависимости зафиксированы точными версиями в `pyproject.toml`, а
полное дерево зависимостей — в `poetry.lock`.

## DVC

### Версии и история Git

| Состояние | Ветка/тег | Git commit |
|---|---|---|
| Базовый Iris | `data-v1` | `854ca7eca1b2a19d40fcdd85c1e3936e6023136c` |
| Iris + Wine | `feature/wine` | `4fd678e9fedbe94e6d58bdd03dd06d8fd9bdb20d` |
| Iris с вычисляемыми признаками | `feature/iris-features` до слияния | `09dbcad4a6e492ec954de42baa722396745f1945` |
| Слияние двух версий | `feature/iris-features` | `7e7cce2b450d5a1e561087563ff5c91f24994263` |
| Итоговая версия | `main` | `fa6f2857d88c1354d896ec53720d7d0e37cbd132` |

Для `.dvc`-файлов настроен merge driver `dvc`. При слиянии
`feature/wine` в `feature/iris-features` Git вывел `Auto-merging data.dvc`,
маркеров конфликта не осталось. История обеих веток сохранена merge-коммитом.

### DVC remote

Remote `local-storage` — отдельная локальная директория вне Git-репозитория:

```text
/Users/maksim/Documents/Codex/2026-10-27/new-chat/work/dvc_remote_malish
```

Путь не содержит секретов и сохранён в `.dvc/config`. Для каждой версии был
выполнен `dvc push`; для итоговой версии `dvc status -c` показал синхронизацию
cache и remote.

### Хэши итоговой версии

| Файл | SHA-256 | Форма |
|---|---|---|
| `data/iris.csv` | `3e1ba63f55da2aabc432d00b1670140f5162bf9ef34b837598ec2fc176c9ff26` | 150 × 7 |
| `data/wine.csv` | `38860ba04ace4d402242340bcf40be4bd6f4ebde9144a3dc89dc12c20f7f8e60` | 178 × 14 |
| `data.dvc` | `b63003486e5847feba08f387096e7a53191e49eae92790a231dc6c99bbce7692` | DVC-метаданные директории |

Базовый `iris.csv` имел SHA-256
`6d700503c9c2e2c71d187b889a87af8a8d3a6dd5bae1266d46f4ee4db5a6349f`.
В ветке `feature/wine` он остался неизменным. В итоговом Iris добавлены только
`sepal_area` и `petal_area`, а Wine совпадает с версией из `feature/wine`.

### Чистое восстановление

Репозиторий был клонирован в новую директорию без копирования старого workspace
и cache. Команда `dvc pull` сообщила `3 files fetched and 2 files added`.
`python check_data.py final` подтвердил два CSV, указанные выше SHA-256, формы,
порядок столбцов, отсутствие пропусков и правильность вычисляемых признаков.

## ClearML Dataset

Подготовлены воспроизводимые скрипты:

- `create_clearml_datasets.py` создаёт `1.0.0`, `1.1.0` и `2.0.0` с требуемым
  lineage и записывает IDs в локальный `clearml_ids.json`;
- `verify_clearml_datasets.py` получает версии строго по ID, скачивает их в три
  директории и сохраняет dependency graph, added/modified/removed files, формы,
  столбцы и SHA-256 в `clearml_verification.json`.

Credentials, файлы IDs и скачанные копии исключены из Git. Выполнение на
учебном ClearML server ожидает настройки SDK командой `clearml-init`.

| Версия | Dataset ID | Родитель | Состав и изменения |
|---|---|---|---|
| `1.0.0` | `BASE_ID_PENDING` | нет | исходный `iris.csv`, тег `base` |
| `1.1.0` | `WINE_ID_PENDING` | `BASE_ID_PENDING` | наследует Iris, добавляет `wine.csv` |
| `2.0.0` | `IRIS_FEATURES_ID_PENDING` | `BASE_ID_PENDING` | изменяет `iris.csv`, не наследует Wine |

Dependency graph и проверочные хэши будут вставлены после подключения к
учебному ClearML server.

## Сравнение DVC и ClearML Dataset

| Наблюдение | DVC | ClearML Dataset |
|---|---|---|
| Идентификатор версии | Git commit фиксирует конкретный `data.dvc`, внутри которого хранится content hash | У каждой финализированной версии есть неизменяемый Dataset ID |
| Расположение метаданных | `data.dvc` и `.dvc/config` находятся в Git | Метаданные версии, ID и lineage находятся на ClearML server |
| Расположение файлов | Содержимое лежит в DVC cache и `local-storage`, а не в Git | Файлы загружаются в storage, настроенный для ClearML server |
| Создание дочерней версии | Из нужного Git commit создаётся ветка и обновляется `data.dvc` | `Dataset.create(..., parent_datasets=[BASE_ID])` |
| Переключение версии | `git switch`, затем `dvc checkout` | `Dataset.get(dataset_id=...)` и скачивание локальной копии |
| Чистое восстановление | В новом clone выполнен `dvc pull`, получены два файла с теми же хэшами | Версия скачивается на другую машину строго по Dataset ID |
| Ветвление | `feature/wine` и `feature/iris-features` созданы независимо от `main` | `1.1.0` и `2.0.0` указывают один `BASE_ID` как parent |
| Слияние | DVC merge driver объединил дерево: изменённый Iris плюс Wine | Lineage отражает родителей, но файловое слияние нужно оформлять новой Dataset version |
| Связь с Git | Версия данных определяется парой Git commit + доступные объекты DVC remote | Dataset ID не зависит от Git commit; ссылку на него нужно записать в отчёт или код |
| UI | В обычном Git UI видны коммиты и `data.dvc`, но не содержимое remote | ClearML UI показывает версии, статусы, теги, файлы и lineage |

## Ответы на вопросы

1. В DVC конкретное состояние однозначно задаёт DVC-метафайл с content hash,
   зафиксированный конкретным Git commit. Сам Git commit без доступного объекта
   в cache или remote не позволяет восстановить данные.
2. Git хранит только маленький `data.dvc`. Без `dvc push` новый clone узнает
   требуемый хэш, но не найдёт соответствующее содержимое для `dvc pull`.
3. Dataset version (`1.0.0`, `1.1.0`) — удобная человеку метка в рамках имени
   Dataset. Dataset ID — глобальный неизменяемый идентификатор конкретного
   объекта, поэтому для точного получения надёжнее использовать ID.
4. `finalize()` переводит текущий mutable Dataset в неизменяемое состояние. Он
   не вызывает `Dataset.create()` и не создаёт другой объект или новую версию.
5. У версии `2.0.0` нужно проверить dependency graph/parent IDs: единственным
   parent должен быть `BASE_ID`, а `WINE_ID` не должен быть её предком.
6. После `git switch` изменится `data.dvc`, но workspace может остаться от
   предыдущей ветки. `dvc checkout` приводит файлы в соответствие метаданным.
7. DVC удобнее здесь для версионирования данных рядом с кодом, ветвления Git и
   объединения независимых файлов. ClearML Dataset удобнее для выдачи версии по
   ID, просмотра lineage и скачивания через общий сервер/UI без переключения Git.
