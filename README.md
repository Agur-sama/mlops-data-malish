# Практические работы MLOps №3–4

Один репозиторий содержит последовательную работу с данными и экспериментами:

- практика №3 — DVC, ветвление версий Iris/Wine и ClearML Dataset;
- практика №4 — шесть MLflow-запусков на Wine, выбор модели по validation,
  однократная проверка на test и публикация лучшей модели в ClearML.

## Практика №3

Код и DVC-метаданные находятся в корне репозитория. Подробные результаты,
версии Dataset и команды проверки описаны в [REPORT.md](REPORT.md).

## Практика №4

Проект находится в [practice_4_wine](practice_4_wine). Основные материалы:

- [инструкция запуска](practice_4_wine/README.md);
- [отчёт с реальными результатами](practice_4_wine/REPORT.md);
- [объяснение для защиты](practice_4_wine/LAB4_EXPLANATION.md);
- [таблица шести запусков](practice_4_wine/artifacts/comparison.csv).

В MLflow сохранено ровно шесть официальных запусков на одном split. Их F1
различается из-за широкого диапазона сложности моделей. Победитель — Random
Forest с 10 деревьями и глубиной 3; validation F1 равен `0.972263`, test F1 —
`0.972474`.

ClearML Task:
<https://app.clear.ml/projects/9e2b5232e1df4f90aec6c7754396d8dc/experiments/902c3e4ad01f400986abecc0d772c78f/output/log>.

## Быстрый запуск практики №4

```bash
cd practice_4_wine
poetry install --with test --no-root
poetry run pytest -q
poetry run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

После запуска открыть <http://127.0.0.1:5000>.
