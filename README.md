# Plantomat

Plantomat — мультимодальный AI-проект для диагностики заболеваний томата по изображению листа и контексту среды выращивания.

Проект объединяет две модальности:
- **изображение** — распознавание болезни по фото листа;
- **сенсоры** — учёт параметров среды выращивания;
- **результат** — класс состояния растения и вспомогательные рекомендации.

> В публичной версии репозитория мультимодальный датасет собирается автоматически из двух открытых источников: датасета изображений и датасета сенсорных измерений. Это baseline-прототип, а не нативно синхронизированный набор `image + sensor`.

## Структура репозитория

### `configs/`
Конфигурации обучения и правила рекомендаций.
- `default_config.yaml` — основные параметры обучения;
- `recommendation_rules.yaml` — правила интерпретации сенсорных значений.

### `scripts/`
Основные исполняемые сценарии.
- `prepare_plantvillage.py` — подготовка CSV по изображениям PlantVillage;
- `prepare_udea_sensor_dataset.py` — очистка и нормализация сенсорного CSV;
- `build_public_multimodal_dataset.py` — сборка публичного мультимодального датасета;
- `build_multimodal_dataset.py` — сборка мультимодального датасета из собственных данных;
- `train.py` — обучение модели;
- `evaluate.py` — оценка модели;
- `infer_image.py` — инференс по одному изображению и JSON датчиков;
- `realtime_inference.py` — режим работы с камерой в реальном времени.

### `tomato_ai/`
Внутренний Python-пакет проекта.
- `datasets.py` — загрузка датасетов;
- `models.py` — архитектуры моделей;
- `engine.py` — цикл обучения и валидации;
- `features.py` — обработка сенсорных признаков;
- `transforms.py` — image transforms;
- `recommender.py` — генерация рекомендаций;
- `utils.py` — общие утилиты.

### `examples/`
Примеры входных файлов.
- `live_sensor_snapshot.json` — пример текущего слепка датчиков;
- `image_manifest_example.csv` — пример манифеста изображений;
- `sensor_log_example.csv` — пример журнала сенсоров.

### `data/`
Рабочие данные проекта.
Обычно здесь лежат:
- `raw/` — исходные скачанные датасеты;
- `external/` — внешние данные;
- `processed/` — подготовленные CSV и объединённые выборки.

### `artifacts/`
Результаты обучения и оценки.
Обычно здесь появляются:
- `best_model.pt`;
- `history.csv`;
- `training_summary.json`;
- `*_metrics.json`;
- `*_classification_report.json`;
- `*_confusion_matrix.png`;
- `*_predictions.csv`.

### `requirements.txt`
Список Python-зависимостей.

## Используемые датасеты

### 1. PlantVillage-Dataset
Открытый датасет изображений листьев растений с классами заболеваний.

Ссылка:
- https://github.com/spMohanty/PlantVillage-Dataset

Что используется в проекте:
- поднабор **томата**;
- версия изображений **`raw/color`**.

Зачем нужен:
- обучение визуальной ветви модели;
- базовая классификация болезней по фото листа.

### 2. Mobile and Manual Dataset for Greenhouse Tomato Crop (UdeA)
Открытый сенсорный датасет по тепличному выращиванию томата.

Ссылки:
- https://zenodo.org/records/16745911
- https://github.com/parregoces/UdeA_TomatoDataset

Что используется в проекте:
- файл `DB_Mobile_Manual_Tomato.csv`.

Зачем нужен:
- формирование сенсорного контекста;
- обучение табличной ветви модели;
- построение публичного мультимодального baseline.

## Как работает пайплайн

1. Подготавливается CSV по изображениям PlantVillage.
2. Подготавливается CSV по сенсорам UdeA.
3. Скрипт собирает единый мультимодальный CSV.
4. Модель обучается на изображении и сенсорных признаках.
5. На выходе получается классификатор состояния растения и модуль рекомендаций.

## Быстрый запуск

### Установка
```bash
pip install -r requirements.txt
```

### Подготовка изображений
```bash
python scripts/prepare_plantvillage.py --dataset-root data/raw/PlantVillage-Dataset --output-csv data/processed/plantvillage_tomato.csv
```

### Подготовка сенсоров
```bash
python scripts/prepare_udea_sensor_dataset.py --input-csv data/raw/UdeA_TomatoDataset/DB_Mobile_Manual_Tomato.csv --output-csv data/processed/udea_tomato_sensor_prepared.csv
```

### Сборка мультимодального датасета
```bash
python scripts/build_public_multimodal_dataset.py --images-csv data/processed/plantvillage_tomato.csv --sensor-csv data/processed/udea_tomato_sensor_prepared.csv --output-csv data/processed/public_multimodal_tomato.csv
```

### Обучение
```bash
python scripts/train.py --csv data/processed/public_multimodal_tomato.csv --config configs/default_config.yaml --output-dir artifacts/public_multimodal
```

### Оценка
```bash
python scripts/evaluate.py --checkpoint artifacts/public_multimodal/best_model.pt --csv data/processed/public_multimodal_tomato.csv --output-dir artifacts/public_multimodal_eval
```

### Инференс
```bash
python scripts/infer_image.py --checkpoint artifacts/public_multimodal/best_model.pt --image path/to/image.jpg --sensor-json examples/live_sensor_snapshot.json --growth-stage vegetative
```