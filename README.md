# Tomato Multimodal Public Datasets

Готовый проект для ВКР, который использует **два открытых датасета**:

1. **изображения** — `spMohanty/PlantVillage-Dataset` (томатный поднабор, 10 классов: healthy + 9 болезней);
2. **сенсоры** — `parregoces/UdeA_TomatoDataset` / `DB_Mobile_Manual_Tomato.csv` (тепличные измерения по томату, 2 664+ записей, 44 колонки по описанию репозитория GitHub).

## Что важно сразу

Этот проект делает **честный публичный прототип мультимодальной системы**:

- по фото модель учится распознавать болезни томата;
- по сенсорам модель получает **реальный тепличный контекст**;
- итоговый публичный мультимодальный CSV собирается автоматически из **двух реальных открытых датасетов**.

Но есть методологическое ограничение:

- PlantVillage — это image-only датасет без сенсорной телеметрии;
- UdeA Tomato Dataset — это sensor-only датасет по томату без меток болезней на те же самые снимки;
- поэтому пары `image + sensor` в этом публичном прототипе создаются автоматически.

Это **подходит для рабочего baseline и демонстрации мультимодальной архитектуры**, но для финальной экспериментальной части ВКР нужно дообучение на **собственных синхронных данных** с камеры и датчиков.

---

## Какие файлы добавлены специально под 2 открытых датасета

- `scripts/prepare_plantvillage.py` — собирает томатный CSV из PlantVillage.
- `scripts/prepare_udea_sensor_dataset.py` — очищает `DB_Mobile_Manual_Tomato.csv`, нормализует время, стадии роста и сенсорные поля.
- `scripts/build_public_multimodal_dataset.py` — объединяет оба открытых источника в один мультимодальный CSV.
- `scripts/train.py` — обучение мультимодальной сети.
- `scripts/evaluate.py` — оценка модели.
- `scripts/infer_image.py` — прогноз по одному фото и JSON со слепком датчиков.
- `scripts/realtime_inference.py` — режим реального времени по камере.

---

## Архитектура

Сеть состоит из двух ветвей:

- **image branch** — `EfficientNet-B0`, `ResNet18` или запасной `simple_cnn`;
- **sensor branch** — MLP по числовым и категориальным сенсорным признакам;
- **fusion head** — объединение признаков и многоклассовая классификация болезни.

Это соответствует логике ВКР: изображение даёт визуальный диагноз, сенсоры дают контекст выращивания, а на выходе система выдаёт и класс болезни, и рекомендации.

---

## Выбранные датасеты

### 1) Датасет изображений

**PlantVillage-Dataset**

Скачать:

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/PlantVillage-Dataset
```

В этом репозитории изображения лежат в `raw/color`, `raw/grayscale`, `raw/segmented`. Для обучения бери **`raw/color`**.

### 2) Датасет сенсоров

**UdeA Tomato Dataset**

Скачать:

```bash
git clone https://github.com/parregoces/UdeA_TomatoDataset.git data/raw/UdeA_TomatoDataset
```

Нужный файл:

```text
data/raw/UdeA_TomatoDataset/DB_Mobile_Manual_Tomato.csv
```

---

## Быстрый запуск

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt
```

---

## Пошаговая инструкция

## Шаг 1. Подготовить PlantVillage

```bash
python scripts/prepare_plantvillage.py --dataset-root data/raw/PlantVillage-Dataset --output-csv data/processed/plantvillage_tomato.csv
```

На выходе:

- `data/processed/plantvillage_tomato.csv`
- `data/processed/plantvillage_tomato.summary.json`

## Шаг 2. Подготовить сенсорный CSV UdeA

```bash
python scripts/prepare_udea_sensor_dataset.py --input-csv data/raw/UdeA_TomatoDataset/DB_Mobile_Manual_Tomato.csv --output-csv data/processed/udea_tomato_sensor_prepared.csv
```

Что делает этот скрипт:

- пытается сам определить разделитель CSV;
- нормализует названия колонок;
- находит или восстанавливает `timestamp`;
- строит `growth_stage`, если она отсутствует явно;
- выделяет канонические поля вроде `air_temp_c`, `air_humidity_pct`, `light_lux`, `solution_ph`, `ec_ms_cm`, если они найдены;
- сохраняет остальные пригодные числовые поля как `extra_*`.

На выходе:

- `data/processed/udea_tomato_sensor_prepared.csv`
- `data/processed/udea_tomato_sensor_prepared.summary.json`

## Шаг 3. Собрать публичный мультимодальный CSV

```bash
python scripts/build_public_multimodal_dataset.py --images-csv data/processed/plantvillage_tomato.csv --sensor-csv data/processed/udea_tomato_sensor_prepared.csv --output-csv data/processed/public_multimodal_tomato.csv
```

Этот шаг создаёт итоговый CSV, где:

- изображения берутся из PlantVillage;
- реальные сенсорные строки берутся из UdeA;
- разбиение на `train/val/test` соблюдается отдельно;
- для каждого изображения добавляется реальный сенсорный контекст из того же split;
- добавляется `sensor_context_label` (`normal`, `temperature_stress`, `humidity_stress`, `light_stress`, `mixed_stress`).

На выходе:

- `data/processed/public_multimodal_tomato.csv`
- `data/processed/public_multimodal_tomato.summary.json`

## Шаг 4. Обучить image-only baseline

```bash
python scripts/train.py --csv data/processed/plantvillage_tomato.csv --config configs/default_config.yaml --output-dir artifacts/image_only_baseline
```

## Шаг 5. Обучить публичную мультимодальную модель

```bash
python scripts/train.py --csv data/processed/public_multimodal_tomato.csv --config configs/default_config.yaml --output-dir artifacts/public_multimodal
```

Если `EfficientNet-B0` не запускается из-за конфликтов `torch/torchvision`, используй запасной вариант:

```bash
python scripts/train.py --csv data/processed/public_multimodal_tomato.csv --config configs/default_config.yaml --backbone simple_cnn --no-pretrained --output-dir artifacts/public_multimodal_simplecnn
```

## Шаг 6. Оценить модель

```bash
python scripts/evaluate.py --checkpoint artifacts/public_multimodal/best_model.pt --csv data/processed/public_multimodal_tomato.csv --output-dir artifacts/public_multimodal_eval
```

## Шаг 7. Инференс по одному фото и текущему JSON сенсоров

```bash
python scripts/infer_image.py --checkpoint artifacts/public_multimodal/best_model.pt --image path/to/test_leaf.jpg --sensor-json examples/live_sensor_snapshot.json --growth-stage vegetative
```

## Шаг 8. Режим реального времени с камерой

```bash
python scripts/realtime_inference.py --checkpoint artifacts/public_multimodal/best_model.pt --sensor-json examples/live_sensor_snapshot.json --growth-stage vegetative --camera 0
```

---

## Команды подряд

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt && git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/PlantVillage-Dataset && git clone https://github.com/parregoces/UdeA_TomatoDataset.git data/raw/UdeA_TomatoDataset && python scripts/prepare_plantvillage.py --dataset-root data/raw/PlantVillage-Dataset --output-csv data/processed/plantvillage_tomato.csv && python scripts/prepare_udea_sensor_dataset.py --input-csv data/raw/UdeA_TomatoDataset/DB_Mobile_Manual_Tomato.csv --output-csv data/processed/udea_tomato_sensor_prepared.csv && python scripts/build_public_multimodal_dataset.py --images-csv data/processed/plantvillage_tomato.csv --sensor-csv data/processed/udea_tomato_sensor_prepared.csv --output-csv data/processed/public_multimodal_tomato.csv && python scripts/train.py --csv data/processed/public_multimodal_tomato.csv --config configs/default_config.yaml --output-dir artifacts/public_multimodal
```

### Windows PowerShell

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt; git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/PlantVillage-Dataset; git clone https://github.com/parregoces/UdeA_TomatoDataset.git data/raw/UdeA_TomatoDataset; python scripts/prepare_plantvillage.py --dataset-root data/raw/PlantVillage-Dataset --output-csv data/processed/plantvillage_tomato.csv; python scripts/prepare_udea_sensor_dataset.py --input-csv data/raw/UdeA_TomatoDataset/DB_Mobile_Manual_Tomato.csv --output-csv data/processed/udea_tomato_sensor_prepared.csv; python scripts/build_public_multimodal_dataset.py --images-csv data/processed/plantvillage_tomato.csv --sensor-csv data/processed/udea_tomato_sensor_prepared.csv --output-csv data/processed/public_multimodal_tomato.csv; python scripts/train.py --csv data/processed/public_multimodal_tomato.csv --config configs/default_config.yaml --output-dir artifacts/public_multimodal
```

---

## Что будет в артефактах после обучения

В `artifacts/public_multimodal/`:

- `best_model.pt`
- `history.csv`
- `training_summary.json`
- `val_metrics.json`
- `test_metrics.json`
- `*_classification_report.json`
- `*_confusion_matrix.png`
- `*_predictions.csv`

---

## Что говорить на защите

Формулировка, которая технически корректна:

> В публичной воспроизводимой версии прототипа используются два открытых датасета: PlantVillage для визуальной диагностики болезней томата и UdeA Tomato Dataset для сенсорного контекста тепличного выращивания. Поскольку общедоступного набора с синхронными снимками листьев и IoT-телеметрией тех же растений найдено не было, мультимодальный обучающий CSV формируется автоматически из двух реальных источников и используется как baseline-реализация архитектуры. Для финальной экспериментальной части ВКР модель должна быть дообучена на собственных синхронных данных установки.

---

## Что делать после этого

После запуска на двух открытых датасетах следующий правильный шаг для ВКР такой:

1. накопить собственные снимки с камеры;
2. писать `sensor_log.csv` с реальными timestamp;
3. связывать фото и телеметрию по времени через `scripts/build_multimodal_dataset.py`;
4. дообучить сеть уже на реальных парах `изображение + датчики`.
