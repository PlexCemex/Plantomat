# Команды

Все команды выполняются **из корня проекта Plantomat**.

## 1. Установка библиотек
Устанавливает все зависимости проекта.

```bash
pip install -r requirements.txt
```

## 2. Подготовка изображений PlantVillage
Создаёт рабочий CSV только по томатным изображениям и делает split `train/val/test`.

```bash
python code/prepare_plantvillage.py --dataset-root data_raw/images/PlantVillage-Dataset --color-subdir color --output-csv data_work/plantvillage_tomato.csv
```

## 3. Подготовка сенсорного CSV
Очищает и приводит сенсорный CSV к каноническим полям проекта.

```bash
python code/prepare_udea_sensors.py --input-csv data_raw/sensors/DB_Mobile_Manual_Tomato.csv --output-csv data_work/udea_sensors_clean.csv
```

## 4. Обучение модели по изображениям
Обучает image-only классификатор с аугментациями: повороты, отражения, perspective, affine, jitter.

```bash
python code/train_image_model.py --csv data_work/plantvillage_tomato.csv --output-dir results/image_model --backbone resnet18 --image-size 224 --epochs 20 --batch-size 32 --device auto
```

## 5. Оценка модели по изображениям
Строит classification report и confusion matrix по test split.

```bash
python code/evaluate_image_model.py --csv data_work/plantvillage_tomato.csv --checkpoint results/image_model/best_image_model.pt --output-dir results/image_model/eval --split test --device auto
```

## 6. Обучение модели по датчикам
Обучает sensor-only автоэнкодер. Если CUDA/eGPU видна системе, будет использовано CUDA-устройство.

```bash
python code/train_sensor_model.py --csv data_work/udea_sensors_clean.csv --output-dir results/sensor_model --epochs 80 --batch-size 64 --device auto
```

## 7. Оценка модели по датчикам
Считает reconstruction error и долю аномалий.

```bash
python code/evaluate_sensor_model.py --csv data_work/udea_sensors_clean.csv --artifact-dir results/sensor_model --output-dir results/sensor_model/eval --device auto
```

## 8. Проверка одного растения
Отдельно анализирует фото листа и JSON датчиков, затем печатает общий вывод.

```bash
python code/analyze_plant.py --image-checkpoint results/image_model/best_image_model.pt --sensor-artifact-dir results/sensor_model --image test_inputs/images/test1.jpg --sensor-json test_inputs/sensors/healthy_demo.json --device auto
```

## 9. Проверка с рисковым JSON
Пример запуска с неблагоприятными условиями среды.

```bash
python code/analyze_plant.py --image-checkpoint results/image_model/best_image_model.pt --sensor-artifact-dir results/sensor_model --image test_inputs/images/test1.jpg --sensor-json test_inputs/sensors/humid_risk_demo.json --device auto
```

## Что положить в папки

### Изображения датасета
- `data_raw/images/PlantVillage-Dataset/`

### CSV датчиков
- `data_raw/sensors/DB_Mobile_Manual_Tomato.csv`

### Тестовые изображения
- `test_inputs/images/`

### Тестовые JSON датчиков
- `test_inputs/sensors/`
