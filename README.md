# Plantomat

Plantomat — проект для диагностики заболеваний листьев томата по фотографии и оценки условий выращивания по данным датчиков.

## Что умеет
- распознаёт состояние листа по изображению;
- анализирует JSON со значениями датчиков;
- формирует итоговый вывод и рекомендации;
- устойчивее работает на реальных фото за счёт обучения на 4 датасетах, сильных аугментаций, auto-crop листа и multi-crop TTA.

## Актуальная структура
- `code/` — Python-скрипты проекта.
- `data_raw/` — сырые датасеты изображений и датчиков.
- `data_work/` — подготовленные CSV и промежуточные файлы.
- `results/` — веса моделей, графики, отчёты, метрики.
- `test_inputs/` — тестовые фотографии и JSON датчиков.

## Используемые датасеты

### Изображения
- `PlantVillage-Dataset/color`
- `PlantDoc-Dataset/color`
- `Pakistan_Tomato_7200/color`
- `Tomato_leaf_diseases_2600/color`

### Датчики
- `DB_Mobile_Manual_Tomato.csv`

## Актуальный пайплайн
1. Подготовить PlantVillage CSV.
2. Собрать общий CSV из 4 image-датасетов.
3. Обучить базовую image-модель.
4. Подготовить датчики и обучить sensor-модель.
5. Собрать `realworld_only.csv`.
6. Дообучить image-модель на real-world only.
7. Проверять фото и датчики одной командой через `code/analyze_plant_final.py`.

## Главный итоговый скрипт
`code/analyze_plant_final.py`

Он:
- принимает фото листа и JSON датчиков;
- автоматически выделяет лист;
- делает TTA-инференс по изображению;
- оценивает датчики;
- печатает единый результат.

## Какие файлы можно удалить
Если они остались от прошлых итераций и ты уже перешёл на текущий пайплайн, можно удалить:
- `code/infer_image.py`
- `code/analyze_plant.py`
- `code/analyze_plant_tta.py`
- `code/analyze_real_photo_v2.py`

Если в проекте остались ранние fusion-скрипты из старой версии, которые ты уже не используешь, их тоже можно убрать:
- `code/build_public_multimodal_dataset.py`
- `code/generate_demo_sensor_data.py`

## Какие файлы должны остаться
- `code/prepare_plantvillage.py`
- `code/prepare_realworld_mix.py`
- `code/prepare_realworld_only.py`
- `code/train_image_model_robust.py`
- `code/evaluate_image_model.py`
- `code/prepare_udea_sensors.py`
- `code/train_sensor_model.py`
- `code/evaluate_sensor_model.py`
- `code/analyze_plant_final.py`

## Итог
Текущая версия проекта использует раздельную схему:
- image-model — диагноз по фото;
- sensor-analysis — оценка среды;
- единый итоговый инференс — один запуск, одно сообщение, одно решение.
