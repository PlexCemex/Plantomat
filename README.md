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
4. Подготовить `realworld_only.csv`.
5. Дообучить image-модель на real-world only.
6. Сделать мягкую донастройку итоговой image-модели.
7. Оценить image-модель через `code/evaluate_image_model.py`.
8. Подготовить датчики и обучить sensor-модель.
9. Проверять фото и датчики одной командой через `code/analyze_plant_final.py`.

## Что изменено в текущей версии

- `COMMANDS_RU.md` синхронизирован с реальным CLI `code/evaluate_image_model.py`.
- `code/evaluate_image_model.py` теперь принимает `--representative-per-class` и `--representative-source`, из-за которых раньше падала команда.
- `code/evaluate_image_model.py` теперь умеет фильтровать оценку по `source` и сохранять `test_source_metrics.csv`.
- оценочный preprocessing можно явно задавать через `--eval-mode`; для robust checkpoint рекомендуется `center-crop`.
- `code/plantomat/image_pipeline.py` поддерживает два eval-режима: обычный `resize` и `center-crop`.

## Главный итоговый скрипт

`code/analyze_plant_final.py`

Он:

- принимает фото листа и JSON датчиков;
- автоматически выделяет лист;
- делает TTA-инференс по изображению;
- оценивает датчики;
- печатает единый результат.

## Оценка image-модели

Основная команда находится в `COMMANDS_RU.md`.

Сейчас для итогового robust checkpoint рекомендуется запускать оценку с:

- `--eval-mode center-crop`
- `--max-error-examples 50`
- `--representative-per-class 50`
- `--representative-source all`

Если в mixed CSV есть колонка `source`, скрипт дополнительно сохраняет:

- `test_source_metrics.csv` — метрики по источникам;
- `test_representative_examples.csv` — representative examples по классам.

## Итог

Текущая версия проекта использует раздельную схему:

- image-model — диагноз по фото;
- sensor-analysis — оценка среды;
- единый итоговый инференс — один запуск, одно сообщение, одно решение.
