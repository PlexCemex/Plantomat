# Актуальные команды Plantomat

## 1. Подготовка PlantVillage CSV

    python "C:\Users\Admin\Studies\Plantomat\code\prepare_plantvillage.py" --dataset-root "C:\Users\Admin\Studies\Plantomat\data_raw\images\PlantVillage-Dataset" --color-subdir color --output-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantvillage_tomato.csv"

## 2. Сбор общего CSV из 4 датасетов

    python "C:\Users\Admin\Studies\Plantomat\code\prepare_realworld_mix.py" --base-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantvillage_tomato.csv" --plantdoc-root "C:\Users\Admin\Studies\Plantomat\data_raw\images\PlantDoc-Dataset\color" --pakistan-root "C:\Users\Admin\Studies\Plantomat\data_raw\images\Pakistan_Tomato_7200\color" --realworld-root "C:\Users\Admin\Studies\Plantomat\data_raw\images\Tomato_leaf_diseases_2600\color" --output-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv"

## 3. Базовое обучение image-модели на 4 датасетах

    python "C:\Users\Admin\Studies\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_model_robust_from_scratch" --backbone efficientnet_b0 --image-size 224 --epochs 20 --batch-size 16 --lr 0.0002 --device auto --workers 4 --realworld-boost 2.5

## 4. Подготовка real-world only CSV

    python "C:\Users\Admin\Studies\Plantomat\code\prepare_realworld_only.py" --mixed-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --output-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_only.csv"

## 5. Дообучение на real-world only

    python "C:\Users\Admin\Studies\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_only.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_model_stage2_realworld" --backbone efficientnet_b0 --image-size 300 --epochs 8 --batch-size 12 --lr 0.00005 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_robust_from_scratch\best_image_model_robust.pt" --realworld-boost 1.0

## 6. Мягкая донастройка ещё на 5 эпох

    python "C:\Users\Admin\Studies\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_only.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_model_stage3_polish" --backbone efficientnet_b0 --image-size 300 --epochs 5 --batch-size 12 --lr 0.00002 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage2_realworld\best_image_model_robust.pt" --realworld-boost 1.0

## 7. Оценка итоговой image-модели новым evaluator

    python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage3_polish_v2" --split test --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source all --balanced-count-per-class 100 --relative-percent-per-class 10

## 7.1. Оценка только по конкретному source

    python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage3_polish_realworld_tomato" --split test --source realworld_tomato --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source realworld_tomato --balanced-count-per-class 100 --relative-percent-per-class 10

Допустимые основные `source` для mixed CSV:

- `plantvillage`
- `plantdoc`
- `pakistan_real`
- `realworld_tomato`

## 8. Сбор hard-focus CSV для проблемных классов

    python "C:\Users\Admin\Studies\Plantomat\code\prepare_hardfocus_csv.py" --input-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --output-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_hardfocus.csv" --focus-classes "target_spot,leaf_mold,spider_mites_two_spotted_spider_mite,tomato_mosaic_virus,early_blight,septoria_leaf_spot" --focus-sources "plantdoc,pakistan_real,realworld_tomato" --focus-class-multiplier 3 --focus-source-multiplier 2 --intersection-multiplier 6 --max-multiplier 6

## 9. Stage4 hard-focus дообучение

    python "C:\Users\Admin\Studies\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_hardfocus.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus" --backbone efficientnet_b0 --image-size 300 --epochs 4 --batch-size 12 --lr 0.00001 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --realworld-boost 1.5

## 10. Проверка stage4 на полном test

    python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage4_hardfocus" --split test --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source all --balanced-count-per-class 100 --relative-percent-per-class 10

## 11. Дополнительная мягкая полировка после hard-focus

    python "C:\Users\Admin\Studies\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_model_stage5_polish_after_hardfocus" --backbone efficientnet_b0 --image-size 300 --epochs 2 --batch-size 12 --lr 0.000005 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus\best_image_model_robust.pt" --realworld-boost 1.25

## 12. Подготовка датчиков

    python "C:\Users\Admin\Studies\Plantomat\code\prepare_udea_sensors.py" --input-csv "C:\Users\Admin\Studies\Plantomat\data_raw\DB_Mobile_Manual_Tomato.csv" --output-csv "C:\Users\Admin\Studies\Plantomat\data_work\udea_sensors_clean.csv"

## 13. Обучение sensor-модели

    python "C:\Users\Admin\Studies\Plantomat\code\train_sensor_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\udea_sensors_clean.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\sensor_model" --device auto

## 14. Оценка sensor-модели

    python "C:\Users\Admin\Studies\Plantomat\code\evaluate_sensor_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\udea_sensors_clean.csv" --artifact-dir "C:\Users\Admin\Studies\Plantomat\results\sensor_model" --output-dir "C:\Users\Admin\Studies\Plantomat\results\sensor_eval"

## 15. Один запуск: фото + датчики

    python "C:\Users\Admin\Studies\Plantomat\code\analyze_plant_final.py" --image-checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --sensor-artifact-dir "C:\Users\Admin\Studies\Plantomat\results\sensor_model" --image "C:\Users\Admin\Studies\Plantomat\test_inputs\images\Test_healthy1.jpg" --sensor-json "C:\Users\Admin\Studies\Plantomat\test_inputs\sensors\Tomato___healthy1.json" --device auto

## Что появится после нового image-тестирования

- `test_confusion_matrix_counts.png` — полный test, абсолютные значения
- `test_confusion_matrix_normalized.png` — полный test, проценты
- `test_balanced_100_confusion_counts.png` — ровно 100 изображений на класс, абсолютные значения
- `test_relative_10_0_confusion_percent.png` — 10% каждого класса, относительные значения
- `test_sampling_metadata.json` — информация о сэмплировании
- `test_metrics.png` — сводные метрики
- `test_per_class_metrics.png` — precision/recall/F1 по классам
- `test_top_confusions.png` — топ перепутываний
- `test_classification_report.json` — полный JSON отчёт
- `test_predictions.csv` — все предсказания по изображениям
- `test_top_errors.csv` — примеры ошибок
- `test_source_metrics.csv` — метрики по каждому source, если колонка `source` есть в CSV
- `test_representative_examples.csv` — representative examples по классам, если задан `--representative-per-class`
- `test_summary_report.md` — текстовый отчёт для диплома
