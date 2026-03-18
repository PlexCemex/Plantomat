# Актуальные команды Plantomat

## 1. Подготовка PlantVillage CSV
```powershell
python "C:\Users\Admin\go\Plantomat\code\prepare_plantvillage.py" --dataset-root "C:\Users\Admin\go\Plantomat\data_raw\images\PlantVillage-Dataset" --color-subdir color --output-csv "C:\Users\Admin\go\Plantomat\data_work\plantvillage_tomato.csv"
```

## 2. Сбор общего CSV из 4 датасетов
```powershell
python "C:\Users\Admin\go\Plantomat\code\prepare_realworld_mix.py" --base-csv "C:\Users\Admin\go\Plantomat\data_work\plantvillage_tomato.csv" --plantdoc-root "C:\Users\Admin\go\Plantomat\data_raw\images\PlantDoc-Dataset\color" --pakistan-root "C:\Users\Admin\go\Plantomat\data_raw\images\Pakistan_Tomato_7200\color" --realworld-root "C:\Users\Admin\go\Plantomat\data_raw\images\Tomato_leaf_diseases_2600\color" --output-csv "C:\Users\Admin\go\Plantomat\data_work\plantomat_realworld_mix.csv"
```

## 3. Базовое обучение image-модели на 4 датасетах
```powershell
python "C:\Users\Admin\go\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\go\Plantomat\data_work\plantomat_realworld_mix.csv" --output-dir "C:\Users\Admin\go\Plantomat\results\image_model_robust_from_scratch" --backbone efficientnet_b0 --image-size 224 --epochs 20 --batch-size 16 --lr 0.0002 --device auto --workers 4 --realworld-boost 2.5
```

## 4. Подготовка real-world only CSV
```powershell
python "C:\Users\Admin\go\Plantomat\code\prepare_realworld_only.py" --mixed-csv "C:\Users\Admin\go\Plantomat\data_work\plantomat_realworld_mix.csv" --output-csv "C:\Users\Admin\go\Plantomat\data_work\plantomat_realworld_only.csv"
```

## 5. Дообучение на real-world only
```powershell
python "C:\Users\Admin\go\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\go\Plantomat\data_work\plantomat_realworld_only.csv" --output-dir "C:\Users\Admin\go\Plantomat\results\image_model_stage2_realworld" --backbone efficientnet_b0 --image-size 300 --epochs 8 --batch-size 12 --lr 0.00005 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\go\Plantomat\results\image_model_robust_from_scratch\best_image_model_robust.pt" --realworld-boost 1.0
```

## 6. Мягкая донастройка ещё на 5 эпох
```powershell
python "C:\Users\Admin\go\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\go\Plantomat\data_work\plantomat_realworld_only.csv" --output-dir "C:\Users\Admin\go\Plantomat\results\image_model_stage3_polish" --backbone efficientnet_b0 --image-size 300 --epochs 5 --batch-size 12 --lr 0.00002 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\go\Plantomat\results\image_model_stage2_realworld\best_image_model_robust.pt" --realworld-boost 1.0
```

## 7. Оценка итоговой image-модели
```powershell
python "C:\Users\Admin\go\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\go\Plantomat\data_work\plantomat_realworld_only.csv" --checkpoint "C:\Users\Admin\go\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --output-dir "C:\Users\Admin\go\Plantomat\results\image_eval_stage3_polish" --split test
```

## 8. Подготовка датчиков
```powershell
python "C:\Users\Admin\go\Plantomat\code\prepare_udea_sensors.py" --input-csv "C:\Users\Admin\go\Plantomat\data_raw\DB_Mobile_Manual_Tomato.csv" --output-csv "C:\Users\Admin\go\Plantomat\data_work\udea_sensors_clean.csv"
```

## 9. Обучение sensor-модели
```powershell
python "C:\Users\Admin\go\Plantomat\code\train_sensor_model.py" --csv "C:\Users\Admin\go\Plantomat\data_work\udea_sensors_clean.csv" --output-dir "C:\Users\Admin\go\Plantomat\results\sensor_model" --device auto
```

## 10. Оценка sensor-модели
```powershell
python "C:\Users\Admin\go\Plantomat\code\evaluate_sensor_model.py" --csv "C:\Users\Admin\go\Plantomat\data_work\udea_sensors_clean.csv" --artifact-dir "C:\Users\Admin\go\Plantomat\results\sensor_model" --output-dir "C:\Users\Admin\go\Plantomat\results\sensor_eval"
```

## 11. Один запуск: фото + датчики
```powershell
python "C:\Users\Admin\go\Plantomat\code\analyze_plant_final.py" --image-checkpoint "C:\Users\Admin\go\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --sensor-artifact-dir "C:\Users\Admin\go\Plantomat\results\sensor_model" --image "C:\Users\Admin\go\Plantomat\test_inputs\images\Test_healthy1.jpg" --sensor-json "C:\Users\Admin\go\Plantomat\test_inputs\sensors\Tomato___healthy1.json" --device auto
```

## 12. Пакетная проверка фото одной командой
```powershell
Get-ChildItem "C:\Users\Admin\go\Plantomat\test_inputs\images" -File | ForEach-Object { Write-Host "`n===== $($_.Name) ====="; python "C:\Users\Admin\go\Plantomat\code\analyze_plant_final.py" --image-checkpoint "C:\Users\Admin\go\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --sensor-artifact-dir "C:\Users\Admin\go\Plantomat\results\sensor_model" --image $_.FullName --sensor-json "C:\Users\Admin\go\Plantomat\test_inputs\sensors\Tomato___healthy1.json" --device auto }
```
