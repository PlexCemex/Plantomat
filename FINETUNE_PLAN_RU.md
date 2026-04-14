# План дообучения Plantomat

## Цель

Поднять качество на проблемных классах и на real-world источниках, не ломая уже хорошие классы.

Главные проблемные зоны по текущей оценке:
- `target_spot`
- `leaf_mold`
- `spider_mites_two_spotted_spider_mite`
- `tomato_mosaic_virus`
- `plantdoc`
- `pakistan_real`

## Что меняется

1. `evaluate_image_model.py`
   - сохраняет обычные полные матрицы по всему test;
   - дополнительно строит матрицу **ровно 100 изображений на класс** в абсолютных значениях;
   - дополнительно строит матрицу **10% каждого класса** в относительных значениях (%).

2. `prepare_hardfocus_csv.py`
   - собирает новый CSV для дообучения;
   - усиливает проблемные классы и real-world источники только в `train`;
   - `val` и `test` не трогает.

## Пошаговый план

### Шаг 2. Снова оценить текущую stage3 модель новым evaluator

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage3_polish_v2" --split test --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source all --balanced-count-per-class 100 --relative-percent-per-class 10
```

### Шаг 3. Построить hard-focus CSV

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\prepare_hardfocus_csv.py" --input-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --output-csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_hardfocus.csv" --focus-classes "target_spot,leaf_mold,spider_mites_two_spotted_spider_mite,tomato_mosaic_virus,early_blight,septoria_leaf_spot" --focus-sources "plantdoc,pakistan_real,realworld_tomato" --focus-class-multiplier 3 --focus-source-multiplier 2 --intersection-multiplier 6 --max-multiplier 6
```

### Шаг 4. Дообучить stage4 на hard-focus CSV

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_hardfocus.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus" --backbone efficientnet_b0 --image-size 300 --epochs 4 --batch-size 12 --lr 0.00001 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage3_polish\best_image_model_robust.pt" --realworld-boost 1.5
```

### Шаг 5. Оценить stage4 на полном test

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage4_hardfocus" --split test --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source all --balanced-count-per-class 100 --relative-percent-per-class 10
```

### Шаг 6. Отдельно проверить real-world источники

#### PlantDoc

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage4_hardfocus_plantdoc" --split test --source plantdoc --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source plantdoc --balanced-count-per-class 100 --relative-percent-per-class 10
```

#### Pakistan Real

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage4_hardfocus_pakistan" --split test --source pakistan_real --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source pakistan_real --balanced-count-per-class 100 --relative-percent-per-class 10
```

#### Realworld Tomato

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\evaluate_image_model.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus\best_image_model_robust.pt" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_eval_stage4_hardfocus_realworld" --split test --source realworld_tomato --batch-size 12 --workers 4 --device auto --max-error-examples 100 --eval-mode center-crop --representative-per-class 100 --representative-source realworld_tomato --balanced-count-per-class 100 --relative-percent-per-class 10
```

## Как принимать решение по итогу

Оставлять stage4 как итоговую модель, если одновременно выполняется:
- `macro_f1` на полном test не падает больше чем на 1 п.п.;
- `target_spot recall` заметно растёт;
- `plantdoc` и `pakistan_real` улучшаются;
- сильные классы (`TYLCV`, `bacterial_spot`, `late_blight`) не проседают сильно.

Если `target_spot` вырос, но сильно просели хорошие классы, тогда сделать ещё одну мягкую полировку:

```powershell
python "C:\Users\Admin\Studies\Plantomat\code\train_image_model_robust.py" --csv "C:\Users\Admin\Studies\Plantomat\data_work\plantomat_realworld_mix.csv" --output-dir "C:\Users\Admin\Studies\Plantomat\results\image_model_stage5_polish_after_hardfocus" --backbone efficientnet_b0 --image-size 300 --epochs 2 --batch-size 12 --lr 0.000005 --device auto --workers 4 --init-checkpoint "C:\Users\Admin\Studies\Plantomat\results\image_model_stage4_hardfocus\best_image_model_robust.pt" --realworld-boost 1.25
```

После этого снова прогнать оценку.

## Какие новые файлы появятся после evaluate

Дополнительно к обычным файлам:
- `test_balanced_100_predictions.csv`
- `test_balanced_100_confusion_counts.png`
- `test_balanced_100_confusion_counts.csv`
- `test_relative_10_0_predictions.csv`
- `test_relative_10_0_confusion_percent.png`
- `test_relative_10_0_confusion_percent.csv`
- `test_relative_10_0_summary.json`
- `test_sampling_metadata.json`

