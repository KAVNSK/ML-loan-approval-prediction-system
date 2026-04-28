# Система прогноза одобрения займа

Проект предсказывает, одобрит ли банк заявку (`Loan_Status`: `Y/N`) по анкете клиента.  
Это учебный end-to-end сценарий: загрузка данных, EDA, обучение нескольких моделей, выбор лучшей, CLI и GUI для прогноза.

## Что умеет проект

- делает базовый EDA: пропуски, распределения, выбросы;
- строит пайплайн препроцессинга: импутация, масштабирование, one-hot кодирование;
- обучает 5 моделей и сравнивает их через кросс-валидацию;
- подбирает гиперпараметры для двух лучших моделей через `GridSearchCV`;
- сохраняет артефакты в `models/` и графики в `output/`;
- умеет делать прогноз в интерактивном режиме, по одной строке и по CSV;
- содержит web GUI на Streamlit для одиночного и пакетного прогноза.

## Быстрый старт

Установить зависимости:

```bash
pip install -r requirements.txt
```

Обучить модели и сохранить метрики:

```bash
python pipeline.py
```

После этого можно запускать прогнозы.

Интерактивный режим:

```bash
python predict.py
```

Одна заявка строкой:

```bash
python predict.py --row "5000,0,120,360,1,Male,Yes,0,Graduate,No,Urban"
```

Пакетный прогноз:

```bash
python predict.py --csv archive/loan_sanction_test.csv
```

Файл результата появится рядом с входным CSV и будет называться `{имя_файла}_predictions.csv`.

GUI (Streamlit):

```bash
python -m streamlit run app.py
```

В GUI есть три вкладки:
- одиночный прогноз по форме;
- пакетный прогноз по загруженному CSV;
- просмотр графиков из `output/`.

## Как читать графики из `output/`

Смотреть лучше в таком порядке:
1. EDA: `eda_histograms.png`, `eda_boxplots.png`, `eda_correlation_heatmap.png`
2. Сравнение моделей: `model_comparison_barchart.png`
3. Ошибки и пороги: `confusion_matrices.png`, `roc_curves_all.png`
4. Интерпретация: `feature_importance.png`

Короткие ориентиры:
- `Credit_History` обычно самый сильный одиночный сигнал.
- На train у бустингов могут быть почти идеальные метрики: ориентируйтесь на CV-результаты, а не только на train-графики.
- Важность признаков показывает вклад в предсказание, но не причинность.

![Гистограммы числовых признаков](output/eda_histograms.png)
![Боксплоты числовых признаков](output/eda_boxplots.png)
![Корреляционная тепловая карта](output/eda_correlation_heatmap.png)
![Сравнение моделей по метрикам](output/model_comparison_barchart.png)
![Матрицы ошибок всех моделей](output/confusion_matrices.png)
![ROC-кривые всех моделей](output/roc_curves_all.png)
![Важность признаков](output/feature_importance.png)

## Технологии

- Данные: `pandas`, `numpy`
- Модели: `scikit-learn`, `xgboost`, `lightgbm`
- Препроцессинг: `ColumnTransformer`, `Pipeline`, `RandomOverSampler`
- Метрики и CV: `StratifiedKFold`, `cross_validate`, `GridSearchCV`
- Визуализация: `matplotlib`, `seaborn`
- GUI: `streamlit`
- Сохранение артефактов: `joblib`

## Данные

Используется датасет Kaggle Loan Prediction / Home Loan Approval:
- `archive/loan_sanction_train.csv`
- `archive/loan_sanction_test.csv`

## Структура

`config.py` - пути и константы.  
`inference.py` - общий слой инференса для CLI и GUI.  
`pipeline.py` - обучение, оценка, графики.  
`predict.py` - CLI для прогнозов.  
`app.py` - Streamlit GUI.  
`archive/` - входные данные.  
`output/` - графики (появляются после обучения).  
`models/` - сохраненные артефакты модели.
