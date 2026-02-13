![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Car Price Prediction

### Описание смысла модели:

Заказчик работает в компании, которая занимается продажей подержанных автомобилей.  **Основная цель**  - построить модель машинного обучения, которая сможет предсказать стоимость автомобиля на основе его характеристик. Это поможет компании более точно оценивать автомобили и предлагать клиентам справедливые цены.

### Основная информация:

В коде используется датасет **[Car Price Dataset](https://www.google.com/url?q=https://www.kaggle.com/datasets/asinow/car-price-dataset/data)**.  Для наших данных построим матрицу корреляции **PhiK** и для наглядности создадим тепловую карту для нее.

Для обучения модели будем использовать бустинг **XGBoost**. **XGBoost** - это одна из самых популярных и мощных библиотек для задач машинного обучения, основанная на алгоритме градиентного бустинга. Она широко используется для задач классификации, регрессии и ранжирования, благодаря своей высокой производительности, точности и гибкости. **XGBoost** — это реализация градиентного бустинга, которая использует ансамбль деревьев решений. 

Также мы перебираем все заданные комбинации гиперпараметров в сетке и выбираем те, которые дают наилучшее качество модели на кросс-валидации. В этом поможет **GridSearchCV** — это инструмент из библиотеки **scikit-learn**, который используется для автоматического подбора оптимальных гиперпараметров модели:

    param_grid = {  #задаем сетку гиперпараметров
			    'n_estimators':  [100,  200],
			    'learning_rate':  [0.1,  0.05],
				'max_depth':  [4,  5]
    }
    
    grid_search = GridSearchCV(
			    estimator=model,
			    param_grid=param_grid,
			    scoring='neg_mean_absolute_error',
			    cv=3,
			    n_jobs=-1
    )

    grid_search.fit(X_train, y_train)  #обучаем модель с помощью GridSearchCV
    print("Лучшие параметры: ", grid_search.best_params_)  #выводим лучшие параметры
