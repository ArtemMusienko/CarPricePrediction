### Описание задачи:

Заказчик работает в компании, которая занимается продажей подержанных автомобилей.  **Основная цель**  - построить модель машинного обучения, которая сможет предсказать стоимость автомобиля на основе его характеристик. Это поможет компании более точно оценивать автомобили и предлагать клиентам справедливые цены.
### Датасет:

Используем датасет  **Car Price Dataset**  ([ссылка на датасет](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fasinow%2Fcar-price-dataset%2Fdata)). Этот датасет содержит информацию о подержанных автомобилях, включая их марку, модель, год выпуска, пробег, тип топлива, коробку передач, мощность двигателя, тип кузова и цену.

После скачивания нужно разархивировать zip-файл и добавить в файлы проекта Colab.

### Основной код творческой задачи:

Для начала скачаем необходимую библиотеку:

    pip install phik

Теперь импортируем в наш код все те библиотеки, которые потребуются для дальнейшей разработки поставленной задачи:

    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    import phik
    import xgboost as xgb
Создадим датафрейм, узнаем его структуру и выведем его в виде таблицы:

    df = pd.read_csv("car_price_dataset.csv")
    df.info()
    df.head(10)
Также воспользуемся методом удаления дубликатов - **drop_duplicates**:

    df = df.dropna().drop_duplicates()
    print("\nДубликаты:", display(df.isna().mean().sort_values(ascending = False)))
Перед разделением данных проверим количество уникальных значений, чтобы убедиться в отсутствии дисбаланса классов:

    display(df['Price'].value_counts())  #проверка количества уникальных значений
Вычислим матрицу корреляции **PhiK** для датафрейма, задав дополнительные параметры, а именно: параметр **interval_cols** указывает, какие столбцы следует рассматривать как интервальные переменные и параметр **bins** определяет, на сколько интервалов (бинов) следует разбить числовые переменные:

    df.phik_matrix(interval_cols=['Price'], bins={'Price':5})
Отфильтруем пары с высокими значениями **PhiK**, чтобы выявить наиболее значимые зависимости:

    # Рассчитываем матрицу корреляции PhiK
    phik_matrix = df.phik_matrix()
    
    # Преобразуем матрицу в таблицу с парами признаков
    phik_table = phik_matrix.stack().reset_index()
    phik_table.columns = ['Price',  'Model',  'PhiK']
    
    # Убираем диагональные элементы
    phik_table = phik_table[phik_table['Price'] != phik_table['Model']]
    
    # Сортируем по убыванию PhiK
    phik_table = phik_table.sort_values(by='PhiK', ascending=False)
    
    # Выводим топ-5 пар с наибольшими значениями PhiK
    print("\nТоп-5 пар с высокими коэффициентами PhiK:")
    print(phik_table.head())
    
    # Фильтруем пары с PhiK > 0.8
    high_phik_pairs = phik_table[phik_table['PhiK'] > 0.8]
    print("\nПары с PhiK > 0.7:")
    print(high_phik_pairs)
Для наглядности создадим тепловую карту для матрицы phik. **annot=True** добавляет значения ячеек на тепловую карту, **fmt=".2f"** форматирует числа в ячейках до двух знаков после запятой, **cmap="coolwarm"** использует цветовую схему от синего (низкие значения) к красному (высокие значения), **linewidths=0.5** задает толщину линий между ячейками:

    plt.figure(figsize=(22,  8))
    sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Тепловая карта матрицы phik")
    plt.xlabel("Столбцы")
    plt.ylabel("Строки")
    plt.show()
Конвертируем категориальные значения в числовые. Для этого выберем колонки: 'Brand', 'Model', 'Fuel_Type', 'Transmission':

    df = pd.get_dummies(df, columns=['Brand',  'Model',  'Fuel_Type',  'Transmission'], dtype = 'int')  #конвертация
Распределим данные. **X** присвоим все столбцы, кроме 'Price', а **y** только 'Price':

    X = df.drop('Price', axis=1)
    y = df['Price']
Разделение на обучающую и тестовую выборки. 20% данных будут использоваться для тестирования, а 80% — для обучения. Зададим параметр, который выдает начальное значение для генератора случайных чисел.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Для обучения модели будем использовать бустинг **XGBoost**. **XGBoost** - это одна из самых популярных и мощных библиотек для задач машинного обучения, основанная на алгоритме градиентного бустинга. Она широко используется для задач классификации, регрессии и ранжирования, благодаря своей высокой производительности, точности и гибкости. **XGBoost** — это реализация градиентного бустинга, которая использует ансамбль деревьев решений. Увеличим количество деревьев до 200, скорость обучения зададим 0.05, максимальную глубину дерева 5. Для ускорения зададим параметр "**n_jobs**". Также указываем параметр **random_state = 42**, чтобы зафиксировать получаемый результат:

    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
Теперь можно перебрать все заданные комбинации гиперпараметров в сетке и выбирать те, которые дают наилучшее качество модели на кросс-валидации. В этом поможет **GridSearchCV** — это инструмент из библиотеки **scikit-learn**, который используется для автоматического подбора оптимальных гиперпараметров модели. Используем **scoring='neg_mean_absolute_error'**, чтобы минимизировать среднюю ошибку:

    param_grid = {  #задаем сетку гиперпараметров
    'n_estimators':  [100,  200],
    'learning_rate':  [0.1,  0.05],
    'max_depth':  [4,  5]
    }
    
    grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  #используем MAE для оценки
    cv=3,
    n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)  #обучаем модель с помощью GridSearchCV
    print("Лучшие параметры: ", grid_search.best_params_)  #выводим лучшие параметры
    
    best_model = grid_search.best_estimator_ #используем лучшую модель для предсказаний
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)  #оцениваем качество модели на тестовых данных
    print("MAE: ", mae)
Используем обученную модель для создания массива предсказанных значений, сохраняя в переменной  **y_pred**.

Воспользуемся метрикой  **mean_absolute_error**, которая вычисляет среднюю абсолютную ошибку. Данная метрика показывает, насколько в среднем предсказания модели отклоняются от реальных значений. Чем меньше её показатель, тем точнее модель.

Воспользуемся метрикой  **r2_score**, она же  **коэффициент детерминации**. Она сравнивает значения переменной  **y_test**  с предсказанными  **y_pred**.

Воспользуемся  **mean_squared_error**.  **Mean_squared_error**  в библиотеке Scikit-learn — это функция, которая возвращает среднее квадратное отклонение между фактическими значениями и предсказанными значениями.

**Mean absolute percentage error (MAPE)**  — это метрика, которая выражает среднее абсолютное отклонение прогнозируемых значений от фактических значений в процентах.

Эта метрика используется для измерения точности моделей машинного обучения. Чем ниже значение  **MAPE**, тем лучше модель предсказывает значения, и наоборот, чем выше значение — тем хуже модель.

    y_pred = model.predict(X_test)
    print("RMSE:", mean_squared_error(y_test, y_pred))
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("Средняя абсолютная ошибка:", mean_absolute_error(y_test, y_pred))
    print("Коэффициент детерминации:", r2_score(y_test, y_pred))
Воспользуемся оценкой важности признаков. Это инструмент для понимания того, какие признаки наиболее значимы для модели, а какие наоборот. Можно наблюдать, что год выпуска, пробег и объём двигателя являются 3-мя самыми важными показателями при обучении модели.

    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns)
    print("Важность признаков:\n", feature_importances.sort_values(ascending=False))
Создадим график "**Реальные vs Предсказанные значения**", на которой отобразим данные **y_test** и **y_pred**, продемонстрировав связь между двумя количественными переменными. Исходя из графика можно сделать вывод, что значительных расхождений при создании взаимосвязи с данными не наблюдается.

    sns.scatterplot(x=range(len(y_test)), y=y_test, color='blue', label='Реальные (y_test)')
    sns.scatterplot(x=range(len(y_pred)), y=y_pred, color='red', label='Предсказанные (y_pred)')
    plt.xlabel('Индексы')
    plt.ylabel('Значение')
    plt.title('Реальные vs Предсказанные значения')
    plt.legend()
    plt.show()
Создадим гистограмму **распределения ошибок** (остатков) между реальными значениями **y_test** и предсказанными значениями **y_pred**. Параметр **kde=True** добавляет график оценки плотности ядра (**Kernel Density Estimate**), который помогает визуализировать плотность распределения. На графике можно наблюдать, что большая частота ошибок наблюдается в диапазоне от -500 до -100:

    residuals = y_test - y_pred
    sns.set_palette("Set2")  #цветовая палитра
    plt.figure(figsize=(14,  6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Ошибки (y_test - y_pred)")
    plt.ylabel("Частота ошибок")
    plt.title("Распределение ошибок")
    plt.show()
