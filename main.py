import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import phik
import xgboost as xgb
from IPython.display import display


df = pd.read_csv("car_price_dataset.csv")
df.info()
df.head(10)

df = df.dropna().drop_duplicates()
print("\nДубликаты:", display(df.isna().mean().sort_values(ascending = False)))

display(df['Price'].value_counts()) #проверка количества уникальных значений

df.phik_matrix(interval_cols=['Price'], bins={'Price':5})

# Рассчитываем матрицу корреляции PhiK
phik_matrix = df.phik_matrix()

# Преобразуем матрицу в таблицу с парами признаков
phik_table = phik_matrix.stack().reset_index()
phik_table.columns = ['Price', 'Model', 'PhiK']

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

plt.figure(figsize=(22, 8))
sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Тепловая карта матрицы phik")
plt.xlabel("Столбцы")
plt.ylabel("Строки")
plt.show()

df = pd.get_dummies(df, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission'], dtype = 'int') #конвертация

X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

param_grid = { #задаем сетку гиперпараметров
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [4, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error', #используем MAE для оценки
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train) #обучаем модель с помощью GridSearchCV
print("Лучшие параметры: ", grid_search.best_params_) #выводим лучшие параметры

best_model = grid_search.best_estimator_ #используем лучшую модель для предсказаний
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred) #оцениваем качество модели на тестовых данных
print("MAE: ", mae)
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
print("Средняя абсолютная ошибка:", mean_absolute_error(y_test, y_pred))
print("Коэффициент детерминации:", r2_score(y_test, y_pred))

importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns)
print("Важность признаков:\n", feature_importances.sort_values(ascending=False))

sns.scatterplot(x=range(len(y_test)), y=y_test, color='blue', label='Реальные (y_test)')
sns.scatterplot(x=range(len(y_pred)), y=y_pred, color='red', label='Предсказанные (y_pred)')
plt.xlabel('Индексы')
plt.ylabel('Значение')
plt.title('Реальные vs Предсказанные значения')
plt.legend()
plt.show()

residuals = y_test - y_pred
sns.set_palette("Set2") #цветовая палитра
plt.figure(figsize=(14, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Ошибки (y_test - y_pred)")
plt.ylabel("Частота ошибок")
plt.title("Распределение ошибок")
plt.show()