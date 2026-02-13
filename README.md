![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Car Price Prediction

[![ru](https://img.shields.io/badge/README_на_русском-2A2C39?style=for-the-badge&logo=github&logoColor=white)](README.ru.md)

### Description of the model's purpose:

The customer works for a company that sells used cars. The **main goal**  is to build a machine learning model that can predict the value of a car based on its characteristics. This will help the company to evaluate cars more accurately and offer fair prices to customers.

### Basic information:

The code uses the dataset **[Car Price Dataset](https://www.google.com/url?q=https://www.kaggle.com/datasets/asinow/car-price-dataset/data)**. For our data, we will build a correlation matrix **PhiK** and create a heat map for it.

We will use **XGBoost** boosting to train the model. **XGBoost** is one of the most popular and powerful libraries for machine learning tasks, based on the gradient boosting algorithm. It is widely used for classification, regression, and ranking tasks due to its high performance, accuracy, and flexibility. **XGBoost** is a gradient boosting implementation that uses an ensemble of decision trees. 

Also, we iterate over all the given combinations of hyperparameters in the grid and select those that give the best model quality on cross-validation. This is where **GridSearchCV** comes in handy - it is a tool from the **scikit-learn** library that is used to automatically select the optimal model hyperparameters:

    param_grid = {  #setting a grid of hyperparameters
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

    grid_search.fit(X_train, y_train)  #train the model using GridSearchCV
    print("The best parameters: ", grid_search.best_params_)  #output the best parameters