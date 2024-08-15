import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle
from car_data_prep import prepare_data

file_path = r'C:\Users\dvir\Desktop\שנה ג\סמסטר ב\כרייה וניתוח נתונים\מטלות\מטלה 2\dataset.csv'
df = pd.read_csv(file_path)
df_prepared = prepare_data(df)


def perform_elastic_net_with_cv(X, y, param_grid):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elastic_net = ElasticNet(random_state=42)

    grid_search = GridSearchCV(elastic_net, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)

    best_model = grid_search.best_estimator_

    neg_mse_scores = cross_val_score(best_model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')

    return best_model


param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}
X = df_prepared.drop(columns='Price')
y = df_prepared['Price']

best_model = perform_elastic_net_with_cv(X, y, param_grid)

pickle.dump(best_model, open("trained_model.pkl", "wb"))
