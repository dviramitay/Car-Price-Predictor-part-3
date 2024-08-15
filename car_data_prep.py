#!/usr/bin/env python
# coding: utf-8

# ## dvir amitay  github - https://github.com/dviramitay/project-/blob/main/Project%20-%20part%202.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

file_path = r'C:\Users\dvir\Desktop\שנה ג\סמסטר ב\כרייה וניתוח נתונים\מטלות\מטלה 2\dataset.csv'
df = pd.read_csv(file_path)


def fill_missing_values(df):
    # Km according to how many years the car has been on the road *15000 (annual average in Israel)
    df['Km'] = df.apply(lambda row: (2024 - row['Year']) * 15000 if pd.isna(row['Km']) else row['Km'], axis=1)

    # For capacity Engine , we will find all vehicles of the same model and fill in the most common value. If there are no more vehicles of this model, we will fill in "0"
    def mode(series):
        return series.mode().iloc[0] if not series.mode().empty else np.nan

    df['capacity_Engine'] = df.groupby(['manufactor', 'Year', 'model'])['capacity_Engine'].transform(
        lambda x: x.fillna(mode(x)))
    df['capacity_Engine'] = df.groupby('model')['capacity_Engine'].transform(lambda x: x.fillna(mode(x)))
    df['capacity_Engine'].fillna('0', inplace=True)
    df['capacity_Engine'] = df['capacity_Engine'].astype(str).str.replace(',', '').str.replace(' ', '')
    df['capacity_Engine'] = df['capacity_Engine'].astype(int)

    # In the past, it was known that most vehicles were produced with a manual transmission and over the years an automatic transmission was used. Therefore, we will check which gear was the most common in that year and fill in the values ​​accordingly
    mode_gear_per_year = df.groupby('Year')['Gear'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')

    def fill_missing_gear(row):
        if pd.isnull(row['Gear']):
            return mode_gear_per_year[row['Year']]
        else:
            return row['Gear']

    df['Gear'] = df.apply(fill_missing_gear, axis=1)
    df['Engine_type'].fillna('Unknown', inplace=True)
    df['Curr_ownership'].fillna('Unknown', inplace=True)
    df['Color'].fillna('Unknown', inplace=True)

    return df


def create_new_columns(df):
    # create a new column in order to accurately determine for the model how much the vehicle was used
    df['Car_age'] = 2024 - df['Year']
    df['Km'] = df['Km'].astype(str).str.replace(',', '').str.replace(' ', '')
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['Km'] = df.apply(lambda row: (2024 - row['Year']) * 15000 if pd.isna(row['Km']) else row['Km'], axis=1)
    df['Km'] = df['Km'].astype(int)
    df.loc[(df['Km'] < 1000) & (df['Car_age'] > 1), 'Km'] *= 1000
    df['Average_km_per_year'] = df['Km'] / df['Car_age']
    df.loc[df['Car_age'] == 0, 'Average_km_per_year'] = 0
    df['Average_km_per_year'] = df['Average_km_per_year'].round().astype(int)

    # create a new column that classifies the car's color by its popularity
    color_counts = df['Color'].value_counts()
    color_popularity_mapping = {color: rank for rank, color in enumerate(color_counts.index, start=1)}
    df['Color_Popularity'] = df['Color'].map(color_popularity_mapping)

    return df


def prepare_data(df):
    df = fill_missing_values(df)
    df = create_new_columns(df)
    columns_to_keep = ['Hand', 'capacity_Engine', 'Km', 'Car_age', 'Average_km_per_year', 'Color_Popularity', 'Price']
    df = df.filter(columns_to_keep + ['manufactor', 'model', 'Gear', 'Engine_type', 'Curr_ownership', 'Color'])
    df = pd.get_dummies(df, columns=['manufactor', 'model', 'Gear', 'Engine_type', 'Curr_ownership', "Color"],
                        drop_first=True)

    return df
