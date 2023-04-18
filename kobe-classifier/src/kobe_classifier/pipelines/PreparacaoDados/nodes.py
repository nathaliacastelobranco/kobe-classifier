"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import mlflow
from kedro_mlflow.io.metrics import MlflowMetricDataSet

def prepared_data(data):
    columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']

    data.dropna(inplace=True)
    data = data[data['shot_type'] == '2PT Field Goal']
    data = data[columns]

    # Salvar tamanho da base no MLFLOW
    shape = {
        'row_2PT': data.shape[0],
        'columns_2PT': data.shape[1]}
    
    line_shape = data.shape[0]
    column_shape = data.shape[1]
   
    metric_column_shape = MlflowMetricDataSet(key='columns_shape_2PT')
    metric_row_shape = MlflowMetricDataSet(key='lines_shape_2PT')
    with mlflow.start_run(nested=True):
        metric_column_shape.save(column_shape)
        metric_row_shape.save(line_shape)

    return data

def split_data(prepared_data, test_size, seed):
    X = prepared_data.drop('shot_made_flag', axis = 1)
    y = prepared_data.drop(columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=seed,
                                                        shuffle=True,
                                                        stratify=y)
    
    train_data = X_train.copy()
    train_data['shot_made_flag'] = y_train.copy()
    
    test_data = X_test.copy()
    test_data['shot_made_flag'] = y_test.copy()
    
    return X_train, X_test, y_train, y_test, train_data, test_data