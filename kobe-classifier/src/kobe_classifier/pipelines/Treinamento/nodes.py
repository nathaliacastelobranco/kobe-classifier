"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

from pycaret.classification import *
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import log_loss, f1_score

import mlflow
from kedro_mlflow.io.metrics import MlflowMetricDataSet

def register_log_loss(model, X_test, y_test):
    y_test_predict = model.predict(X_test)
    metric_log_loss = log_loss(y_test, y_test_predict)

    return {
        'log_loss': {'value': metric_log_loss, 'step': 1}
    }

def register_f1_score(model, X_test, y_test):
    y_test_predict = model.predict(X_test)
    metric_f1_score = f1_score(y_test, y_test_predict)

    return {
        'f1_score': {'value': metric_f1_score, 'step': 1}
    }


def train_lr_pycaret(X_test, y_test, train_data, SEED):
    exp = ClassificationExperiment()
    exp.setup(train_data, 
          target = 'shot_made_flag', 
          session_id = SEED,  
          n_jobs=-2, 
          log_experiment='mlflow', 
          experiment_name='kobe_classifier')
    
    exp.add_metric('logloss', 
               'Log Loss', 
               log_loss, 
               greater_is_better = False)
    
    lr_model = exp.create_model('lr', 
                            verbose=True)
    
    test_lr_log_loss = register_log_loss(lr_model, X_test, y_test)
    
    return test_lr_log_loss



def train_knn_pycaret(X_test, y_test, train_data, SEED):
    exp = ClassificationExperiment()
    exp.setup(train_data, 
          target = 'shot_made_flag', 
          session_id = SEED,  
          n_jobs=-2, 
          log_experiment='mlflow', 
          experiment_name='kobe_classifier')
    
    exp.add_metric('logloss', 
               'Log Loss', 
               log_loss, 
               greater_is_better = False)
    
    exp.add_metric('f1_score', 
               'F1 Score', 
               f1_score, 
               greater_is_better = True)
    
    knn_model = exp.create_model('knn', 
                            verbose=True)
        
    # with mlflow.start_run():
    #     mlflow.sklearn.log_model(sk_model=knn_model, 
    #                              artifact_path='model',
    #                              registered_model_name='knn_model')
    
    knn_log_loss = register_log_loss(knn_model, X_test, y_test)
    knn_f1_score = register_f1_score(knn_model, X_test, y_test)
    

    return knn_model, knn_log_loss, knn_f1_score