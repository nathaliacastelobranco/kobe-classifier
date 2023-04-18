"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_lr_pycaret, register_log_loss, train_knn_pycaret

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
             func=train_lr_pycaret,
            name='train_lr_pycaret',
            inputs=['X_test', 'y_test','train_data', 'params:seed'],
            outputs='test_lr_log_loss'
        ),
        node(
             func=train_knn_pycaret,
            name='train_knn_pycaret',
            inputs=['X_test', 'y_test','train_data', 'params:seed'],
            outputs=['knn_model', 'test_knn_log_loss', 'test_knn_f1_score']
        ),

    ])
    