"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepared_data, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepared_data,
            name='prepared_data',
            inputs='raw_data',
            outputs='data_filtered'
        ),
        node(
            func=split_data,
            name='split_data',
            inputs=['data_filtered', 'params:test_size', 'params:seed'],
            outputs=['X_train', 'X_test', 'y_train', 'y_test', 'train_data', 'test_data']
        )
    ])
