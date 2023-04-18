import streamlit as st
import requests
import pandas as pd
from json import loads
from sklearn.metrics import log_loss, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# KNN Model pelo MLFlow
uri = 'http://localhost:5001/invocations'

# Import da base de 3PT
url = r'C:\Users\natha\Pictures\Infnet\kobe-classifier\data\01_raw\kobe_datase.csv'
data = pd.read_csv(url)
columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
data.dropna(inplace=True)
data = data[data['shot_type'] == '3PT Field Goal']
y_real_3PT = list(data['shot_made_flag'])
data = data[columns]

# Import base de treino e teste 2PT
test_data = pd.read_parquet(r'C:\Users\natha\Pictures\Infnet\kobe-classifier\data\05_model_input\test_data.parquet')
train_data = pd.read_parquet(r'C:\Users\natha\Pictures\Infnet\kobe-classifier\data\05_model_input\train_data.parquet')

# Import de bases separadas para model_input (2PT)
X_test = pd.read_parquet(r'C:\Users\natha\Pictures\Infnet\kobe-classifier\data\04_feature\X_test.parquet')
y_test = pd.read_parquet(r'C:\Users\natha\Pictures\Infnet\kobe-classifier\data\04_feature\y_test.parquet')

# Predict pela API
def predict(test_df):
    test_df = data.to_json(orient='records')
    parsed = loads(test_df)
    evaluation = {"dataframe_records": parsed}
    response = requests.post(uri, json=evaluation)
    results = response.json()

    predict = results['predictions']

    return predict

predict_3PT = predict(data)
predict_2PT = predict(X_test)

# Avaliando predição 3PT
predict_3PT_log_loss = log_loss(y_real_3PT, predict_3PT)
predict_3PT_f1_score = f1_score(y_real_3PT, predict_3PT)


# Dashboard

st.title('Monitoramento - Kobe Classifier')


c1, c2 = st.columns(2)
with c1:
    st.metric(label='Log loss - 3PT', value=f'{predict_3PT_log_loss:.2f}')
with c2:
    st.metric(label='F1 Score - 3PT', value=f'{predict_3PT_f1_score:.2f}')
cm = confusion_matrix(y_real_3PT, predict_3PT, labels=[0.0, 1.0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0.0, 1.0])
disp.plot()
plt.plot()
fig = disp.figure_

st.pyplot(fig)


with st.expander('Características do treino'):
    st.markdown('**Shot-type: 2PT Field Goal**')
    st.write(f'- Tamanho da base de treino: {train_data.shape}')
    st.write(f'- Tamanho da base de teste: {test_data.shape}')