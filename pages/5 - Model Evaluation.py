#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import confusion_matrix

#########################
## Important Variables ##
#########################

model_evaluation = '''
Model evaluation is an essential step in the machine learning process because it assesses the performance of 
the model on unseen data. This helps to determine how well the model generalizes to new situations and how
accurate its predictions are. There are various metrics used for evaluating models, such as accuracy, precision,
recall, and F1 score, and the choice of metric depends on the task and the problem being solved. Model evaluation
provides valuable information on the strengths and weaknesses of the model and can guide further improvement 
and refinement. Additionally, evaluating the model on a separate test set helps to prevent overfitting, where a 
model may have memorized the training data but fails to generalize to new data. By regularly evaluating the model,
machine learning practitioners can make informed decisions on which models to deploy in real-world applications
and improve their overall performance.
'''

######################
## Helper Functions ##
######################

def get_all_trained_models():
  '''
  Retrieves all the files from the `models_trained` directory and returns instantiated versions of the models.
  '''
  file_list = os.listdir('models_trained')
  
  trained_models = []
  for file in file_list:
    trained_models.append(joblib.load('models_trained/' + file))
  return trained_models

def get_model_predictions(trained_models):
  '''
  Retrieves the test data files from the `data_test` directory and returns predictions using the models
  in `trained_models` and the test target data.
  '''
  test_features = pd.read_csv('data_test/test_features.csv')
  test_target = pd.read_csv('data_test/test_target.csv')
  
  predictions = []
  for model in trained_models:
    predictions.append(model.predict(test_features))
  
  return predictions, test_target

#################
## Application ##
#################

st.title('Model Evaluation')
st.write(model_evaluation)

predictions, y_test = get_model_predictions(get_all_trained_models())
metric_type = st.selectbox('What type of evaluation should be provided?', ['Classification', 'Regression', 'Clustering'])

if (metric_type == 'Classification'):
  # AUC curve
  # Classification report
  # Confusion Matrix
  
  ##### Use the altair charts library that is provided by streamlit
  
  # Log-loss
  # Feature importance
  # Precision-recall curve
  pass
  
