#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time

# Plotting
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(style='whitegrid')  

import altair as alt

# Classification Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

# Regression metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


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
  in `trained_models`, the test features data, and the test target data.
  '''
  test_features = pd.read_csv('data_test/test_features.csv')
  test_target = pd.read_csv('data_test/test_target.csv')
  
  predictions = []
  for model in trained_models:
    predictions.append(model.predict(test_features))
  
  return predictions, test_features, test_target

def submission():
  '''
  Function that produces aesthetically pleasing displays when submission occurs. 
  '''
  submission_bar = st.progress(0)
  for percent_complete in range(100):
    time.sleep(0.01)
    submission_bar.progress(percent_complete + 1)
  
  st.balloons()

#################
## Application ##
#################

st.title('Model Evaluation')
st.write(model_evaluation)

all_models = get_all_trained_models()
predictions, X_test, y_test = get_model_predictions(all_models)
metric_type = st.selectbox('What type of evaluation should be provided?', ['Classification', 'Regression', 'Clustering'])

if (st.button('Submit')):
  submission()
  
  if (metric_type == 'Classification'):
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_model_name = []
    
    for i in range(len(all_models)):
      model = all_models[i]
      model_name = type(model).__name__
      y_pred = predictions[i]
      
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      
      all_accuracy.append(accuracy)
      all_precision.append(precision)
      all_recall.append(recall)
      all_model_name.append(model_name)
            
      with st.expander(model_name + ' Evaluation'):
        st.header(model_name)

        # Important Metrics
        st.metric('Accuracy', accuracy.round(2))
        st.metric('Precision', precision.round(2))
        st.metric('Recall', recall.round(2))
        
        # Classification report
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.dataframe(df)
        
        # Confusion Matrix
        st.subheader('Confusion Matrix')
        fig, ax = plt.subplots()
        cf_matrix = confusion_matrix(y_test, y_pred)
        ax = sns.heatmap(cf_matrix, annot=True)
        st.pyplot(fig)
        
        # Precision Recall Curve
        st.subheader('Precision-Recall Curve')
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        st.pyplot(fig)
        
        # AUC Curve
        if (model_name != 'LinearSVC' and model_name != 'SVC'):
          st.subheader('AUC-ROC Curve')
          y_pred_proba = model.predict_proba(X_test)[::,1]
          fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
          
          fig, ax = plt.subplots()
          ax.plot(fpr, tpr, color='purple')
          plt.ylabel('True Positive Rate')
          plt.xlabel('False Positive Rate')
          st.pyplot(fig)
    
    st.subheader('Compare Important Metrics')
    
    fig_all, ax_all = plt.subplots()     
    ax_all = sns.barplot(x=all_model_name, y=all_accuracy)
    plt.title('Accuracy')
    st.pyplot(fig_all)
    
    fig_all, ax_all = plt.subplots()     
    ax_all = sns.barplot(x=all_model_name, y=all_precision)
    plt.title('Precision')
    st.pyplot(fig_all)
    
    fig_all, ax_all = plt.subplots()     
    ax_all = sns.barplot(x=all_model_name, y=all_recall)
    plt.title('Recall')
    st.pyplot(fig_all)
    
  elif (metric_type == 'Regression'):
    all_r2 = []
    all_mse = []
    all_mae = []
    all_model_name = []
    
    for i in range(len(all_models)):
      model = all_models[i]
      model_name = type(model).__name__
      y_pred = predictions[i]
      
      r2 = r2_score(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      mae = mean_absolute_error(y_test, y_pred)
      
      all_r2.append(r2)
      all_mse.append(mse)
      all_mae.append(mae)
      all_model_name.append(model_name)
            
      with st.expander(model_name + ' Evaluation'):
        st.header(model_name)

        # Important Metrics
        st.metric('R^2 Score', r2.round(2))
        st.metric('MSE (Mean Squared Error)', mse.round(2))
        st.metric('MAE (Mean Absolute Error)', mae.round(2))
        
        # Showing the Regression curve
        st.subheader('Regression Results on Test Data')
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(y_test)), y_test, color='purple', label='Target')
        ax.plot(np.arange(len(y_pred)), y_pred, color='skyblue', label='Predictions')
        plt.xlabel('Datapoint')
        plt.legend()
        st.pyplot(fig)
    
    st.subheader('Compare Important Metrics')
    fig_all, ax_all = plt.subplots()     
    ax_all = sns.barplot(x=all_model_name, y=all_r2)
    plt.title('R^2 Score')
    st.pyplot(fig_all)
    
    fig_all, ax_all = plt.subplots()     
    ax_all = sns.barplot(x=all_model_name, y=all_mse)
    plt.title('MSE (Mean Squared Error)')
    st.pyplot(fig_all)
    
    fig_all, ax_all = plt.subplots()     
    ax_all = sns.barplot(x=all_model_name, y=all_mae)
    plt.title('MAE (Mean Absolute Error)')
    st.pyplot(fig_all)
  
