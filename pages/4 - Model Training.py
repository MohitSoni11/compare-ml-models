#############
## Imports ##
#############

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.model_selection import train_test_split

#########################
## Important Variables ##
#########################

model_training = '''
Training a machine learning model is critical for achieving good performance and accuracy in tasks such
as classification, regression, and clustering. It involves feeding the model a large amount of data and
adjusting the model's parameters to minimize the error between the predicted output and actual output. 
The choice of training data and the optimization algorithm used can significantly impact the final model
performance. The training process also helps to prevent overfitting, where a model learns the training data
too well and does not generalize well to new data. A well-trained model is essential for making accurate 
predictions and decisions in real-world applications, which is why it is crucial to invest time and resources
in properly training the model.
'''

test_size_text = '''
The test size when splitting data is essential as it determines the portion of the data used for evaluating
the performance of the model. A good test size should be large enough to provide a representative sample
of the data, but not too large that it leaves limited data for training the model. A common practice is to 
split the data into training and test sets, with a commonly used ratio of 70:30 or 80:20. So, if you are not
sure of what you want to enter for the test size, consider entering 20 or 30.
'''

######################
## Helper Functions ##
######################

def get_all_models():
  '''
  Retrieves all the files from the `models` directory and returns instantiated versions of the models.
  '''
  file_list = os.listdir('models')
  
  models_list = []
  for file in file_list:
    if (not file == 'placeholder.txt'):
      models_list.append(joblib.load('models/' + file))
  return models_list

def train_all_models(models_list, test_size):
  '''
  Trains all the models in `models_list` based on the `features.csv` and `target.csv` stored in the `data` folder.
  Returns the trained models with the test data.
  '''
  # Get data
  features = pd.read_csv('data/features.csv')
  target = pd.read_csv('data/target.csv')
  
  # Split data
  X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size)
  
  # Train models
  trained_models = []
  for model in models_list:
    model_name = type(model).__name__
    model_training = st.empty()
    model_training.info(f'{model_name} training...')
    time.sleep(2)
        
    start_time = time.time()
    model.fit(X_train, y_train)
    trained_models.append(model)
    elapsed_time = time.time() - start_time
    
    model_training.empty()
    st.success(f'{model_name} trained in {elapsed_time:.2f}s', icon='✅')
  
  st.balloons()
      
  # Return the trained models and the test data
  return trained_models, X_test, y_test
  

#################
## Application ##
#################

st.title('Model Training')

with st.expander('Why is Model Training so important?'):
  st.write(model_training)

if (len(os.listdir('models')) <= 1):
  st.error('Models not yet selected!', icon='🚨')
  st.stop()

st.header('Test Size')
with st.expander('Want some help on choosing a test size?'):
  st.write(test_size_text)

models_list = get_all_models()
test_size = st.slider('Pick a test size for splitting the data', 0, 100)

if (st.button('Train Models')):
  trained_models, X_test, y_test = train_all_models(models_list, test_size)
  
  # Save the test data
  X_test.to_csv('data_test/test_features.csv', index=False)
  y_test.to_csv('data_test/test_target.csv', index=False)
  
  # Save the models
  for model in trained_models:
    model_name = type(model).__name__
    joblib.dump(model, f'models_trained/{model_name}.pkl')

